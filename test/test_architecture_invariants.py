import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.agents.contracts import ResearchRequest
from src.graph.graph import run_graph
from src.llm.base import LLMResponse
from src.paper_retrieval.models import PaperDocument
from src.research.agents import ModelTasks, verify
from src.research.context import ScopedEvidence
from src.research.models import AnalysisRelation, EvidenceSpan, ResearchOptions, validate_checkpoint_options
from src.research.storage import load_evidence
from src.research.workflow import run_research_node
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.session_runs import SessionRunService
from src.utils.read_utils.cache import safe_cache_name
from src.utils.read_utils.chunkers import build_chunks_file, chunks_are_current
from test.test_research_workflow import FakeModel, item, state


class SourceIntegrityTest(unittest.TestCase):
    def test_preserves_abstract_references_appendix_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            path.write_text("Title before pages\n<!-- page: 1 -->\nAbstract\nBoundary condition\nIntroduction\nMethod\n<!-- page: 2 -->\nReferences\nCitation\nAppendix\nCounterexample")
            built = build_chunks_file(PaperDocument(id="p", title="P"), markdown_path=path)
            combined = "\n".join(c.content for c in built.chunks)
            for text in ("Title before pages", "Boundary condition", "Citation", "Counterexample"):
                self.assertIn(text, combined)
            self.assertEqual(built.chunks[-1].page_start, 2)

    def test_pdf_converter_uses_parser_without_discarding_appendix(self):
        from src.utils.read_utils.read_fulltext import _convert_pdf
        pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in
                 ("Abstract\nMain claim", "", "References\nCitation\nAppendix\nCounterexample")]
        with tempfile.TemporaryDirectory() as directory, patch("pypdf.PdfReader", return_value=SimpleNamespace(pages=pages)):
            path = Path(directory) / "paper.md"
            result = _convert_pdf(PaperDocument(id="p", title="P"), path.with_suffix(".pdf"), None, path)
            self.assertEqual(result.page_count, 3)
            self.assertIn("<!-- page: 3 -->", path.read_text())
            self.assertIn("Counterexample", path.read_text())

    def test_old_cache_and_changed_source_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            paper = PaperDocument(id="p", title="P")
            path.write_text("new evidence")
            (path.parent / "chunk.json").write_text(json.dumps({"chunks": [{"chunkId": "p:old", "paperId": "p", "content": "stale"}]}))
            first = build_chunks_file(paper, markdown_path=path)
            self.assertEqual(first.chunks[0].content, "new evidence")
            self.assertTrue(chunks_are_current(paper, path))
            path.write_text("revised evidence")
            self.assertFalse(chunks_are_current(paper, path))
            value = {"paper": paper.to_dict(), "full_text": {"markdown_path": str(path)}}
            self.assertEqual(load_evidence(value)[0].text, "revised evidence")

    def test_normalized_ids_cannot_collide_or_escape_cache(self):
        values = ["a/b", "a:b", "a_b", ".", "..", "x" * 161, "x" * 162]
        self.assertEqual(len({safe_cache_name(v) for v in values}), len(values))
        for value in values:
            self.assertNotIn(safe_cache_name(value), {".", ".."})
            self.assertEqual(Path(safe_cache_name(value)).name, safe_cache_name(value))

    def test_read_artifacts_for_normalized_ids_do_not_overwrite(self):
        from src.repositories.node_persistence.read_persistence import ReadPersistenceSink
        from src.models.read_models import PaperReadResult
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            sink = ReadPersistenceSink(repo, session_key=session.key, turn_id="turn")
            first = sink.persist_paper(PaperReadResult(paper=PaperDocument(id="a/b", title="First")))
            second = sink.persist_paper(PaperReadResult(paper=PaperDocument(id="a:b", title="Second")))
            self.assertNotEqual(first.artifacts[0]["path"], second.artifacts[0]["path"])
            content = Path(first.artifacts[0]["path"]).read_text()
            self.assertEqual(json.loads(content)["paper"]["title"], "First")

    def test_rejects_conflicting_evidence_identity(self):
        a = EvidenceSpan(id="e", paper_id="p", version="v", text="one", source_level="full_text")
        with self.assertRaises(ValueError):
            ScopedEvidence([a, a.model_copy(update={"text": "different"})], {"p"})

    def test_restored_chunk_count_is_not_a_score(self):
        from src.graph.read_node import _full_text_from_payload
        self.assertEqual(_full_text_from_payload({"chunk_count": 400}).chunk_count, 400)
        self.assertEqual(_full_text_from_payload({"chunk_count": -1}).chunk_count, 0)
        self.assertEqual(_full_text_from_payload({"chunk_count": "invalid"}).chunk_count, 0)

    def test_duplicate_supplied_paper_identity_rejected(self):
        with self.assertRaises(ValueError):
            ResearchOptions.from_constraints({"papers": [{"id": "one", "title": "A", "doi": "same"}, {"id": "two", "title": "B", "doi": "same"}]})


class WorkflowIntegrityTest(unittest.IsolatedAsyncioTestCase):
    async def test_cross_verifier_requires_both_papers(self):
        spans = [EvidenceSpan(id=p, paper_id=p, version="v", text="evidence", source_level="full_text") for p in ("p1", "p2")]
        target = AnalysisRelation(id="r", version="v", claim_ids=["c1", "c2"], relation="supports", conditions="same", rationale="match", evidence_ids=["p1", "p2"])
        class OneSided(FakeModel):
            async def chat(self, messages, **kwargs):
                return LLMResponse(content=json.dumps({"verdict": "verified", "reason": "one side", "evidence_ids": ["p1"]}))
        result = await verify(ModelTasks(OneSided(), ResearchOptions()), target, ScopedEvidence(spans, {"p1", "p2"}))
        self.assertEqual(result.verdict, "insufficient_evidence")

    async def test_rejected_writer_prose_is_not_published_as_review(self):
        class RejectedWriter(FakeModel):
            async def chat(self, messages, **kwargs):
                payload = json.loads(messages[-1]["content"])
                if payload["role"] == "evidence_verifier" and "claim_ids" in payload["target"] and "statement" in payload["target"]:
                    return LLMResponse(content=json.dumps({"verdict": "contradicted", "reason": "not supported", "repair_request": "fix", "evidence_ids": [e["id"] for e in payload["evidence"]]}))
                return await super().chat(messages, **kwargs)
        task = state(RejectedWriter(), [item("p1"), item("p2")], output_mode="review", cross_paper_enabled=True)
        result = await run_research_node()(task)
        self.assertNotIn("# 综述正文", result["research_markdown"])
        self.assertIn("# 未通过核查的写作草稿", result["research_markdown"])
        self.assertTrue(result["research_report"]["writing_history"])

    async def test_invalid_resume_is_rejected_before_session_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            service = SessionRunService(repo)
            checkpoint = {"request": {"topic": "Q", "constraints": {"cross_paper_enabled": "false"}}}
            with self.assertRaises(ValueError):
                await service.start_run(session.key, {"read_resume_checkpoint": checkpoint})
            record = repo.get(session.key)
            self.assertFalse(record.messages)
            self.assertNotEqual(record.status, "running")

    async def test_resume_cannot_silently_override_no_cross(self):
        checkpoint = {"phase": "research", "request": {"topic": "Q", "constraints": {"cross_paper_enabled": True}}}
        with self.assertRaises(ValueError):
            await run_graph(ResearchRequest("Q", {"cross_paper_enabled": False}), state_overrides={"read_resume_checkpoint": checkpoint, "research_node_llm": FakeModel()})

    async def test_async_pdf_source_change_invalidates_markdown_and_chunks(self):
        from src.utils.read_utils.read_fulltext import async_convert_fulltext_to_markdown
        from src.utils.read_utils.pdf_parsers import PdfParseResult, ParsedPdfPage
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "paper.pdf"
            source.write_bytes(b"version one")
            paper = PaperDocument(id="p", title="P")
            parser = SimpleNamespace(parse=lambda path: PdfParseResult(pages=[ParsedPdfPage(page_number=1, text=path.read_text())]))
            with patch("src.utils.read_utils.read_fulltext.get_pdf_parser", return_value=parser):
                first = await async_convert_fulltext_to_markdown(paper, source_path=source, source_url=None)
                value = {"paper": paper.to_dict(), "full_text": {"source_path": str(source), "markdown_path": str(first.markdown_path)}}
                evidence1 = await asyncio.to_thread(load_evidence, value)
                source.write_bytes(b"version two")
                evidence2 = await asyncio.to_thread(load_evidence, value)
            self.assertIn("version one", evidence1[0].text)
            self.assertIn("version two", evidence2[0].text)
            self.assertNotEqual(evidence1[0].version, evidence2[0].version)

    async def test_bad_source_does_not_discard_other_paper_reports(self):
        from src.research.storage import load_evidence as original
        def read(value):
            if value["paper"]["id"] == "p1":
                raise ValueError("broken source")
            return original(value)
        with patch("src.research.workflow.load_evidence", side_effect=read):
            result = await run_research_node()(state(FakeModel(), [item("p1"), item("p2")]))
        cards = result["research_report"]["cards"]
        self.assertFalse(cards[0]["claims"])
        self.assertIn("broken source", " ".join(cards[0]["warnings"]))
        self.assertTrue(cards[1]["claims"])

    async def test_explicit_papers_with_same_title_are_not_deduplicated(self):
        import importlib
        module = importlib.import_module("src.graph.read_node")
        papers = [PaperDocument(id="p1", title="Same"), PaperDocument(id="p2", title="Same")]
        request = ResearchRequest("Q", {"papers": [p.to_dict() for p in papers]})
        process = AsyncMock(return_value=0)
        with patch.object(module, "_process_papers_concurrently", process):
            await module.run_read_node()({"request": request, "search_results": papers, "read_node_llm": None})
        self.assertEqual([p.id for p in process.call_args.kwargs["papers"]], ["p1", "p2"])

    async def test_read_owned_provider_closes_on_failure(self):
        import importlib
        module = importlib.import_module("src.graph.read_node")
        model = SimpleNamespace(aclose=AsyncMock())
        with patch.object(module, "_resolve_llm", return_value=model), patch.object(module, "_run_read_node_async", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                await module.run_read_node()({})
        model.aclose.assert_awaited_once()

    async def test_search_owned_provider_closes_on_failure(self):
        import importlib
        module = importlib.import_module("src.graph.search_node")
        model = SimpleNamespace(aclose=AsyncMock())
        with patch.object(module, "load_search_agent_llm", return_value=model), patch.object(module.SearchAgent, "async_run", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                await module.run_search_agent_node()({"request": ResearchRequest("Q")})
        model.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
