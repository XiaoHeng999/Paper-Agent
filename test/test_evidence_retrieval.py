import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from src.llm.base import EmbeddingResponse
from src.research.context import ScopedEvidence
from src.research.models import EvidenceSpan
from src.research.retrieval import EvidenceIndex, LocalE5Index
from src.research.workflow import run_research_node
from test.test_research_workflow import FakeModel, state, item


class Embeddings:
    def __init__(self):
        self.calls = []

    async def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        return EmbeddingResponse(embeddings=[[1.0, 0.0] if "small" in text else [0.0, 1.0] for text in texts])


def span(name, paper, version="v"):
    return EvidenceSpan(id=name, paper_id=paper, version=version, text="small sample " + version, source_level="full_text")


class EvidenceRetrievalTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.provider = Embeddings()
        self.snapshot = SimpleNamespace(provider=self.provider, signature="embedding-v1")
        self.index = EvidenceIndex(self.snapshot, self.directory.name, "test_papers")

    async def asyncTearDown(self):
        self.directory.cleanup()

    async def test_local_model_uses_distinct_passage_and_query_modes(self):
        with patch.dict("os.environ", {"PAPER_AGENT_TEXT_E5_DIR": self.directory.name}):
            index = EvidenceIndex.from_config()
        self.assertIsInstance(index, LocalE5Index)
        index.path = self.directory.name
        source = span("a", "p1")
        with patch.object(index, "_encode", return_value=([[1.0, 0.0]], 3)) as encode:
            await index.prepare([source])
            self.assertEqual(await index.search("question", ScopedEvidence([source], {"p1"})), [source])
            self.assertEqual([call.args for call in encode.call_args_list],
                             [([source.text], False), (["question"], True)])

    async def test_persistent_index_reuse_and_single_paper_scope(self):
        a, b = span("a", "p1"), span("b", "p2")
        await self.index.prepare([a])
        await self.index.prepare([b])
        reopened = EvidenceIndex(self.snapshot, self.directory.name, "test_papers")
        before = len(self.provider.calls)
        await reopened.prepare([a])
        self.assertEqual(len(self.provider.calls), before)
        self.assertEqual(await reopened.search("small", ScopedEvidence([a, b], {"p1"})), [a])
        self.assertEqual(await reopened.search("small", ScopedEvidence([a, b], set())), [])

    async def test_revision_and_embedding_model_are_isolated(self):
        old, new = span("a", "p1"), span("a", "p1", "v2")
        await self.index.prepare([old])
        await self.index.prepare([new])
        self.assertEqual(await self.index.search("small", ScopedEvidence([new], {"p1"})), [new])
        other = EvidenceIndex(SimpleNamespace(provider=self.provider, signature="embedding-v2"), self.directory.name, "test_papers")
        self.assertNotEqual(other.collection, self.index.collection)
        await other.prepare([new])
        self.assertEqual(other.indexed, 1)

    async def test_untrusted_database_hit_cannot_cross_scope(self):
        a = span("a", "p1")
        await self.index.prepare([a])
        hits = [SimpleNamespace(metadata={"evidence_id": "foreign", "paper_id": "p2"}, distance=0),
                SimpleNamespace(metadata={"evidence_id": "a", "version": "old", "corpus": self.index.key([a])}, distance=0)]
        with patch.object(self.index, "_store", return_value=hits):
            self.assertEqual(await self.index.search("small", ScopedEvidence([a], {"p1"})), [])

    async def test_workflow_indexes_and_retrieves_without_cross_context(self):
        model = FakeModel()
        task = state(model)
        task.update(read_results=[item("p1"), item("p2")], research_evidence_index=self.index)
        result = await run_research_node()(task)
        self.assertEqual(result["research_report"]["retrieval"]["status"], "ready")
        self.assertGreater(result["research_report"]["retrieval"]["queries"], 0)
        self.assertFalse(result["research_report"]["cross_paper_executed"])
        for request in model.requests:
            self.assertLessEqual(len({x["paper_id"] for x in request.get("evidence", [])}), 1)
        self.assertNotIn("cross_paper_analyst", [r["role"] for r in model.requests])

    async def test_cross_paper_retrieval_only_when_enabled(self):
        model = FakeModel()
        task = state(model)
        task["request"].constraints.update(cross_paper_enabled=True, output_mode="comparison")
        task.update(read_results=[item("p1"), item("p2")], research_evidence_index=self.index)
        result = await run_research_node()(task)
        self.assertTrue(result["research_report"]["cross_paper_executed"])
        self.assertTrue(result["research_report"]["relations"])

    async def test_embedding_failure_preserves_report_and_records_gap(self):
        async def fail(*args, **kwargs):
            raise RuntimeError("offline")
        self.provider.embed = fail
        task = state(FakeModel())
        task["research_evidence_index"] = self.index
        result = await run_research_node()(task)
        self.assertTrue(result["research_report"]["cards"][0]["claims"])
        self.assertEqual(result["research_report"]["retrieval"]["status"], "unavailable")
        self.assertTrue(result["research_report"]["retrieval"]["warnings"])

    async def test_fulltext_workflow_auto_initializes_and_closes_index(self):
        paper_dir = Path(self.directory.name) / "paper"
        paper_dir.mkdir()
        markdown = paper_dir / "paper.md"
        markdown.write_text("small samples")
        import json
        (paper_dir / "chunk.json").write_text(json.dumps({"chunks": [
            {"chunkId": "p1:1", "paperId": "p1", "content": "small samples", "page_start": 1}
        ]}))
        task = state(FakeModel())
        task["read_results"][0]["full_text"] = {"markdown_path": str(markdown)}
        self.index.aclose = AsyncMock()
        with patch("src.research.workflow.EvidenceIndex.from_config", return_value=self.index) as factory:
            result = await run_research_node()(task)
        factory.assert_called_once()
        self.index.aclose.assert_awaited_once()
        self.assertEqual(result["research_report"]["retrieval"]["status"], "ready")
        self.assertEqual(result["research_report"]["evidence"][0]["source_level"], "full_text")
        self.assertIn("论文来源", result["research_markdown"])

    def test_restored_search_tool_and_scope_registry(self):
        from src.agents import build_paper_search_tool, ToolRegistry
        from src.paper_retrieval.models import PaperDocument
        from unittest.mock import Mock
        service = Mock()
        service.search.return_value = SimpleNamespace(papers=[PaperDocument(id="p", title="P")])
        tool = build_paper_search_tool(service)
        registry = ToolRegistry()
        registry.register(tool)
        self.assertEqual(registry.select(("paper_search",)).require("paper_search").call(query="topic")[0]["id"], "p")
        service.search.assert_called_once()
        with self.assertRaises(ValueError):
            registry.select(()).require("paper_search")

    async def test_reader_prioritization_preserves_all_original_spans(self):
        a, b, c = [span(name, "p") for name in "abc"]
        scope = ScopedEvidence([a, b, c], {"p"})
        self.assertEqual([s.id for batch in scope.batches(4000, ["c"]) for s in batch], ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
