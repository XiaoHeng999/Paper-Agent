import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.agents.contracts import ResearchRequest
from src.graph.graph import build_graph, _build_initial_state
from src.graph.runtime import WorkflowCancellation
from src.llm.base import LLMResponse
from src.research.agents import ModelTasks, read_claims, verify
from src.research.context import ScopedEvidence
from src.research.models import EvidenceSpan, ResearchOptions, Claim, version_of
from src.research.storage import load_evidence
from src.research.workflow import run_research_node
from src.repositories.sessions.sqlite import SQLiteSessionRepository


class FakeModel:
    def __init__(self, qualified=False, malformed=False, delay=0):
        self.requests = []
        self.qualified = qualified
        self.malformed = malformed
        self.delay = delay
        self.signature = "test-v1"
        self.provider = self

    async def chat(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        self.requests.append(payload)
        if self.delay:
            await asyncio.sleep(self.delay)
        role = payload["role"]
        if self.malformed:
            return LLMResponse(content="bad json")
        if role == "paper_reader":
            value = {"claims": [{"statement": "在小样本下有效" if payload.get("feedback") else "有效",
                "kind": "result", "conditions": "小样本", "experiment": "数据集 D，基线 B，准确率",
                "evidence_ids": [payload["evidence"][0]["id"]]}]}
            if payload.get('feedback'):
                value = value['claims'][0]
        elif role == "argument_builder":
            value = {'arguments': []}
        elif role == "research_synthesis":
            value = {'arguments': [], 'insights': [], 'external_support': []}
        elif role == "reasoning_verifier":
            value = {'checks': []}
        elif role == "evidence_verifier":
            source = next(x for x in payload["evidence"] if x["id"] in payload["target"]["evidence_ids"])
            qualified = self.qualified and payload["target"].get("statement") == "有效"
            value = {"verdict": "qualified" if qualified else "verified", "reason": "原文仅支持小样本条件",
                "fact_citations": [{"clause": clause, "evidence_id": source["id"], "quote": source["text"]}
                    for clause in payload.get("checkable_clauses", [])],
                "repair_request": "补充小样本前提" if qualified else "",
                "evidence_ids": [x["id"] for x in payload["evidence"]],
                "fidelity": {"status": "supported", "reason": "原文支持", "evidence_ids": [x["id"] for x in payload["evidence"]]},
                "experimental_support": {"status": "unknown", "reason": "未独立复现", "evidence_ids": []}}
        elif role == "cross_paper_analyst":
            value = {"relations": [{"claim_ids": [x["id"] for x in payload["claims"]],
                "relation": "supports", "conditions": "相同数据集和基线", "rationale": "相同条件下结果相容",
                "evidence_ids": [x["id"] for x in payload["evidence"]]}]}
        elif role == "writer":
            value = {"paragraphs": [{"text": "在小样本条件下，现有证据支持有效性。", "claim_ids": [x["id"] for x in payload["claims"]], "relation_ids": [r["id"] for r in payload.get("relations", [])]}]}
        else:
            raise AssertionError(role)
        return LLMResponse(content=json.dumps(value, ensure_ascii=False), usage={"input_tokens": 10, "output_tokens": 5})


def item(paper_id="p1", abstract="Results apply to small samples."):
    return {"paper": {"id": paper_id, "paperId": paper_id, "title": f"Paper {paper_id}", "abstract": abstract}, "full_text": {}}


def state(model, papers=None, **constraints):
    return {"request": ResearchRequest("方法有效吗", {"output_mode": "paper_report", "cross_paper_enabled": False, "research_reasoning_enabled": False, **constraints}),
            "read_results": papers if papers is not None else [item()], "research_node_llm": model}


class ResearchWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_timeout_preserves_report_and_marks_verification_incomplete(self):
        class TimeoutModel(FakeModel):
            async def chat(self, messages, **kwargs):
                if json.loads(messages[-1]["content"])["role"] == "evidence_verifier":
                    raise TimeoutError()
                return await super().chat(messages, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            task = state(TimeoutModel())
            task.update(session_repo=repo, session_key=session.key, turn_id="timeout-case")
            result = await run_research_node()(task)
            check = result["research_report"]["cards"][0]["verifications"][0]
            self.assertEqual(check["verdict"], "insufficient_evidence")
            self.assertIn("超时", check["reason"])
            self.assertIn("本次语义核查未完成", check["unchecked"])
            self.assertTrue(result["research_markdown"])
            self.assertTrue(any(a["artifact_type"] == "research_state" for a in repo.get(session.key).artifacts))

    async def test_single_paper_verifies_without_cross_analysis(self):
        model = FakeModel()
        result = await run_research_node()(state(model))
        self.assertEqual([x["role"] for x in model.requests], ["paper_reader", "evidence_verifier"])
        self.assertFalse(result["research_report"]["cross_paper_executed"])
        self.assertIn("仅取得摘要", result["research_markdown"])
        self.assertEqual(result["research_report"]["cards"][0]["verifications"][0]["source_levels"], ["abstract"])
        card = result["research_report"]["cards"][0]
        self.assertTrue(card["reading_plan"]["subqueries"])
        self.assertTrue(card["claims"][0]["subquery_ids"])
        self.assertTrue(card["evidence_packets"])
        self.assertTrue(card["subquery_coverage"])
        self.assertEqual(model.requests[0]["reading_plan"]["paper_id"], "p1")
        supplied = {item["id"] for item in [*model.requests[0]["evidence"], *model.requests[0]["context_evidence"]]}
        for packet in model.requests[0]["evidence_packets"]:
            self.assertLessEqual(set(packet["evidence_ids"]), supplied)
            self.assertLessEqual({item["evidence_id"] for item in packet["candidate_trace"]}, supplied)

    async def test_many_papers_no_cross_context_leak(self):
        model = FakeModel()
        result = await run_research_node()(state(model, [item("p1", "UNIQUE_FIRST"), item("p2", "UNIQUE_SECOND")], output_mode="review"))
        self.assertEqual(len(result["research_report"]["cards"]), 2)
        for request in model.requests:
            self.assertIn(request["role"], {"paper_reader", "evidence_verifier"})
            self.assertEqual(len({x["paper_id"] for x in request["evidence"]}), 1)
        self.assertIn("报告汇编", result["research_markdown"])

    async def test_comparison_and_relation_verification(self):
        model = FakeModel()
        result = await run_research_node()(state(model, [item(), item("p2")], output_mode="comparison", cross_paper_enabled=True))
        report = result["research_report"]
        self.assertTrue(report["cross_paper_executed"])
        self.assertEqual(report["relations"][0]["relation"], "incomparable")
        self.assertTrue(report["relations"][0]["unresolved_questions"])
        self.assertEqual(report["verifications"][0]["scope"], "cross_paper")
        self.assertTrue(any(r["role"] == "cross_paper_analyst" for r in model.requests))

    async def test_single_comparison_degrades_without_fabricating_relations(self):
        model = FakeModel()
        result = await run_research_node()(state(model, output_mode="comparison", cross_paper_enabled=True))
        self.assertFalse(result["research_report"]["cross_paper_executed"])
        self.assertEqual(result["research_report"]["relations"], [])
        self.assertIn("不足两篇", result["research_markdown"])

    async def test_repair_reverifies_new_version(self):
        result = await run_research_node()(state(FakeModel(qualified=True)))
        card = result["research_report"]["cards"][0]
        self.assertEqual(card["claims"][0]["statement"], "在小样本下有效")
        self.assertEqual(card["claims"][0]["version"], card["verifications"][0]["target_version"])
        self.assertEqual(card["verifications"][0]["verdict"], "verified")

    async def test_missing_source_never_verified(self):
        model = FakeModel()
        result = await run_research_node()(state(model, [item(abstract="")]))
        self.assertEqual(model.requests, [])
        self.assertEqual(result["research_report"]["cards"][0]["verifications"], [])
        self.assertIn("没有可读取", result["research_markdown"])

    async def test_budget_delivers_explicit_partial_result(self):
        result = await run_research_node()(state(FakeModel(), research_max_calls=1))
        card = result["research_report"]["cards"][0]
        self.assertEqual(card["verifications"][0]["verdict"], "insufficient_evidence")
        self.assertEqual(result["research_report"]["calls"], 1)

    async def test_malformed_model_response_is_visible(self):
        result = await run_research_node()(state(FakeModel(malformed=True)))
        card = result["research_report"]["cards"][0]
        self.assertTrue(card["warnings"])
        self.assertTrue(card["omitted_evidence_ids"])

    async def test_cancellation_propagates(self):
        cancellation = WorkflowCancellation()
        cancellation.request()
        task_state = state(FakeModel())
        task_state["runtime_context"] = SimpleNamespace(cancellation=cancellation, sync_port=None)
        with self.assertRaises(asyncio.CancelledError):
            await run_research_node()(task_state)

    async def test_report_writer_is_only_used_when_cross_enabled(self):
        model = FakeModel()
        result = await run_research_node()(state(model, [item(), item("p2")], output_mode="review", cross_paper_enabled=True))
        self.assertTrue(result["research_report"]["paragraphs"])
        self.assertIn("综述正文", result["research_markdown"])
        self.assertIn("writer", [x["role"] for x in model.requests])

    async def test_persistence_reuse_and_source_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            model = FakeModel()
            task_state = state(model)
            task_state.update(session_repo=repo, session_key=session.key, turn_id="t1")
            first = await run_research_node()(task_state)
            second = await run_research_node()(task_state)
            self.assertEqual(second["research_report"]["calls"], 0)
            self.assertTrue(any(x["artifact_type"] == "paper_report" for x in repo.get(session.key).artifacts))
            task_state["read_results"] = [item(abstract="Changed source")]
            third = await run_research_node()(task_state)
            self.assertGreater(third["research_report"]["calls"], 0)
            self.assertNotEqual(first["research_report"]["cards"][0]["version"], third["research_report"]["cards"][0]["version"])

    async def test_graph_skips_old_analysis_and_writer_for_new_task(self):
        # Import the module explicitly because src.graph exports graph symbols.
        import importlib
        graph_module = importlib.import_module("src.graph.graph")
        read = AsyncMock(return_value={"read_results": [item()]})
        with patch.object(graph_module, "run_read_node", return_value=read):
            result = await graph_module.run_graph(ResearchRequest("read", {"papers": [{"id": "p1", "title": "P"}]}), state_overrides={"research_node_llm": FakeModel()})
        self.assertEqual(set(build_graph().nodes) - {"__start__"}, {"run_search_agent", "run_read", "run_research", "compose_reply"})
        self.assertIn("核心论证与研究见解", result.state["assistant_message"])


class ContextTest(unittest.TestCase):
    def test_scope_rejects_other_papers_and_unknown_ids(self):
        a = EvidenceSpan(id="a", paper_id="p1", version="v", text="one", source_level="full_text")
        b = EvidenceSpan(id="b", paper_id="p2", version="v", text="two", source_level="full_text")
        scope = ScopedEvidence([a, b], {"p1"})
        with self.assertRaises(ValueError):
            scope.read(["b"])
        self.assertEqual(scope.read(["a"]), [a])

    def test_context_batches_preserve_every_character(self):
        evidence = [EvidenceSpan(id=str(i), paper_id="p", version="v", text="x" * 3000, source_level="full_text") for i in range(4)]
        batches = list(ScopedEvidence(evidence, {"p"}).batches(4000))
        self.assertEqual(sum(len(x.text) for batch in batches for x in batch), 12000)

    def test_default_and_strict_options(self):
        self.assertEqual(ResearchOptions.from_constraints({}), ResearchOptions())
        self.assertFalse(ResearchOptions.from_constraints({"output_mode": "comparison"}).cross_paper_enabled)
        for value in ("false", 0, 1):
            with self.assertRaises(ValueError):
                ResearchOptions.from_constraints({"cross_paper_enabled": value})

    def test_fulltext_evidence_keeps_pages_and_rejects_foreign_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "chunk.json").write_text(json.dumps({"chunks": [
                {"chunkId": "p1:p1", "paperId": "p1", "content": "x" * 9000, "page_start": 2},
                {"chunkId": "p2:p1", "paperId": "p2", "content": "foreign"}]}))
            value = item()
            value["full_text"] = {"markdown_path": str(path / "full.md")}
            spans = load_evidence(value)
            self.assertEqual(sum(len(x.text) for x in spans), 9000)
            self.assertTrue(all(x.page_start == 2 and x.paper_id == "p1" for x in spans))


class ResearchBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_reader_rejects_unknown_and_unseen_evidence(self):
        class InvalidReader(FakeModel):
            def __init__(self, evidence_id):
                super().__init__()
                self.evidence_id = evidence_id

            async def chat(self, messages, **kwargs):
                return LLMResponse(content=json.dumps({"claims": [
                    {"statement": "claim", "evidence_ids": [self.evidence_id]}
                ]}))

        spans = [EvidenceSpan(id=name, paper_id="p", version="v", text="raw", source_level="full_text")
                 for name in ("provided", "unseen")]
        for evidence_id in ("missing", "unseen"):
            with self.subTest(evidence_id=evidence_id), self.assertRaises(ValueError):
                await read_claims(ModelTasks(InvalidReader(evidence_id), ResearchOptions()),
                                  "question", "p", ScopedEvidence(spans, {"p"}), ["provided"])

    def test_prepared_source_resume_requires_existing_chunks(self):
        from src.graph.read_node import _deep_read_needs_processing
        from src.models.read_models import PaperReadResult, FullTextStatus
        from src.paper_retrieval.models import PaperDocument
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            path.write_text("raw")
            result = PaperReadResult(paper=PaperDocument(id="p", title="P"),
                                     full_text=FullTextStatus(status="chunks_saved", markdown_path=str(path)))
            self.assertTrue(_deep_read_needs_processing(result))
            (path.parent / "chunk.json").write_text("{}")
            self.assertTrue(_deep_read_needs_processing(result))
            from src.utils.read_utils.chunkers import build_chunks_file
            build_chunks_file(result.paper, markdown_path=path)
            self.assertFalse(_deep_read_needs_processing(result))
            path.unlink()
            self.assertTrue(_deep_read_needs_processing(result))

    async def test_unavailable_verifier_is_not_a_pass(self):
        span = EvidenceSpan(id="e", paper_id="p", version="v", text="raw", source_level="full_text")
        claim = Claim(id="c", version="v", paper_id="p", statement="claim", evidence_ids=["e"])
        checked = await verify(ModelTasks(None, ResearchOptions()), claim, ScopedEvidence([span], {"p"}))
        self.assertEqual(checked.verdict, "insufficient_evidence")

    async def test_verifier_cannot_cite_an_unseen_source(self):
        class BadModel(FakeModel):
            async def chat(self, messages, **kwargs):
                return LLMResponse(content=json.dumps({"verdict": "verified", "reason": "ok", "evidence_ids": ["other-paper"]}))
        span = EvidenceSpan(id="e", paper_id="p", version="v", text="raw", source_level="full_text")
        claim = Claim(id="c", version="v", paper_id="p", statement="claim", evidence_ids=["e"])
        checked = await verify(ModelTasks(BadModel(), ResearchOptions()), claim, ScopedEvidence([span], {"p"}))
        self.assertEqual(checked.verdict, "insufficient_evidence")

    async def test_expired_deadline_does_not_call_model(self):
        tasks = ModelTasks(FakeModel(), ResearchOptions())
        tasks.deadline = 0
        span = EvidenceSpan(id="e", paper_id="p", version="v", text="raw", source_level="full_text")
        claim = Claim(id="c", version="v", paper_id="p", statement="claim", evidence_ids=["e"])
        checked = await verify(tasks, claim, ScopedEvidence([span], {"p"}))
        self.assertEqual(tasks.calls, 0)
        self.assertEqual(checked.verdict, "insufficient_evidence")

    async def test_no_cross_deep_read_selection(self):
        from src.graph.read_node import _assign_deep_read_selection
        from src.models.read_models import PaperReadResult
        from src.paper_retrieval.models import PaperDocument
        papers = [PaperDocument(id=f"p{i}", title=f"Paper {i}") for i in range(3)]
        results = {paper.id: PaperReadResult(paper=paper) for paper in papers}
        with patch("src.graph.read_node._select_deep_read_by_llm", new_callable=AsyncMock) as choose:
            selected = await _assign_deep_read_selection(papers, results, {}, 1, topic="read", llm=None, allow_cross_selection=False)
        choose.assert_not_called()
        self.assertEqual(selected, {"p0"})

    async def test_research_checkpoint_skips_search_and_read(self):
        import importlib
        module = importlib.import_module("src.graph.graph")
        checkpoint = {"phase": "research", "request": {"topic": "read", "constraints": {"output_mode": "paper_report"}}, "read_results": [item()]}
        never = AsyncMock(side_effect=AssertionError("upstream called"))
        with patch.object(module, "run_search_agent_node", return_value=never), patch.object(module, "run_read_node", return_value=never):
            result = await module.run_graph(ResearchRequest("placeholder"), state_overrides={"read_resume_checkpoint": checkpoint, "research_node_llm": FakeModel()})
        never.assert_not_called()
        self.assertTrue(result.state["research_report"]["cards"])

    async def test_reader_does_not_receive_supplied_paper_list(self):
        from src.agents.readAgent import ReadAgent
        from src.agents.base import AgentContext
        from src.paper_retrieval.models import PaperDocument
        reader = ReadAgent(AgentContext())
        messages = reader._abstract_messages(PaperDocument(id="p1", title="First", abstract="A"), "read", {"papers": [{"title": "SECRET_OTHER_PAPER"}], "output_mode": "paper_report"})
        self.assertNotIn("SECRET_OTHER_PAPER", json.dumps(messages))


class ResearchApiTest(unittest.TestCase):
    def test_invalid_task_is_rejected_before_session_starts(self):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            client = TestClient(create_app(sessions_repo=repo))
            response = client.post(f"/api/sessions/{session.key}/runs", json={"content": "read", "constraints": {"cross_paper_enabled": "false"}})
            self.assertEqual(response.status_code, 400)
            self.assertFalse(repo.get(session.key).run_started_at)
            self.assertFalse(repo.get(session.key).messages)

    def test_synchronous_api_outputs_downloadable_independent_report(self):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        import importlib
        module = importlib.import_module("src.graph.graph")
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            fake = FakeModel()
            async def close():
                pass
            fake.aclose = close
            with patch.object(module, "run_read_node", return_value=AsyncMock(return_value={"read_results": [item()]})), patch("src.research.workflow.load_read_agent_llm", return_value=fake):
                client = TestClient(create_app(sessions_repo=repo))
                response = client.post(f"/api/sessions/{session.key}/messages", json={"content": "read", "constraints": {"output_mode": "paper_report", "papers": [{"id": "p1", "title": "First"}]}})
            self.assertEqual(response.status_code, 200)
            artifacts = repo.get(session.key).artifacts
            final = next(x for x in artifacts if x["artifact_type"] == "final_report")
            downloaded = client.get(f"/api/sessions/{session.key}/artifacts/{final['id']}")
            self.assertEqual(downloaded.status_code, 200)
            self.assertIn("核心论证与研究见解", downloaded.text)
            self.assertTrue(any(x["artifact_type"] == "research_checkpoint" for x in artifacts))


if __name__ == "__main__":
    unittest.main()
