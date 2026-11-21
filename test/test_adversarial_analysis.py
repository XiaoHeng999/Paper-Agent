import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse
from src.paper_retrieval.models import PaperDocument, SearchResponse
from src.research.agents import ModelTasks, compare, verify
from src.research.assessment import assess_card, comparison_candidate
from src.research.context import ScopedEvidence
from src.research.models import AXES, DIMENSIONS, Claim, EvidenceSpan, PaperCard, ResearchOptions, VerificationResult
from src.research.supplement import supplement_sources
from src.research.workflow import run_research_node
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.research import latest_report, revision_request
from test.test_research_workflow import FakeModel, item, state


def evidence(name="p1", level="full_text", text="A controlled experiment"):
    return EvidenceSpan(id=name, paper_id=name, version="v", text=text, source_level=level)


def claim(name="p1", kind="result", **kwargs):
    return Claim(id="c" + name, version="v", paper_id=name, statement="effective", kind=kind, evidence_ids=[name], **kwargs)


class RelationModel(FakeModel):
    def __init__(self, relation="conflicts", changed_axis=None, unknown_axis=None):
        super().__init__()
        self.relation, self.changed_axis, self.unknown_axis = relation, changed_axis, unknown_axis

    async def chat(self, messages, **kwargs):
        data = json.loads(messages[-1]["content"])
        return LLMResponse(content=json.dumps({"relations": [{"claim_ids": [c["id"] for c in data["claims"]],
            "relation": self.relation, "conditions": "conditions", "rationale": "compared",
            "evidence_ids": [e["id"] for e in data["evidence"]],
            "axes": [{"axis": axis, "left": "A", "right": "B" if axis == self.changed_axis else "A",
                      "status": "unknown" if axis == self.unknown_axis else "different" if axis == self.changed_axis else "matched",
                      "evidence_ids": [e["id"] for e in data["evidence"]]} for axis in AXES]}]}))


class AdversarialTest(unittest.IsolatedAsyncioTestCase):
    async def test_internal_counterexample_is_read_without_vector_service(self):
        spans = [evidence(text="Our selected result"), EvidenceSpan(id="far", paper_id="p1", version="v", text="Limitations: fails on small samples", source_level="full_text")]
        model = FakeModel()
        await verify(ModelTasks(model, ResearchOptions()), claim(), ScopedEvidence(spans, {"p1"}))
        self.assertIn("far", [e["id"] for e in model.requests[0]["evidence"]])

    async def test_different_budget_is_not_a_conflict(self):
        result = await compare(ModelTasks(RelationModel(changed_axis="budget"), ResearchOptions()), "Q", [claim(), claim("p2")], ScopedEvidence([evidence(), evidence("p2")], {"p1", "p2"}))
        self.assertEqual(result[0].relation, "different_scope")

    async def test_different_questions_are_incomparable(self):
        result = await compare(ModelTasks(RelationModel(changed_axis="question"), ResearchOptions()), "Q", [claim(), claim("p2")], ScopedEvidence([evidence(), evidence("p2")], {"p1", "p2"}))
        self.assertEqual(result[0].relation, "incomparable")

    async def test_missing_metric_is_incomparable(self):
        result = await compare(ModelTasks(RelationModel(unknown_axis="metric"), ResearchOptions()), "Q", [claim(), claim("p2")], ScopedEvidence([evidence(), evidence("p2")], {"p1", "p2"}))
        self.assertEqual(result[0].relation, "incomparable")
        self.assertTrue(result[0].unresolved_questions)

    async def test_aligned_fulltext_can_preserve_conflict(self):
        result = await compare(ModelTasks(RelationModel(), ResearchOptions()), "Q", [claim(), claim("p2")], ScopedEvidence([evidence(), evidence("p2")], {"p1", "p2"}))
        self.assertEqual(result[0].relation, "conflicts")

    async def test_similar_words_do_not_establish_equivalence(self):
        result = await compare(ModelTasks(RelationModel("equivalent", changed_axis="method"), ResearchOptions()), "Q", [claim(), claim("p2")], ScopedEvidence([evidence(), evidence("p2")], {"p1", "p2"}))
        self.assertEqual(result[0].relation, "different_scope")

    async def test_abstract_cannot_certify_experimental_conflict(self):
        result = await compare(ModelTasks(RelationModel(), ResearchOptions()), "Q", [claim(), claim("p2")], ScopedEvidence([evidence(level="abstract"), evidence("p2")], {"p1", "p2"}))
        self.assertEqual(result[0].relation, "incomparable")

    async def test_author_report_is_not_experimental_proof(self):
        model = FakeModel()
        checked = await verify(ModelTasks(model, ResearchOptions()), claim(), ScopedEvidence([evidence()], {"p1"}))
        self.assertEqual(checked.fidelity.status, "supported")
        self.assertEqual(checked.experimental_support.status, "unknown")
        inferred = await verify(ModelTasks(model, ResearchOptions()), claim(attribution="system_inference"), ScopedEvidence([evidence()], {"p1"}))
        self.assertEqual(inferred.verdict, "insufficient_evidence")

    async def test_scoring_cannot_cite_unseen_sources(self):
        class BadScore(FakeModel):
            async def chat(self, messages, **kwargs):
                result = await super().chat(messages, **kwargs)
                value = json.loads(result.content)
                value["dimensions"] = [{"dimension": "reproducibility", "score": 4, "reason": "code", "evidence_ids": ["foreign"]}]
                return LLMResponse(content=json.dumps(value))
        checked = await verify(ModelTasks(BadScore(), ResearchOptions()), claim(), ScopedEvidence([evidence()], {"p1"}))
        self.assertEqual(checked.verdict, "insufficient_evidence")

    def test_result_and_limitation_are_comparison_candidates(self):
        self.assertTrue(comparison_candidate(claim(), claim("p2", kind="limitation")))

    def test_weights_require_goal_and_unknown_is_not_zero(self):
        with self.assertRaises(ValueError):
            ResearchOptions.from_constraints({"score_weights": {"reproducibility": 1}})
        for weights in ({"unknown_dimension": 1}, {"scope_clarity": float("nan")}, {"scope_clarity": True}):
            with self.assertRaises(ValueError):
                ResearchOptions.from_constraints({"scoring_goal": "reproduce", "score_weights": weights})
        options = ResearchOptions.from_constraints({"scoring_goal": "reproduce", "score_weights": {"reproducibility": 1}})
        assessment = assess_card(PaperCard(paper_id="p", title="P", version="v"), options)
        self.assertIsNone(assessment["weighted_score"])
        self.assertTrue(all(d["score"] is None for d in assessment["dimensions"]))

    def test_weighted_score_uses_only_explicit_weights(self):
        c = claim()
        checked = VerificationResult(target_id=c.id, target_version=c.version, scope="paper", source_levels=["full_text"], verdict="verified", reason="supported", evidence_ids=["p1"],
            fidelity={"status": "supported", "reason": "source", "evidence_ids": ["p1"]},
            dimensions=[{"dimension": "reproducibility", "score": 4, "reason": "code/data", "evidence_ids": ["p1"]}, {"dimension": "scope_clarity", "score": 2, "reason": "partial", "evidence_ids": ["p1"]}])
        card = PaperCard(paper_id="p1", title="P", version="v", claims=[c], verifications=[checked])
        options = ResearchOptions.from_constraints({"scoring_goal": "reproduce", "score_weights": {"reproducibility": 3, "scope_clarity": 1}})
        self.assertEqual(assess_card(card, options)["weighted_score"], 3.5)
        self.assertIsNone(assess_card(card, ResearchOptions())["weighted_score"])

    async def test_external_search_disabled_by_default(self):
        service = SimpleNamespace(async_search=AsyncMock())
        task = state(FakeModel())
        task["research_search_service"] = service
        _, audit = await supplement_sources(task, [item()], ModelTasks(FakeModel(), ResearchOptions()))
        service.async_search.assert_not_awaited()
        self.assertFalse(audit["attempted"])
        self.assertTrue(ResearchOptions.from_constraints({"allow_external_search": True}).allow_external_search)

    async def test_external_search_is_bounded_and_resume_does_not_repeat(self):
        service = SimpleNamespace(async_search=AsyncMock(return_value=SearchResponse(query="Q", papers=[PaperDocument(id=f"p{i}", title=f"P{i}", abstract="Abstract") for i in range(2, 6)])))
        options = ResearchOptions.from_constraints({"cross_paper_enabled": True, "allow_external_search": True, "external_paper_limit": 2})
        task = state(FakeModel(), cross_paper_enabled=True, allow_external_search=True)
        task["research_search_service"] = service
        task['research_challenges'] = [{'claim_id':'c', 'statement':'effect', 'premise':'matched budget', 'required_test':'independent ablation'}]
        with patch("src.research.supplement.async_download_paper_fulltext", AsyncMock(return_value=SimpleNamespace(file_path=None))):
            items, audit = await supplement_sources(task, [item()], ModelTasks(FakeModel(), options))
        self.assertEqual(len(items), 3)
        task["read_resume_checkpoint"] = {"external_search": audit}
        await supplement_sources(task, items, ModelTasks(FakeModel(), options))
        self.assertEqual(service.async_search.await_count, 1)

    async def test_fulltext_workflow_delivers_dimension_scores(self):
        class Scoring(FakeModel):
            async def chat(self, messages, **kwargs):
                result = await super().chat(messages, **kwargs)
                payload = json.loads(messages[-1]["content"])
                if payload["role"] == "evidence_verifier":
                    value = json.loads(result.content)
                    value["dimensions"] = [{"dimension": key, "score": 3, "reason": "有明确原文依据", "evidence_ids": [e["id"] for e in payload["evidence"]]} for key in DIMENSIONS]
                    result.content = json.dumps(value)
                return result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            path.write_text("Methods and results. Code and data are specified.")
            fulltext = item()
            fulltext["full_text"] = {"markdown_path": str(path)}
            with patch("src.research.workflow.EvidenceIndex.from_config", return_value=None):
                result = await run_research_node()(state(Scoring(), [fulltext], scoring_goal="prioritize reproduction", score_weights={"reproducibility": 1}))
            self.assertEqual(result["research_report"]["assessments"][0]["weighted_score"], 3)
        checked = await verify(ModelTasks(Scoring(), ResearchOptions()), claim(), ScopedEvidence([evidence(level="abstract")], {"p1"}))
        self.assertTrue(all(d.score is None for d in checked.dimensions))

    async def test_revision_is_versioned_reverified_and_scope_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            task = state(FakeModel(), [item(), item("p2")], cross_paper_enabled=True)
            task.update(session_repo=repo, session_key=session.key, turn_id="one")
            first = await run_research_node()(task)
            latest = latest_report(repo, session.key)
            old = first["research_report"]["cards"][0]["claims"][0]
            body = {"artifact_id": latest["artifact_id"], "claim_id": old["id"], "target_version": old["version"], "reason": "缩小适用范围", "changes": {"statement": "仅在指定实验条件下有效"}}
            request = revision_request(repo, session.key, body)
            checkpoint = request["read_resume_checkpoint"]
            revised_task = state(FakeModel(), checkpoint["read_results"], cross_paper_enabled=True)
            revised_task.update(session_repo=repo, session_key=session.key, turn_id="two", read_resume_checkpoint=checkpoint)
            second = await run_research_node()(revised_task)
            current = second["research_report"]["cards"][0]["claims"][0]
            self.assertNotEqual(current["version"], old["version"])
            self.assertEqual(current["statement"], body["changes"]["statement"])
            self.assertEqual(second["research_report"]["cards"][0]["verifications"][0]["target_version"], current["version"])
            self.assertTrue(second["research_report"]["revision_history"])
            self.assertTrue(checkpoint["revision"]["invalidated_relation_ids"])
            for relation in second["research_report"]["relations"]:
                self.assertNotIn(old["id"], relation["claim_ids"])
            with self.assertRaises(ValueError):
                revision_request(repo, session.key, body)
            original = json.loads(repo.read_artifact_path(session.key, latest["artifact_id"]).read_text())
            self.assertEqual(original["cards"][0]["claims"][0]["statement"], old["statement"])

    async def test_revision_rejects_foreign_evidence_and_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            task = state(FakeModel(), [item(), item("p2")])
            task.update(session_repo=repo, session_key=session.key, turn_id="one")
            first = await run_research_node()(task)
            latest = latest_report(repo, session.key)
            old = first["research_report"]["cards"][0]["claims"][0]
            foreign = first["research_report"]["cards"][1]["claims"][0]["evidence_ids"]
            body = {"artifact_id": latest["artifact_id"], "claim_id": old["id"], "target_version": old["version"], "reason": "revision", "changes": {"evidence_ids": foreign}}
            with self.assertRaises(ValueError):
                revision_request(repo, session.key, body)
            body["changes"] = {"statement": "changed"}
            checkpoint = revision_request(repo, session.key, body)["read_resume_checkpoint"]
            checkpoint["read_results"][0]["paper"]["abstract"] = "Changed source"
            with self.assertRaises(ValueError):
                await run_research_node()({**task, "turn_id": "two", "read_results": checkpoint["read_results"], "read_resume_checkpoint": checkpoint})


class ResearchApiTest(unittest.TestCase):
    def test_report_and_revision_routes_validate_session_and_version(self):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create()
            task = state(FakeModel())
            task.update(session_repo=repo, session_key=session.key, turn_id="original")
            asyncio.run(run_research_node()(task))
            app = create_app(sessions_repo=repo)
            accepted = {"session_key": session.key, "run_id": "revision-run", "turn_id": "revision-turn", "status": "accepted", "stream_url": "/stream"}
            with TestClient(app) as client:
                response = client.get(f"/api/sessions/{session.key}/research")
                self.assertEqual(response.status_code, 200)
                report = response.json()
                target = report["report"]["cards"][0]["claims"][0]
                body = {"artifact_id": report["artifact_id"], "claim_id": target["id"], "target_version": target["version"], "reason": "更准确的前提", "changes": {"conditions": "指定数据集"}}
                with patch("src.services.session_runs.SessionRunService.start_run", AsyncMock(return_value=accepted)) as start:
                    result = client.post(f"/api/sessions/{session.key}/research/revisions", json=body)
                    self.assertEqual(result.status_code, 202)
                    checkpoint = start.await_args.args[1]["read_resume_checkpoint"]
                    self.assertFalse(checkpoint["request"]["constraints"]["cross_paper_enabled"])
                    self.assertEqual(checkpoint["revision"]["replacement"]["conditions"], "指定数据集")
                body["target_version"] = "stale"
                self.assertEqual(client.post(f"/api/sessions/{session.key}/research/revisions", json=body).status_code, 400)
                other = repo.create()
                self.assertEqual(client.post(f"/api/sessions/{other.key}/research/revisions", json=body).status_code, 400)


if __name__ == "__main__":
    unittest.main()
