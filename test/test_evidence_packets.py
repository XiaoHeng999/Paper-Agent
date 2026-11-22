import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse
from src.research.agents import ModelTasks, verify
from src.research.context import ScopedEvidence
from src.research.models import (
    Claim, EvidenceRequirement, EvidenceSpan, ReadingPlan, ReadingSubquery, ResearchOptions, VerificationDraft,
)
from src.research.planning import coverage_for_plan
from src.research.selection import build_evidence_packets, packet_priority, refine_packets_with_model


class RerankingModel:
    signature = "reranking-test"
    provider = None

    def __init__(self):
        self.provider = self
        self.roles = []

    async def chat(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        self.roles.append(payload["role"])
        if payload["role"] == "node_verifier":
            judgments = [{"subquery_id": item["subquery_id"], "evidence_id": item["evidence_id"],
                "direct_support": 4 if item["evidence_id"] == "result" else 2,
                "supported_requirements": item["candidate_requirements"], "reason": "listwise relevance"}
                for item in payload["candidates"]]
            return LLMResponse(content=json.dumps({"judgments": judgments}))
        if payload["role"] == "increment_verifier":
            judgments = [{"subquery_id": item["subquery_id"], "anchor_id": item["anchor_id"],
                "evidence_id": item["evidence_id"],
                "role": "redundant" if item["evidence_id"] == "duplicate" else "conflict",
                "supported_requirements": item["candidate_requirements"], "reason": "increment role"}
                for item in payload["candidates"]]
            return LLMResponse(content=json.dumps({"judgments": judgments}))
        raise AssertionError(payload["role"])


def subquery(requirements, question="Does the latency result hold on the reported hardware?"):
    return ReadingSubquery(id="rq:test", source="user", lens="experimental_design",
        question=question, retrieval_queries=[question], priority=5,
        evidence_requirements=[EvidenceRequirement(kind=kind) for kind in requirements])


def plan(query):
    return ReadingPlan(paper_id="p1", mode="targeted", paper_type="system",
        user_prompt=query.question, user_questions=[query.question], subqueries=[query])


class EvidencePacketTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_seed_expands_relations_and_covers_complementary_requirements(self):
        result = EvidenceSpan(id="result", paper_id="p1", version="1", source_level="full_text",
            text="The latency result improves performance over the baseline.", section="4 Results",
            next_id="hardware", parent_id="experiment")
        hardware = EvidenceSpan(id="hardware", paper_id="p1", version="1", source_level="full_text",
            text="Experiments use A100 GPU hardware with batch size 8.", section="4 Results",
            previous_id="result", parent_id="experiment")
        limitation = EvidenceSpan(id="limitation", paper_id="p1", version="1", source_level="full_text",
            text="A limitation is failure on long sequences.", section="6 Limitations")
        scope = ScopedEvidence([result, hardware, limitation], {"p1"})
        scope.search = AsyncMock(return_value=[result])
        query = subquery(["result", "hardware", "limitation"])
        packets = await build_evidence_packets(scope, plan(query), None)
        packet = packets[0]
        self.assertEqual(packet.anchor_id, "result")
        self.assertEqual(set(packet.covered_requirements), {"result", "hardware", "limitation"})
        self.assertEqual(packet.missing_requirements, [])
        self.assertEqual({span.id for span in packet_priority(packets, scope)}, set(packet.evidence_ids))
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        self.assertIn("dense", trace["result"].retrieval_channels)
        self.assertEqual(trace["hardware"].relation, "adjacent")

    async def test_precision_gap_reaches_setup_through_explicit_reference(self):
        def span(key, text, **fields):
            return EvidenceSpan(id=key, text=text, paper_id="p1", version="1",
                source_level="full_text", **fields)

        seed = span("seed", "Latency result.", next_id="bridge")
        bridge = span("bridge", "Table 2: Timings")
        # Exhaust the global requirement ranking with passages outside the packet budget.
        distractions = [span(f"other{i}", "FP16 " + "unrelated " * 100) for i in range(16)]
        wrong = span("wrong", "Table 20 uses FP16.")
        setup = span("setup", "For Table 2 we used FP16.")
        scope = ScopedEvidence([seed, bridge, *distractions, wrong, setup], {"p1"})
        scope.search = AsyncMock(return_value=[seed])
        query = subquery(["precision"], "Latency?")
        packet = (await build_evidence_packets(scope, plan(query), None, per_packet_chars=120))[0]
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        self.assertIn("setup", packet.evidence_ids)
        self.assertNotIn("wrong", trace)
        path = trace["setup"].retrieval_path
        self.assertEqual([(e.source_id, e.relation, e.target_id) for e in path],
            [("seed", "adjacent", "bridge"), ("bridge", "explicit_reference", "setup")])
        self.assertIn("table:2", path[-1].provenance)
        self.assertEqual(packet.missing_requirements, [])

        claim = Claim(id="claim", paper_id="p1", version="1", statement="Latency with FP16",
            evidence_ids=["seed", "setup"])
        tasks = SimpleNamespace(options=SimpleNamespace(context_chars=4000),
            ask=AsyncMock(return_value=VerificationDraft(verdict="verified", reason="reported",
                evidence_ids=["seed"], fidelity={"status": "supported", "reason": "reported",
                    "evidence_ids": ["seed"]})))
        with patch("src.research.visual.verification_images", return_value=([], [], [])):
            result = await verify(tasks, claim, scope, evidence_packets=[packet])
        self.assertEqual(result.verdict, "qualified")
        self.assertIn("bridge", result.inspected_evidence_ids)
        self.assertTrue(result.repair_request)
        self.assertEqual(len(tasks.ask.await_args.args[2]["retrieval_chains"]), 1)

        inferred = claim.model_copy(update={"attribution": "system_inference", "evidence_ids": ["setup"],
            "subquery_ids": [packet.subquery_id]})
        with patch("src.research.visual.verification_images", return_value=([], [], [])):
            checked = await verify(tasks, inferred, scope, evidence_packets=[packet])
        self.assertEqual(checked.retrieval_paths, [path])
        self.assertTrue({"seed", "bridge", "setup"} <= set(checked.inspected_evidence_ids))

    async def test_controlled_multihop_cases(self):
        from evaluation.multihop_retrieval import evaluate
        metrics = await evaluate()
        self.assertTrue(all(case["passed"] for case in metrics["cases"]), metrics)

    async def test_optional_image_channel_is_scoped_and_fails_back_to_text(self):
        oid = "p1:visual-object:figure:1"
        span = EvidenceSpan(id="caption", paper_id="p1", version="1", text="Figure 1: Results.",
            source_level="full_text", visual_object_ids=[oid], view_type="caption")
        scope = ScopedEvidence([span], {"p1"})
        query = subquery(["figure"], "Which figure contains dashed curves?")
        tasks = ModelTasks(None, ResearchOptions(image_embedding_enabled=True))
        tasks.visual_pages = {("p1", 1): {"pixel_bboxes": {"figure:1": [0,0,2,2]}},
            ("p2", 1): {"pixel_bboxes": {"figure:1": [0,0,2,2]}}}
        with patch.object(tasks.image_index, "_rank", return_value=([(oid, .8)], 1)) as rank:
            packet, = await build_evidence_packets(scope, plan(query), tasks)
        candidates = rank.call_args.args[1]
        self.assertEqual([item[0] for item in candidates], [oid])
        self.assertIn("image_embedding", packet.candidate_trace[0].retrieval_channels)
        with patch.object(tasks.image_index, "_rank", side_effect=ValueError("unavailable")):
            packet, = await build_evidence_packets(scope, plan(query), tasks)
        self.assertEqual(packet.evidence_ids, ["caption"])
        self.assertNotIn("image_embedding", packet.candidate_trace[0].retrieval_channels)
        self.assertTrue(tasks.reranking_warnings["p1"])

    async def test_exact_reference_channel_is_auditable(self):
        table = EvidenceSpan(id="table", paper_id="p1", version="1", source_level="full_text",
            text="Table 2 reports latency and accuracy results.", section="Results")
        scope = ScopedEvidence([table], {"p1"})
        query = subquery(["table", "result"], "What does Table 2 report?")
        packet = (await build_evidence_packets(scope, plan(query), None))[0]
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        self.assertIn("exact_reference", trace["table"].retrieval_channels)

    async def test_missing_requirements_create_bounded_repair_queries(self):
        result = EvidenceSpan(id="result", paper_id="p1", version="1", source_level="full_text",
            text="The reported result improves latency.", section="Results")
        query = subquery(["result", "hardware"])
        packet = (await build_evidence_packets(ScopedEvidence([result], {"p1"}), plan(query), None))[0]
        self.assertEqual(packet.covered_requirements, ["result"])
        self.assertEqual(packet.missing_requirements, ["hardware"])
        self.assertGreaterEqual(len(packet.repair_queries), 1)
        self.assertLessEqual(len(packet.repair_queries), 4)

    async def test_one_gap_dense_round_adds_semantic_candidate_without_false_coverage(self):
        result = EvidenceSpan(id="result", paper_id="p1", version="1", source_level="full_text",
            text="The reported result improves latency.", section="Results")
        hardware = EvidenceSpan(id="hardware", paper_id="p1", version="1", source_level="full_text",
            text="All runs use one accelerator device.", section="Setup")
        scope = ScopedEvidence([result, hardware], {"p1"})
        scope.search = AsyncMock(side_effect=[[result], [hardware]])
        tasks = SimpleNamespace(options=SimpleNamespace(max_repair_rounds=1))
        query = subquery(["result", "hardware"])
        packet = (await build_evidence_packets(scope, plan(query), tasks))[0]
        self.assertIn("hardware", packet.evidence_ids)
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        self.assertIn("gap_dense", trace["hardware"].retrieval_channels)
        self.assertEqual(packet.missing_requirements, ["hardware"])
        self.assertEqual(scope.search.await_count, 2)

    async def test_claim_and_complete_packet_are_both_required_for_covered_status(self):
        span = EvidenceSpan(id="complete", paper_id="p1", version="1", source_level="full_text",
            text="Results on dataset D outperform baseline B on accuracy metric using an A100 GPU, "
                 "with a fixed budget; Table 2 reports performance.", section="Experiments")
        query = subquery(["dataset", "baseline", "metric", "budget", "hardware", "result", "table"])
        reading_plan = plan(query)
        packet = (await build_evidence_packets(ScopedEvidence([span], {"p1"}), reading_plan, None))[0]
        without_claim = coverage_for_plan(reading_plan, [], [packet])[0]
        self.assertEqual(without_claim.status, "unanswered")
        claim = Claim(id="c", paper_id="p1", version="1", statement="reported result",
            evidence_ids=[span.id], subquery_ids=[query.id])
        with_claim = coverage_for_plan(reading_plan, [claim], [packet])[0]
        self.assertEqual(with_claim.status, "covered")
        self.assertEqual(with_claim.missing_requirements, [])

    async def test_optional_listwise_verifiers_remove_redundancy_and_keep_conflict(self):
        spans = [
            EvidenceSpan(id="result", paper_id="p1", version="1", source_level="full_text",
                text="The result improves latency.", section="Results"),
            EvidenceSpan(id="duplicate", paper_id="p1", version="1", source_level="full_text",
                text="The reported result improves latency.", section="Abstract"),
            EvidenceSpan(id="limitation", paper_id="p1", version="1", source_level="full_text",
                text="A limitation is failure on long sequences.", section="Limitations"),
        ]
        scope = ScopedEvidence(spans, {"p1"})
        scope.search = AsyncMock(return_value=[spans[0], spans[1]])
        query = subquery(["result", "limitation", "hardware"])
        model = RerankingModel()
        tasks = ModelTasks(model, ResearchOptions(evidence_reranking_enabled=True, max_repair_rounds=0))
        packets = await build_evidence_packets(scope, plan(query), tasks)
        self.assertIn("duplicate", packets[0].evidence_ids)
        refined = await refine_packets_with_model(tasks, scope, packets)
        self.assertEqual(model.roles, ["node_verifier", "increment_verifier"])
        self.assertNotIn("duplicate", refined[0].evidence_ids)
        self.assertIn("limitation", refined[0].evidence_ids)
        trace = {item.evidence_id: item for item in refined[0].candidate_trace}
        self.assertEqual(trace["limitation"].incremental_role, "conflict")
        self.assertEqual(trace["result"].direct_support, 4)


if __name__ == "__main__":
    unittest.main()
