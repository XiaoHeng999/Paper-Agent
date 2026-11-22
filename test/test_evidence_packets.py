import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.llm.base import LLMResponse
from src.research.agents import ModelTasks
from src.research.context import ScopedEvidence
from src.research.models import (
    Claim, EvidenceRequirement, EvidenceSpan, ReadingPlan, ReadingSubquery, ResearchOptions,
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
