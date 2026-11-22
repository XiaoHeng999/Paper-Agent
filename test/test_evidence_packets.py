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

        for quotes, verdict in [({}, "qualified"),
                ({"seed":seed.text, "bridge":bridge.text, "setup":setup.text}, "verified"),
                ({"seed":"fabrication"}, "insufficient_evidence")]:
            tasks.ask.return_value = VerificationDraft(verdict="verified", reason="reported",
                evidence_ids=["seed"], fidelity={"status":"supported","evidence_ids":["seed"]},
                chain_checks=[{"evidence_ids":["seed","bridge","setup"], "status":"consistent",
                    "reason":"aligned", "quotes":quotes}])
            with patch("src.research.visual.verification_images", return_value=([], [], [])):
                checked = await verify(tasks, claim, scope, evidence_packets=[packet])
            self.assertEqual(checked.verdict, verdict)

        inferred = claim.model_copy(update={"attribution": "system_inference", "evidence_ids": ["setup"],
            "subquery_ids": [packet.subquery_id]})
        with patch("src.research.visual.verification_images", return_value=([], [], [])):
            checked = await verify(tasks, inferred, scope, evidence_packets=[packet])
        self.assertEqual(checked.retrieval_paths, [path])
        self.assertTrue({"seed", "bridge", "setup"} <= set(checked.inspected_evidence_ids))

    def test_local_answer_evaluation_requires_both_real_evidence_ids(self):
        from evaluation.local_pdf_answers import score
        case = {"paper_id":"p", "evidence":[{"id":"a"}, {"id":"b"}],
            "expected_sources":[["a"], ["b"]], "answer_terms":[["FP16", "half precision"], ["200"]]}
        for citations, expected in [([1,2], True), ([1], False), ([0,1,2], False), ([True,2], False)]:
            result = score(case, json.dumps({"answer":"200 runs at half precision", "citations":citations}))
            self.assertEqual(result["fact_term_matches"], [True, True])
            self.assertEqual(result["both_sources_cited"], expected)
        self.assertFalse(any(score(case, "malformed")["fact_term_matches"]))

    def test_human_evidence_benchmark_excludes_ambiguous_or_single_section_labels(self):
        from evaluation.qasper_evidence import reference_sets
        locations = {"a":["a"], "b":["b"], "duplicate":["a","b"]}
        sections = {"a":1, "b":2}
        def answer(*texts):
            return {"answer":{"unanswerable":False, "evidence":texts}}
        self.assertEqual(reference_sets({"answers":[answer("a","b")]}, locations, sections), [{"a","b"}])
        for answers in ([answer("a","b"), answer("a")], [answer("a","duplicate")], [answer("missing")]):
            self.assertEqual(reference_sets({"answers":answers}, locations, sections), [])

    async def test_controlled_multihop_cases(self):
        from evaluation.multihop_retrieval import evaluate
        metrics = await evaluate()
        self.assertTrue(all(case["passed"] for case in metrics["cases"]), metrics)

    async def test_cross_page_acronym_requires_one_explicit_expansion(self):
        seed = EvidenceSpan(id="seed", paper_id="p1", version="v", source_level="full_text",
            page_start=5, text="DSR reduces latency.")
        definition = seed.model_copy(update={"id": "definition", "page_start": 2,
            "text": "We introduce the Dynamic Sparse Router (DSR)."})
        query = subquery(["method_step"], "How does DSR reduce latency?")
        scope = ScopedEvidence([seed, definition], {"p1"})
        scope.search = AsyncMock(return_value=[seed])
        packet, = await build_evidence_packets(scope, plan(query), None)
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        hop, = trace["definition"].retrieval_path
        self.assertEqual(hop.relation, "definition_of")
        self.assertIn("Dynamic Sparse Router (DSR)", hop.provenance)

        ambiguous = definition.model_copy(update={"id": "ambiguous", "text": "Dense Signal Reader (DSR)."})
        scope = ScopedEvidence([seed, definition, ambiguous], {"p1"})
        scope.search = AsyncMock(return_value=[seed])
        packet, = await build_evidence_packets(scope, plan(query), None)
        self.assertFalse(any(h.relation == "definition_of" for t in packet.candidate_trace for h in t.retrieval_path))

    def test_visual_model_selection_and_legacy_options(self):
        from src.research.image_retrieval import ClipIndex, GmeIndex
        for constraints, expected in [
                ({}, GmeIndex), ({"image_embedding_enabled": True}, GmeIndex),
                ({"image_embedding_model": "clip"}, ClipIndex),
                ({"image_embedding_enabled": False, "image_embedding_model": "clip"}, type(None))]:
            options = ResearchOptions.from_constraints(constraints)
            self.assertIsInstance(ModelTasks(None, options).image_index, expected)
        with self.assertRaises(ValueError):
            ResearchOptions.from_constraints({"image_embedding_model": "unknown"})

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

    async def test_implicit_setting_requires_a_scoped_antecedent_and_verbatim_quote(self):
        setup = EvidenceSpan(id="setup", paper_id="p1", version="v", section="Results",
            page_start=2, source_level="full_text", text="The low-memory configuration uses FP16.")
        result = setup.model_copy(update={"id": "result", "page_start": 3, "previous_id": "setup",
            "text": "Under this setting latency decreases."})
        scope = ScopedEvidence([setup, result], {"p1"})
        claim = Claim(id="c", paper_id="p1", version="v", statement=result.text, evidence_ids=[result.id])
        valid = {"source_id": "result", "antecedent_id": "setup", "quote": setup.text,
            "referent": "low-memory configuration"}
        for resolutions, expected in [([valid], "verified"), ([], "qualified"),
                ([{**valid, "quote": "Uses the high-memory configuration."}], "insufficient_evidence")]:
            draft = VerificationDraft(verdict="verified", reason="reported", evidence_ids=["result"],
                fidelity={"status": "supported", "evidence_ids": ["result"]}, resolved_references=resolutions)
            tasks = ModelTasks(None, ResearchOptions())
            tasks.ask = AsyncMock(return_value=draft)
            checked = await verify(tasks, claim, scope)
            self.assertEqual(checked.verdict, expected)
            self.assertEqual(tasks.ask.await_args.args[2]["reference_candidates"]["result"]["candidates"], ["setup"])

    async def test_cross_section_setup_needs_unique_versioned_experiment(self):
        setup = EvidenceSpan(id="setup", paper_id="p1", version="v", source_version="pdf",
            section="Design", page_start=2, source_level="full_text",
            text="The TinyNet configuration on CIFAR10 uses FP16.")
        result = setup.model_copy(update={"id":"result", "section":"Results", "page_start":6,
            "text":"Latency decreases under this setting for TinyNet on CIFAR10."})
        alternative = setup.model_copy(update={"id":"alt", "text":setup.text.replace("FP16", "FP32")})
        target = Claim(id="c", version="v", paper_id="p1", statement=result.text, evidence_ids=["result"])
        for version, expected in [("other", "verified"), ("pdf", "qualified")]:
            tasks = ModelTasks(None, ResearchOptions())
            tasks.ask = AsyncMock(return_value=VerificationDraft(verdict="verified", reason="reported",
                evidence_ids=["result"], fidelity={"status":"supported","evidence_ids":["result"]},
                resolved_references=[{"source_id":"result","antecedent_id":"setup",
                    "quote":setup.text,"referent":"TinyNet configuration"}]))
            scope = ScopedEvidence([result,setup,alternative.model_copy(update={"source_version":version})], {"p1"})
            checked = await verify(tasks, target, scope)
            self.assertEqual(checked.verdict, expected)
            self.assertEqual(bool(checked.resolved_references), expected == "verified")

    async def test_visual_gap_requires_an_explicit_topic_and_missing_evidence(self):
        from src.research.visual import _visual_gaps
        for kind, question, answer in [
                ("budget", "What is the compute budget?", "The compute budget was 300 GPU hours."),
                ("ablation", "Where are the ablation results?", "Our ablation removes the router."),
                ("metric", "Which accuracy metric is shown?", "The metric is top-1 accuracy.")]:
            query = subquery([kind], question)
            self.assertEqual((await _visual_gaps(plan(query), []))[0]["requirements"], [kind])
            span = EvidenceSpan(id="a", paper_id="p1", version="v", source_level="full_text", text=answer)
            self.assertEqual(await _visual_gaps(plan(query), [span]), [])
            self.assertEqual(await _visual_gaps(plan(subquery([kind], "Summarize the paper")), []), [])

    async def test_verifying_a_continued_cell_reads_both_original_pages(self):
        from src.research.multimodal import build_multimodal_catalog
        first = EvidenceSpan(id="first", paper_id="p1", version="v", source_version="pdf",
            source_level="full_text", page_start=1, text="Table 1: Setup\n|Model|Precision|\n|---|---|\n|A|FP32|")
        second = first.model_copy(update={"id": "second", "page_start": 2,
            "text": "Table 1 (continued)\n|B|FP16|"})
        spans, _ = build_multimodal_catalog([first, second])
        cell, = [s for s in spans if s.table_cell and s.parent_id == "second"]
        scope = ScopedEvidence(spans, {"p1"})
        tasks = ModelTasks(None, ResearchOptions())
        tasks.visual_catalog["p1"] = [{"page": n, "captions": ["table:1"]} for n in (1, 2)]
        tasks.visual_pages = {("p1", n): {"url": "image", "image_version": str(n), "source_version": "pdf"}
                              for n in (1, 2)}
        tasks.ask = AsyncMock(return_value=VerificationDraft(verdict="insufficient_evidence",
            reason="unconfirmed", evidence_ids=[]))
        target = Claim(id="c", paper_id="p1", version="v", statement="B uses FP16", evidence_ids=[cell.id])
        checked = await verify(tasks, target, scope)
        self.assertIn("first", checked.inspected_evidence_ids)
        self.assertEqual({p["page"] for p in tasks.ask.await_args.args[2]["visual_pages"]}, {1, 2})

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
