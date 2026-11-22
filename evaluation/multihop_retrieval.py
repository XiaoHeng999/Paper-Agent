"""Run the controlled two-hop benchmark with python -m evaluation.multihop_retrieval."""
import asyncio
import json
from pathlib import Path

from src.research.context import ScopedEvidence
from src.research.models import EvidenceRequirement, EvidenceSpan, ReadingPlan, ReadingSubquery
from src.research.selection import build_evidence_packets


async def evaluate():
    cases = json.loads(Path(__file__).with_name("multihop-retrieval.json").read_text())
    results = []
    for case in cases:
        def span(key, text, **fields):
            return EvidenceSpan(id=key, text=text, paper_id=case["id"], version="fixture",
                source_level="full_text", **fields)

        # Fill the global requirement ranking with over-budget passages. Only bounded
        # expansion can bring the short target back into the packet.
        spans = [span("seed", "Latency result.", next_id="bridge"), span("bridge", case["bridge"])]
        spans += [span(f"noise-{i}", case["requirement"] + " FP16 A100 " + "noise " * 90) for i in range(16)]
        spans.extend([span("distractor", case["distractor"]), span("candidate", case["candidate"])])
        query = ReadingSubquery(id="q", source="user", lens="experimental_design", question="Latency?",
            retrieval_queries=["latency"], evidence_requirements=[EvidenceRequirement(kind=case["requirement"])])
        plan = ReadingPlan(paper_id=case["id"], mode="targeted", user_prompt=query.question, subqueries=[query])
        packet, = await build_evidence_packets(ScopedEvidence(spans, {case["id"]}), plan, None,
            per_packet_chars=120)
        paths = {t.evidence_id: t.retrieval_path for t in packet.candidate_trace if len(t.retrieval_path) == 2}
        expected = case["relation"]
        recalled = "candidate" in packet.evidence_ids and "candidate" in paths and (
            paths["candidate"][-1].relation == expected)
        wrong = set(paths) - ({"candidate"} if expected else set())
        results.append({"id": case["id"], "expected": expected, "recalled": recalled,
            "wrong_paths": sorted(wrong), "passed": not wrong and (recalled if expected else not paths)})
    positive = [r for r in results if r["expected"]]
    return {"cases": results, "positive_path_recall": sum(r["recalled"] for r in positive)/len(positive),
        "wrong_paths": sum(len(r["wrong_paths"]) for r in results)}


if __name__ == "__main__":
    metrics = asyncio.run(evaluate())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    raise SystemExit(int(not all(case["passed"] for case in metrics["cases"])))
