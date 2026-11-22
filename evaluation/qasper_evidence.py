"""Score cross-section evidence retrieval on the human-annotated QASPER v0.3 dev split."""
import argparse
import asyncio
import hashlib
import json
import tarfile
import time
from types import SimpleNamespace
from collections import defaultdict
from pathlib import Path

from src.research.context import ScopedEvidence
from src.research.models import EvidenceSpan, version_of
from src.research.multimodal import build_multimodal_catalog
from src.research.planning import build_reading_plan
from src.research.selection import _rank_question_terms, _select_packet, build_evidence_packets
from src.research.retrieval import LocalE5Index


ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"


def reference_sets(question, locations, sections):
    """Require each answer annotation to identify two or more unambiguous sections."""
    alternatives = []
    for item in question["answers"]:
        answer = item["answer"]
        paragraphs = [locations.get(text.strip(), []) for text in answer["evidence"]]
        if answer["unanswerable"] or any(len(ids) != 1 for ids in paragraphs):
            return []
        ids = {matches[0] for matches in paragraphs}
        if len({sections[eid] for eid in ids}) < 2:
            return []
        alternatives.append(ids)
    return alternatives


async def compare_channels(scope, query, index, gold, limit=40):
    lexical = list(_rank_question_terms(scope.spans.values(), query.question, limit))
    dense = await index.search(query.question, scope, limit)
    if not dense:
        raise ValueError("Empty dense ranking")
    limit = min(limit, len(lexical), len(dense))
    rankings = {"bm25": lexical, "dense": list(enumerate(dense, 1))}
    comparisons = {}
    for mode in ("bm25", "dense", "fusion"):
        scores = defaultdict(float)
        channels = defaultdict(list)
        for name in rankings if mode == "fusion" else [mode]:
            for rank, span in rankings[name]:
                scores[span.id] += 1 / (60 + rank)
                channels[span.id].append(name)
        ids = sorted(scores, key=lambda eid: (-scores[eid], eid))[:limit]
        ranked = [(scope.spans[eid], scores[eid], channels[eid]) for eid in ids]
        selected = _select_packet(scope, query, ranked, 4200, 5, {}).evidence_ids
        comparisons[mode] = {"candidate_complete": any(g <= set(ids) for g in gold),
            "complete": any(g <= set(selected) for g in gold), "candidate_count": len(ids)}
    return comparisons


async def evaluate(archive, index=None):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Expected the original QASPER v0.3 train/dev archive")
    with tarfile.open(archive, "r:gz") as source:
        papers = json.load(source.extractfile("qasper-dev-v0.3.json"))
    cases = []
    total = 0
    for pid, paper in papers.items():
        spans = []
        locations = defaultdict(list)
        sections = {}
        for section_idx, section in enumerate(paper["full_text"]):
            for offset, paragraph in enumerate(section["paragraphs"]):
                text = paragraph.strip()
                if not text:
                    continue
                eid = f"{pid}:{section_idx}:{offset}"
                locations[text].append(eid)
                sections[eid] = section_idx
                span = EvidenceSpan(id=eid, paper_id=pid, version=version_of(text), text=text,
                    section=section["section_name"] or "", source_level="full_text",
                    previous_id=spans[-1].id if spans else "")
                if spans:
                    spans[-1].next_id = eid
                spans.append(span)
        spans, _ = build_multimodal_catalog(spans)
        scope = ScopedEvidence(spans, {pid}, index)
        prepared = False
        tasks = SimpleNamespace(cancellation=None, deadline=time.monotonic() + 3600)
        for question in paper["qas"]:
            total += 1
            gold = reference_sets(question, locations, sections)
            if not gold:
                continue
            plan = build_reading_plan(question["question"], pid, paper["title"])
            queries = [q for q in plan.subqueries if q.source == "user"]
            if len(queries) != 1:
                continue
            if index and not prepared:
                await index.prepare(spans)
                prepared = True
            packet, = await build_evidence_packets(scope, plan.model_copy(update={"subqueries": queries}), tasks)
            selected = set(packet.evidence_ids)
            candidates = [item.evidence_id for item in packet.candidate_trace]
            cases.append({"selected": packet.evidence_ids, "candidates": candidates,
                "selected_chars": sum(len(scope.spans[i].text) for i in selected),
                "candidate_complete": any(ids <= set(candidates) for ids in gold), "paper": pid, "question_id": question["question_id"],
                "complete": any(ids <= selected for ids in gold),
                "recall": max(len(ids & selected)/len(ids) for ids in gold)})
            if index:
                cases[-1]["channels"] = await compare_channels(scope, queries[0], index, gold)
    if index and index.warnings:
        raise RuntimeError(index.warnings)
    return {"split": "dev", "questions_examined": total, "eligible_questions": len(cases),
        "papers": len({c["paper"] for c in cases}), "candidate_complete": sum(c["candidate_complete"] for c in cases), "complete": sum(c["complete"] for c in cases),
        "mean_paragraph_recall": sum(c["recall"] for c in cases)/len(cases) if cases else None,
        "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--e5-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()
    if args.e5_dir and not args.index_dir:
        parser.error("--e5-dir requires --index-dir")
    index = LocalE5Index(args.e5_dir, args.index_dir, "qasper") if args.e5_dir else None
    print(json.dumps(asyncio.run(evaluate(args.archive, index)), indent=2))
