"""Score cross-section evidence retrieval on the human-annotated QASPER v0.3 dev split."""
import argparse
import asyncio
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from src.research.context import ScopedEvidence
from src.research.models import EvidenceSpan, version_of
from src.research.multimodal import build_multimodal_catalog
from src.research.planning import build_reading_plan
from src.research.selection import build_evidence_packets


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


async def evaluate(archive):
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
        for index, section in enumerate(paper["full_text"]):
            for offset, paragraph in enumerate(section["paragraphs"]):
                text = paragraph.strip()
                if not text:
                    continue
                eid = f"{pid}:{index}:{offset}"
                locations[text].append(eid)
                sections[eid] = index
                span = EvidenceSpan(id=eid, paper_id=pid, version=version_of(text), text=text,
                    section=section["section_name"] or "", source_level="full_text",
                    previous_id=spans[-1].id if spans else "")
                if spans:
                    spans[-1].next_id = eid
                spans.append(span)
        spans, _ = build_multimodal_catalog(spans)
        scope = ScopedEvidence(spans, {pid})
        for question in paper["qas"]:
            total += 1
            gold = reference_sets(question, locations, sections)
            if not gold:
                continue
            plan = build_reading_plan(question["question"], pid, paper["title"])
            queries = [q for q in plan.subqueries if q.source == "user"]
            if len(queries) != 1:
                continue
            packet, = await build_evidence_packets(scope, plan.model_copy(update={"subqueries": queries}), None)
            selected = set(packet.evidence_ids)
            cases.append({"paper": pid, "question_id": question["question_id"],
                "complete": any(ids <= selected for ids in gold),
                "recall": max(len(ids & selected)/len(ids) for ids in gold)})
    return {"split": "dev", "questions_examined": total, "eligible_questions": len(cases),
        "papers": len({c["paper"] for c in cases}), "complete": sum(c["complete"] for c in cases),
        "mean_paragraph_recall": sum(c["recall"] for c in cases)/len(cases) if cases else None,
        "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(asyncio.run(evaluate(parser.parse_args().archive)), indent=2))
