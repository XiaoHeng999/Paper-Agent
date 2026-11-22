"""Paired offline visual retrieval on the annotated PDF, using local CLIP weights."""
import argparse
import asyncio
import json
import re
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium

from src.research.agents import ModelTasks
from src.research.context import ScopedEvidence
from src.research.models import EvidenceRequirement, EvidenceSpan, ReadingPlan, ReadingSubquery, ResearchOptions
from src.research.selection import build_evidence_packets
from src.research.visual import inspect_pdf


async def evaluate(path):
    fixture = json.loads(Path(__file__).with_name("visual-retrieval.json").read_text())
    pages, _, _ = inspect_pdf(path, 20)
    if not pages or pages[0]["source_version"] != fixture["source_version"]:
        raise ValueError("评测 PDF 与标注的源版本不同")
    pid = fixture["paper_id"]
    spans = []
    with pdfium.PdfDocument(path) as doc:
        for page in sorted(pages, key=lambda p: p["page"]):
            with closing(doc[page["page"]-1]) as native, closing(native.get_textpage()) as textpage:
                lines = textpage.get_text_range().splitlines()
            for label in page["pixel_bboxes"]:
                number = label.split(":")[1]
                caption = next(line for line in lines if re.match(rf"^\s*Figure\s+{number}\s*[:.]", line, re.I))
                spans.append(EvidenceSpan(id=label, text=caption, version=page["image_version"],
                    section="Results", source_level="full_text", paper_id=pid, view_type="caption",
                    visual_object_ids=[f"{pid}:visual-object:{label}"]))
    scope = ScopedEvidence(spans, {pid})
    output = {}
    for enabled in (False, True):
        tasks = ModelTasks(None, ResearchOptions(image_embedding_enabled=enabled))
        tasks.visual_pages = {(pid, p["page"]): p for p in pages}
        results = []
        for case in fixture["cases"]:
            query = ReadingSubquery(id=case["id"], source="user", lens="experimental_design",
                question=case["question"], retrieval_queries=[case["question"]],
                evidence_requirements=[EvidenceRequirement(kind="figure")])
            plan = ReadingPlan(paper_id=pid, mode="targeted", user_prompt=query.question, subqueries=[query])
            packet, = await build_evidence_packets(scope, plan, tasks)
            results.append({"query": query.id, "anchor": packet.anchor_id,
                "correct": packet.anchor_id == case["label"]})
        if tasks.reranking_warnings:
            raise RuntimeError(tasks.reranking_warnings)
        output["clip" if enabled else "caption"] = {"cases": results,
            "recall_at_1": sum(r["correct"] for r in results)/len(results),
            "image_queries": tasks.image_index.audit if enabled else []}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    print(json.dumps(asyncio.run(evaluate(parser.parse_args().pdf)), ensure_ascii=False, indent=2))
