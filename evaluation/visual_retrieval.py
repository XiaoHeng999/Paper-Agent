"""Paired offline visual retrieval on the annotated PDF, using local GME or CLIP weights."""
import argparse
import asyncio
import json
import re
import time
import tempfile
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium

from src.research.label_ocr import label_views
from src.research.agents import ModelTasks
from src.research.context import ScopedEvidence
from src.research.models import Claim, EvidenceRequirement, EvidenceSpan, ReadingPlan, ReadingSubquery, ResearchOptions
from src.research.multimodal import build_multimodal_catalog
from src.research.storage import load_evidence
from src.utils.read_utils.pdf_parsers import PypdfParser
from src.research.selection import build_evidence_packets
from src.research.visual import _crop_png, inspect_pdf, verification_images


def _normalize(text):
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


async def evaluate(path, full_text=False, read_crops=False, image_model="qwen2-vl"):
    fixtures = json.loads(Path(__file__).with_name("visual-retrieval.json").read_text())
    pages, _, _ = inspect_pdf(path, 20)
    fixture = next((f for f in fixtures if pages and f["source_version"] == pages[0]["source_version"]), None)
    if not fixture:
        raise ValueError("评测 PDF 与标注的源版本不同")
    pid = fixture["paper_id"]
    spans = []
    with pdfium.PdfDocument(path) as doc:
        for page in sorted(pages, key=lambda p: p["page"]):
            with closing(doc[page["page"]-1]) as native, closing(native.get_textpage()) as textpage:
                lines = textpage.get_text_range().splitlines()
            for label in page["pixel_bboxes"]:
                number = label.split(":")[1]
                caption = next(line for line in lines if re.match(rf"^\s*Fig(?:ure|\.)?\s+{number}\s*[:.|]", line, re.I))
                spans.append(EvidenceSpan(id=label, text=caption, version=page["image_version"],
                    page_start=page["page"], page_end=page["page"], section="Results", source_level="full_text", paper_id=pid, view_type="caption",
                    visual_object_ids=[f"{pid}:visual-object:{label}"]))
    locations = {s.id: s.page_start for s in spans}
    if full_text:
        parsed = PypdfParser().parse(path)
        with tempfile.TemporaryDirectory() as directory:
            markdown = Path(directory) / "paper.md"
            markdown.write_text("\n\n".join(f"<!-- page: {p.page_number} -->\n{p.text}" for p in parsed.pages))
            native = load_evidence({"paper": {"paperId": pid}, "full_text": {"markdown_path": str(markdown)}})
        spans, _ = build_multimodal_catalog(native, pages, {(pid, p["page"]): p for p in pages})
    scope = ScopedEvidence(spans, {pid})
    output = {}
    ocr = {}
    for enabled in (False, True):
        tasks = ModelTasks(None, ResearchOptions(image_embedding_enabled=enabled, image_embedding_model=image_model))
        tasks.visual_pages = {(pid, p["page"]): p for p in pages}
        tasks.visual_catalog = {pid: pages}
        results = []
        for case in fixture["cases"]:
            query = ReadingSubquery(id=case["id"], source="user", lens="experimental_design",
                question=case["question"], retrieval_queries=[case["question"]],
                evidence_requirements=[EvidenceRequirement(kind="figure")])
            plan = ReadingPlan(paper_id=pid, mode="targeted", user_prompt=query.question, subqueries=[query])
            packet, = await build_evidence_packets(scope, plan, tasks)
            target = Claim(id=query.id, version="test", paper_id=pid, statement=query.question,
                evidence_ids=[packet.anchor_id])
            images, context, missing = verification_images(tasks, scope.read(target.evidence_ids), target)
            expected_page = locations[case["label"]]
            anchor = scope.spans[packet.anchor_id]
            predicted = [oid.removeprefix(f"{pid}:visual-object:") for oid in anchor.visual_object_ids]
            correct_page = bool(images) and not missing and any(p["page"] == expected_page and
                p["source_version"] == fixture["source_version"] for p in context)
            result = {"original_page_available": correct_page, "query": query.id, "anchor": packet.anchor_id,
                "predicted": predicted, "correct": predicted == [case["label"]]}
            if read_crops and case.get("visible_text"):
                text = []
                native_text = ""
                if len(predicted) == 1 and predicted[0] in locations:
                    label = predicted[0]
                    page = tasks.visual_pages[pid, locations[label]]
                    native_text = page.get("object_texts", {}).get(label, "")
                    if label not in ocr:
                        readings = label_views(_crop_png(page, label), time.monotonic()+9)
                        ocr[label] = list(readings.values())
                    text = ocr[label]
                expected = _normalize(case["visible_text"])
                method = "native" if expected in _normalize(native_text) else "ocr" if any(expected in _normalize(view) for view in text) else None
                result["text_match_method"] = method
                result["verified_visible_text"] = result["correct"] and correct_page and bool(method)
            results.append(result)
        if tasks.reranking_warnings:
            raise RuntimeError(tasks.reranking_warnings)
        mode = image_model if enabled else "text" if full_text else "caption"
        output[mode] = {"cases": results,
            "recall_at_1": sum(r["correct"] for r in results)/len(results),
            "image_queries": tasks.image_index.audit if enabled else []}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--full-text", action="store_true", help="Include native PDF chunks and object views")
    parser.add_argument("--read-crops", action="store_true", help="Check annotated visible text using local Tesseract")
    parser.add_argument("--image-model", choices=["qwen2-vl", "clip"], default="qwen2-vl")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(args.pdf, args.full_text, args.read_crops, args.image_model)), ensure_ascii=False, indent=2))
