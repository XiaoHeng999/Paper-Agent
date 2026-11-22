"""Measure joint retrieval of two source passages in actual version-pinned PDFs."""
import argparse
import asyncio
import hashlib
import json
import re
import tempfile
from pathlib import Path

from src.research.context import ScopedEvidence
from src.research.models import EvidenceRequirement, ReadingPlan, ReadingSubquery
from src.research.multimodal import build_multimodal_catalog
from src.research.selection import build_evidence_packets
from src.research.storage import load_evidence
from src.research.visual import inspect_pdf
from src.utils.read_utils.pdf_parsers import PypdfParser


def normalized(text):
    return re.sub(r"[^\w]", "", text.casefold(), flags=re.UNICODE)


async def evaluate(cache, reader_inputs=None):
    dataset = json.loads(Path(__file__).with_name("pdf-evidence-pairs.json").read_text())
    results = []
    inputs = []
    for case in dataset["cases"]:
        pid = case["paper_id"]
        path = cache / pid / "original.pdf"
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError(f"Unexpected PDF revision for {pid}")
        pages = PypdfParser().parse(path).pages
        with tempfile.TemporaryDirectory() as tmp:
            markdown = Path(tmp) / "paper.md"
            markdown.write_text("\n\n".join(f"<!-- page: {p.page_number} -->\n{p.text}" for p in pages))
            spans = load_evidence({"paper":{"paperId":pid}, "full_text":{"markdown_path":str(markdown)}})
        _, _, catalog = inspect_pdf(path, 20)
        spans, _ = build_multimodal_catalog(spans, catalog)
        scope = ScopedEvidence(spans, {pid})
        query = ReadingSubquery(id="q", source="user", lens="experimental_design", question=case["question"],
            retrieval_queries=[case["question"]], evidence_requirements=[EvidenceRequirement(kind=k) for k in case["requirements"]])
        plan = ReadingPlan(paper_id=pid, mode="targeted", user_prompt=query.question, subqueries=[query])
        packet, = await build_evidence_packets(scope, plan, None)
        matches = []
        for page, quote in case["passages"]:
            ids = [s.id for s in spans if s.page_start and s.page_start <= page <= (s.page_end or s.page_start)
                   and normalized(quote) in normalized(s.text)]
            if not ids:
                raise ValueError(f"Missing reference passage on {pid} p.{page}: {quote}")
            matches.append(sorted(set(ids) & set(packet.evidence_ids)))
        inputs.append({"paper_id": pid, "question": query.question,
            "evidence": [{"id": s.id, "page": s.page_start, "text": s.text} for s in scope.read(packet.evidence_ids)],
            "expected_sources": matches, "answer_terms": case["answer_terms"]})
        results.append({"paper_id": pid, "matched_evidence_ids": matches, "both_retrieved": all(matches)})
    if reader_inputs:
        with reader_inputs.open("x", encoding="utf-8") as output:
            json.dump(inputs, output, ensure_ascii=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--reader-inputs", type=Path, help="Create a new file for the optional local answer evaluation")
    args = parser.parse_args()
    results = asyncio.run(evaluate(args.cache, args.reader_inputs))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    raise SystemExit(int(not all(r["both_retrieved"] for r in results)))
