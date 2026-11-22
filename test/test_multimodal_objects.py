import unittest
from unittest.mock import AsyncMock

from src.research.context import ScopedEvidence
from src.research.models import EvidenceRequirement, EvidenceSpan, ReadingPlan, ReadingSubquery
from src.research.multimodal import build_multimodal_catalog, find_object_labels
from src.research.selection import build_evidence_packets


class MultimodalObjectTests(unittest.IsolatedAsyncioTestCase):
    def evidence(self):
        return [
            EvidenceSpan(id="caption", paper_id="p1", version="1", source_level="full_text",
                text="Figure 2: Latency by sequence length", page_start=3, page_end=3,
                section="Results"),
            EvidenceSpan(id="discussion", paper_id="p1", version="1", source_level="full_text",
                text="As shown in Figure 2, latency grows slowly.", page_start=4, page_end=4,
                section="Results"),
            EvidenceSpan(id="ocr", paper_id="p1", version="1", source_level="full_text",
                text="sequence length 1024 latency 10 ms", page_start=3, page_end=3,
                section="扫描页视觉转录，须回看原图", modality="visual_transcription",
                image_version="image-v1"),
        ]

    def test_caption_ocr_and_discussion_bind_to_one_canonical_object(self):
        updated, objects = build_multimodal_catalog(self.evidence(),
            [{"page": 3, "captions": ["figure:2"]}],
            {("p1", 3): {"image_version": "image-v1"}})
        self.assertEqual(len(objects), 1)
        obj = objects[0]
        self.assertEqual(obj.id, "p1:visual-object:figure:2")
        self.assertEqual(obj.page, 3)
        self.assertEqual(obj.caption, "Figure 2: Latency by sequence length")
        self.assertEqual(obj.image_version, "image-v1")
        self.assertIsNone(obj.bbox)
        self.assertEqual(obj.caption_evidence_ids, ["caption"])
        self.assertEqual(obj.ocr_evidence_ids, ["ocr"])
        self.assertEqual(obj.discussion_evidence_ids, ["discussion"])
        views = {span.id: span for span in updated}
        self.assertEqual(views["caption"].view_type, "caption")
        self.assertEqual(views["ocr"].view_type, "ocr")
        self.assertEqual(views["discussion"].view_type, "nearby_discussion")
        self.assertTrue(all(obj.id in span.visual_object_ids for span in updated))

    def test_unlabelled_transcription_on_mixed_page_stays_page_scoped(self):
        updated, objects = build_multimodal_catalog(self.evidence(),
            [{"page": 3, "captions": ["figure:2", "table:1"]}])
        self.assertEqual(updated[-1].visual_object_ids, [])
        self.assertTrue(all(not obj.ocr_evidence_ids for obj in objects))

    async def test_rectangular_table_preserves_headers_units_and_parent_evidence(self):
        text = ("Table 2: Controlled comparison\n"
                "| Method | Precision | Latency (ms) |\n|---|---|---|\n"
                "| Ours | FP16 | 10* |\n| Baseline | FP32 | 14 |\n* At batch size 1.")
        table = EvidenceSpan(id="t", paper_id="p1", version="v", text=text, source_level="full_text",
            page_start=4, page_end=4)
        spans, objects = build_multimodal_catalog([table])
        cells = [span for span in spans if span.table_cell]
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[1].table_cell.model_dump(), {"row": 1, "column": 3,
            "row_header": "Ours", "column_header": "Latency (ms)", "value": "10*"})
        self.assertEqual(cells[1].parent_id, "t")
        self.assertEqual(cells[1].table_notes, ["* At batch size 1."])
        self.assertFalse(cells[3].table_notes)
        self.assertEqual(objects[0].cell_evidence_ids, [s.id for s in cells])
        self.assertEqual(build_multimodal_catalog(spans), (spans, objects))
        query = ReadingSubquery(id="rq", source="user", lens="experimental_design",
            question="What precision is the baseline?", retrieval_queries=["baseline precision"],
            evidence_requirements=[EvidenceRequirement(kind="baseline_precision")])
        plan = ReadingPlan(paper_id="p1", mode="targeted", user_prompt=query.question, subqueries=[query])
        packet = (await build_evidence_packets(ScopedEvidence(spans, {"p1"}), plan, None))[0]
        self.assertIn(cells[2].id, packet.evidence_ids)
        self.assertFalse(packet.missing_requirements)

        ragged = table.model_copy(update={"text": text.replace("| FP32 | 14 |", "| 14 |")})
        spans, objects = build_multimodal_catalog([ragged])
        self.assertEqual(len(spans), 1)
        self.assertEqual(objects[0].cell_evidence_ids, [])

    def test_explicit_multilevel_headers_and_malformed_spans(self):
        text = ("Table 7: Timings\n<table><tr><th rowspan='2'>Model</th>"
            "<th colspan='2'>Latency (ms)</th></tr><tr><th>Train</th><th>Eval</th></tr>"
            "<tr><th>A</th><td>12</td><td>8</td></tr></table>")
        span = EvidenceSpan(id="t", paper_id="p", version="v", text=text, source_level="full_text")
        spans, _ = build_multimodal_catalog([span])
        self.assertEqual([s.text for s in spans if s.table_cell],
            ["A | Latency (ms) / Train | 12", "A | Latency (ms) / Eval | 8"])
        for malformed in (text.replace("colspan='2'", "colspan='3'"), text.replace("rowspan='2'", "rowspan='0'")):
            spans, _ = build_multimodal_catalog([span.model_copy(update={"text": malformed})])
            self.assertFalse(any(s.table_cell for s in spans))

    def test_continued_table_inherits_headers_only_from_same_version_and_adjacent_page(self):
        first = EvidenceSpan(id="first", paper_id="p", version="1", source_version="pdf1",
            page_start=2, source_level="full_text",
            text="Table 3: Timings\n| Model | Latency (ms) |\n|---|---|\n| A | 12 |")
        second = first.model_copy(update={"id": "next", "page_start": 3,
            "text": "Table 3 (continued)\n| B | 15 |"})
        spans, _ = build_multimodal_catalog([second, first])
        cell, = [s for s in spans if s.table_cell and s.parent_id == "next"]
        self.assertEqual((cell.table_cell.row, cell.table_cell.column_header, cell.table_cell.value),
            (2, "Latency (ms)", "15"))
        self.assertEqual(cell.header_evidence_id, "first")
        for update in ({"page_start": 4}, {"source_version": "pdf2"},
                {"text": second.text.replace("continued", "other")}):
            spans, _ = build_multimodal_catalog([first, second.model_copy(update=update)])
            self.assertFalse(any(s.table_cell and s.parent_id == "next" for s in spans))

    async def test_packet_expands_from_text_reference_to_other_object_views(self):
        updated, _ = build_multimodal_catalog(self.evidence(),
            [{"page": 3, "captions": ["figure:2"]}])
        scope = ScopedEvidence(updated, {"p1"})
        scope.search = AsyncMock(return_value=[next(span for span in updated if span.id == "discussion")])
        query = ReadingSubquery(id="rq:figure", source="user", lens="experimental_design",
            question="What does Figure 2 show?", retrieval_queries=["Figure 2 latency"], priority=5,
            evidence_requirements=[EvidenceRequirement(kind="figure"), EvidenceRequirement(kind="result")])
        plan = ReadingPlan(paper_id="p1", mode="targeted", paper_type="system",
            user_prompt=query.question, user_questions=[query.question], subqueries=[query])
        packet = (await build_evidence_packets(scope, plan, None))[0]
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        self.assertIn("ocr", trace)
        self.assertEqual(trace["ocr"].relation, "same_object")
        self.assertIn("relation", trace["ocr"].retrieval_channels)

    def test_reference_without_caption_remains_explicitly_unlocated(self):
        span = EvidenceSpan(id="reference", paper_id="p1", version="1", source_level="full_text",
            text="See Table 7 for details.", page_start=2)
        _, objects = build_multimodal_catalog([span])
        self.assertEqual(find_object_labels(span.text), [("table:7", "table")])
        self.assertEqual(objects[0].label, "table:7")
        self.assertIsNone(objects[0].page)
        self.assertEqual(objects[0].caption, "")


if __name__ == "__main__":
    unittest.main()
