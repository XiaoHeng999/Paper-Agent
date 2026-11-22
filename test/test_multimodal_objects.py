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
