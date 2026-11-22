import unittest
import tempfile
import json
from pathlib import Path

from src.paper_retrieval.models import PaperDocument
from src.research.context import ScopedEvidence
from src.research.models import Claim, EvidenceSpan
from src.research.planning import (
    assign_subqueries, build_reading_plan, coverage_for_plan, prioritize_for_plan,
)
from src.research.storage import load_evidence
from src.utils.read_utils.chunkers import StructureAwareChunker, build_chunks_file


class ReadingPlanTests(unittest.IsolatedAsyncioTestCase):
    def test_generic_prompt_uses_bounded_default_plan(self):
        plan = build_reading_plan("阅读这篇论文", "p1", "An Efficient Attention Method")
        self.assertEqual(plan.mode, "open_ended")
        self.assertEqual(plan.user_questions, [])
        self.assertEqual(len(plan.subqueries), 6)
        self.assertEqual(len({query.lens for query in plan.subqueries}), 6)
        self.assertTrue(plan.warnings)

    def test_targeted_question_is_preserved_and_prioritized(self):
        prompt = "该方法的延迟优势是否在相同硬件和 batch size 下成立？"
        plan = build_reading_plan(prompt, "p1", "Fast Inference System")
        self.assertEqual(plan.mode, "targeted")
        self.assertEqual(plan.user_questions, [prompt.rstrip("？")])
        self.assertEqual(plan.subqueries[0].source, "user")
        self.assertEqual(plan.subqueries[0].priority, 5)
        self.assertEqual(plan.subqueries[0].lens, "experimental_design")
        self.assertLessEqual(len(plan.subqueries), 8)

    def test_paper_type_changes_default_lenses(self):
        plan = build_reading_plan("总结", "p1", "A Survey of Multimodal Retrieval")
        self.assertEqual(plan.paper_type, "survey")
        self.assertNotIn("causal_validation", {query.lens for query in plan.subqueries})

    def test_claim_binding_rejects_foreign_subquery(self):
        plan = build_reading_plan("阅读", "p1", "Method")
        with self.assertRaises(ValueError):
            assign_subqueries("result", "reported result", plan, ["foreign"])
        selected = assign_subqueries("method", "the method transforms inputs", plan, [])
        lookup = {query.id: query for query in plan.subqueries}
        self.assertEqual(lookup[selected[0]].lens, "method_mechanism")

    async def test_priority_round_robin_recovers_different_lenses_without_embeddings(self):
        spans = [
            EvidenceSpan(id="method", paper_id="p1", version="1", source_level="full_text",
                text="The architecture transforms the input through an attention module."),
            EvidenceSpan(id="experiment", paper_id="p1", version="1", source_level="full_text",
                text="Experimental setup uses dataset D, baseline B, latency and GPU hardware."),
            EvidenceSpan(id="limitation", paper_id="p1", version="1", source_level="full_text",
                text="Limitations include failure on long sequences and missing implementation details."),
        ]
        plan = build_reading_plan("阅读", "p1", "Method")
        priority = await prioritize_for_plan(ScopedEvidence(spans, {"p1"}), plan, None, per_query=1)
        self.assertIn("method", {span.id for span in priority})
        self.assertIn("experiment", {span.id for span in priority})
        self.assertIn("limitation", {span.id for span in priority})

    def test_coverage_is_conservative_until_typed_requirements_exist(self):
        plan = build_reading_plan("阅读", "p1", "Method")
        target = plan.subqueries[0]
        claim = Claim(id="c", paper_id="p1", version="1", statement="claim",
            evidence_ids=["e"], subquery_ids=[target.id])
        coverage = {item.subquery_id: item for item in coverage_for_plan(plan, [claim])}
        self.assertEqual(coverage[target.id].status, "partial")
        self.assertTrue(coverage[target.id].missing_requirements)
        self.assertIn("unanswered", {item.status for item in coverage.values()})


class StructureAwareChunkerTests(unittest.TestCase):
    def test_long_page_splits_on_sentences_with_bounded_overlap(self):
        sentence = "The method transforms an input and preserves the stated condition. "
        markdown = "<!-- page: 1 -->\n\n1 Method\n\n" + sentence * 45
        chunks = StructureAwareChunker().chunk(PaperDocument(id="p1", paperId="p1", title="P"), markdown)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.content) <= StructureAwareChunker.max_chunk_characters for chunk in chunks))
        self.assertTrue(all(chunk.section == "1 Method" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["boundary_policy"] == "paragraph_sentence_then_safe_cut" for chunk in chunks))
        self.assertGreater(chunks[1].metadata["overlap_characters"], 0)
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].chunk_id)

    def test_chunk_relations_survive_evidence_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            markdown_path = Path(directory) / "paper.md"
            markdown_path.write_text("<!-- page: 1 -->\n\n1 Method\n\n" +
                "The method applies an operation. " * 60, encoding="utf-8")
            paper = PaperDocument(id="p1", paperId="p1", title="P")
            build_chunks_file(paper, markdown_path=markdown_path)
            evidence = load_evidence({"paper": {"id": "p1", "paperId": "p1", "title": "P"},
                "full_text": {"markdown_path": str(markdown_path)}})
            self.assertGreater(len(evidence), 1)
            self.assertEqual(evidence[0].next_id, evidence[1].id)
            self.assertEqual(evidence[1].previous_id, evidence[0].id)
            self.assertTrue(evidence[0].parent_id)
            neighbors, _ = ScopedEvidence(evidence, {"p1"}).neighbors([evidence[0].id], 4000)
            self.assertIn(evidence[1], neighbors)

    def test_legacy_oversized_chunk_keeps_internal_span_relations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_markdown = root / "missing.md"
            payload = [{
                "chunkId": "p1:p0001", "paperId": "p1", "chunk_index": 0,
                "content": "x" * 8500, "page_start": 1, "page_end": 1,
                "section": "page_1", "previous_chunk_id": None,
                "next_chunk_id": None, "metadata": {"chunker": "legacy"},
            }]
            (root / "chunk.json").write_text(json.dumps(payload), encoding="utf-8")
            evidence = load_evidence({"paper": {"id": "p1", "paperId": "p1", "title": "P"},
                "full_text": {"markdown_path": str(missing_markdown)}})
            self.assertEqual(len(evidence), 3)
            self.assertEqual(evidence[0].next_id, evidence[1].id)
            self.assertEqual(evidence[1].previous_id, evidence[0].id)
            self.assertEqual(evidence[1].next_id, evidence[2].id)


if __name__ == "__main__":
    unittest.main()
