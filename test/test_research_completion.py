import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from src.llm.base import LLMResponse
from src.llm.anthropic import _convert_messages
from src.research.agents import ModelTasks, verify
from src.research.context import ScopedEvidence
from src.research.evaluation import load_dataset, case_version, human_labels, score, run_cases
from src.research.incremental import IncrementalPlan
from src.research.models import Claim, EvidenceSpan, ResearchOptions, PaperCard
from src.research.visual import prepare_visual, render_candidates, verification_images
from src.research.workflow import run_research_node
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.research import latest_report, revision_request
from test.test_research_workflow import FakeModel, item, state


class VisualModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.images = []

    async def chat(self, messages, **kwargs):
        content = messages[-1]['content']
        if isinstance(content, list):
            self.images.append([b for b in content if b['type'] == 'image_url'])
            messages = copy.deepcopy(messages)
            messages[-1]['content'] = content[0]['text']
        payload = json.loads(messages[-1]['content'])
        if payload['role'] == 'visual_transcription':
            self.requests.append(payload)
            return LLMResponse(content=json.dumps({'text': 'Table 1: accuracy 80%, baseline 70%.', 'readable': True, 'uncertainties': []}))
        result = await super().chat(messages, **kwargs)
        if payload['role'] == 'evidence_verifier' and payload.get('visual_pages'):
            value = json.loads(result.content)
            value['visual_observations'] = [{k: page[k] for k in ('paper_id', 'page', 'image_version')} |
                {'locator': 'Table 1, accuracy column', 'observation': 'The row reports accuracy 80%, baseline 70%.',
                 'value': '80', 'unit': '%', 'conditions': 'Table 1 experiment', 'status': 'supports'} for page in payload['visual_pages']]
            return LLMResponse(content=json.dumps(value))
        return result


def scanned_pdf(path, pages=1):
    image = Image.new('RGB', (600, 800), 'white')
    ImageDraw.Draw(image).text((30, 50), 'Table 1: accuracy 80%, baseline 70%', fill='black')
    image.save(path, format='PDF', save_all=True, append_images=[image] * (pages - 1))


class VisualTests(unittest.IsolatedAsyncioTestCase):
    async def test_scanned_pdf_runs_transcription_reader_and_original_image_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'scan.pdf'
            scanned_pdf(path)
            paper = item()
            paper['full_text']['source_path'] = str(path)
            model = VisualModel()
            report = (await run_research_node()(state(model, [paper])))['research_report']
            self.assertEqual(report['evidence'][0]['modality'], 'visual_transcription')
            self.assertEqual(report['evidence'][0]['page_start'], 1)
            self.assertEqual(len(model.images), 2)  # OCR and independent verification
            self.assertTrue(model.images[1][0]['image_url']['url'].startswith('data:image/png;base64,'))
            check = report['cards'][0]['verifications'][0]
            self.assertEqual(check['visual_pages'][0]['paper_id'], 'p1')
            self.assertNotIn('base64', json.dumps(report))
            self.assertEqual(check['verdict'], 'verified')

    async def test_page_limit_and_disabled_visual_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'scan.pdf'
            scanned_pdf(path, 3)
            pages, omitted = render_candidates(path, 1)
            self.assertEqual(len(pages), 1)
            self.assertEqual(omitted, [2, 3])
            paper = item()
            paper['full_text']['source_path'] = str(path)
            model = VisualModel()
            _, audit = await prepare_visual(ModelTasks(model, ResearchOptions(visual_verification=False)), paper, [])
            self.assertEqual(audit['status'], 'disabled')
            self.assertFalse(model.requests)

    async def test_missing_visual_cannot_be_verified(self):
        span = EvidenceSpan(id='e', paper_id='p', version='v', text='Table 1: 80%', source_level='full_text', page_start=1)
        claim = Claim(id='c', paper_id='p', version='v', statement='Table 1 shows 80%', kind='result', evidence_ids=['e'])
        result = await verify(ModelTasks(FakeModel(), ResearchOptions()), claim, ScopedEvidence([span], {'p'}))
        self.assertEqual(result.verdict, 'needs_human_review')
        self.assertTrue(result.unchecked)

    def test_visual_scope_and_image_versions_are_enforced(self):
        tasks = ModelTasks(FakeModel(), ResearchOptions())
        tasks.visual_pages[('foreign', 1)] = {'url': 'secret', 'image_version': 'v', 'source_version': 's'}
        span = EvidenceSpan(id='e', paper_id='p', version='v', text='Table 1', source_level='full_text', page_start=1, image_version='wrong')
        claim = Claim(id='c', paper_id='p', version='v', statement='Table 1', kind='result', evidence_ids=['e'])
        images, _, missing = verification_images(tasks, [span], claim)
        self.assertFalse(images)
        self.assertTrue(missing)
        tasks.visual_pages[('p', 1)] = {'url': 'image', 'image_version': 'correct', 'source_version': 's'}
        self.assertFalse(verification_images(tasks, [span], claim)[0])

    async def test_corrupt_pdf_is_reported_without_claiming_visual_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.pdf'
            path.write_text('not a PDF')
            paper = item()
            paper['full_text']['source_path'] = str(path)
            _, audit = await prepare_visual(ModelTasks(FakeModel(), ResearchOptions()), paper, [])
            self.assertEqual(audit['status'], 'unavailable')
            self.assertTrue(audit['warnings'])

    def test_anthropic_receives_native_image_blocks(self):
        _, messages = _convert_messages([{'role': 'user', 'content': [{'type': 'text', 'text': 'inspect'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YWJj'}}]}])
        self.assertEqual(messages[0]['content'][1], {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'YWJj'}})


class ParagraphModel(FakeModel):
    async def chat(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if payload['role'] == 'writer':
            self.requests.append(payload)
            return LLMResponse(content=json.dumps({'paragraphs': [{'text': c['statement'], 'claim_ids': [c['id']]} for c in payload['claims']]}))
        return await super().chat(messages, **kwargs)


class IncrementalTests(unittest.IsolatedAsyncioTestCase):
    async def revision_run(self, directory, model=None, cross=True, review=True):
        repo = SQLiteSessionRepository(storage_root=directory)
        session = repo.create()
        options = dict(cross_paper_enabled=cross, output_mode='review' if review else 'paper_report')
        task = state(model or ParagraphModel(), [item(), item('p2'), item('p3')], **options)
        task.update(session_repo=repo, session_key=session.key, turn_id='first')
        initial = (await run_research_node()(task))['research_report']
        latest = latest_report(repo, session.key)
        old = initial['cards'][0]['claims'][0]
        checkpoint = revision_request(repo, session.key, {'artifact_id': latest['artifact_id'], 'claim_id': old['id'], 'target_version': old['version'], 'reason': '限定范围', 'changes': {'statement': '修订后的结论'}})['read_resume_checkpoint']
        second_model = ParagraphModel()
        task = state(second_model, checkpoint['read_results'], **options)
        task.update(session_repo=repo, session_key=session.key, turn_id='second', read_resume_checkpoint=checkpoint)
        updated = (await run_research_node()(task))['research_report']
        return initial, updated, second_model

    async def test_revision_reuses_unrelated_pairs_and_paragraphs(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, model = await self.revision_run(d)
            roles = Counter(p['role'] for p in model.requests)
            self.assertEqual(roles['paper_reader'], 0)
            self.assertEqual(roles['cross_paper_analyst'], 2)
            self.assertEqual(after['incremental']['reused']['cards'], 2)
            self.assertEqual(after['incremental']['reused']['pairs'], 1)
            # Every paragraph saw an incident relation to the edited claim; all must refresh.
            self.assertEqual(after['incremental']['reused']['paragraphs'], 0)
            self.assertLess(after['calls'], before['calls'])
            self.assertTrue(all(p['paragraph']['relation_context'] for p in after['paragraphs']))
            self.assertEqual(before['cards'][1:], after['cards'][1:])
            self.assertEqual(len(after['pair_outcomes']), 3)
            self.assertTrue(after['dependencies'])
            original = before['cards'][0]['claims'][0]['id']
            self.assertTrue(all(original not in r['claim_ids'] for r in after['relations']))
            written = [c for p in model.requests if p['role'] == 'writer' for c in p['claims']]
            self.assertEqual(len(written), 3)

    async def test_no_cross_means_only_changed_claim_is_reverified(self):
        with tempfile.TemporaryDirectory() as d:
            _, after, model = await self.revision_run(d, cross=False, review=False)
            self.assertEqual([p['role'] for p in model.requests], ['evidence_verifier'])
            self.assertEqual(after['relations'], [])
            self.assertEqual(after['pair_outcomes'], [])

    async def test_policy_or_source_change_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            before, _, _ = await self.revision_run(d)
            evidence = [EvidenceSpan.model_validate(e) for e in before['evidence']]
            plan = IncrementalPlan(before, evidence, 'different-policy')
            self.assertIsNone(plan.card('p2'))
            changed = [e.model_copy(update={'version': 'changed'}) if e.paper_id == 'p2' else e for e in evidence]
            plan = IncrementalPlan(before, changed, before['execution_policy'])
            self.assertIsNone(plan.card('p2'))
            self.assertIsNotNone(plan.card('p3'))
            first = Claim.model_validate(before['cards'][1]['claims'][0])
            second = Claim.model_validate(before['cards'][2]['claims'][0])
            self.assertIsNone(plan.pair(first, second))

    async def test_empty_comparison_is_reused_and_new_pairs_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            before, _, _ = await self.revision_run(d)
            evidence = [EvidenceSpan.model_validate(e) for e in before['evidence']]
            first = Claim.model_validate(before['cards'][0]['claims'][0])
            second = Claim.model_validate(before['cards'][1]['claims'][0])
            before['pair_outcomes'][0].update(relations=[], verifications=[])
            plan = IncrementalPlan(before, evidence, before['execution_policy'])
            self.assertEqual(plan.pair(first, second), ([], []))
            self.assertIsNone(plan.pair(first.model_copy(update={'id': 'new', 'version': 'new'}), second))
            cards = [PaperCard.model_validate(c) for c in before['cards']]
            cards[0].verifications[0].reason = '重新核查后，理由改变'
            plan.bind_cards(cards)
            self.assertIsNone(plan.pair(first, second))


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sources, self.cases = load_dataset('evaluation/real-papers.json')

    def review(self, reviewer='Expert A'):
        case = self.cases[0]
        return {'case_id': case.id, 'case_version': case_version(case, self.sources), 'reviewer': reviewer, 'expertise': 'NLP',
            'reviewed_at': '2026-11-21', 'origin': 'human', 'source_ids': case.source_ids, 'labels': case.reference.model_dump(exclude_none=True), 'rationale': 'Checked the original source.'}

    def test_drafts_are_not_expert_labels_and_missing_predictions_count(self):
        result = score(self.sources, self.cases, [])
        self.assertEqual(result['expert_consensus']['cases'], 0)
        self.assertEqual(result['expert_status'], 'pending_human_annotation')
        self.assertEqual(result['reference_draft']['accuracy']['fidelity'], 0)

    def test_expert_consensus_requires_two_distinct_agreeing_reviewers(self):
        self.assertFalse(human_labels(self.cases, self.sources, [self.review()])[0])
        first, second = self.review(), self.review('Expert B')
        self.assertEqual(len(human_labels(self.cases, self.sources, [first, second])[0]), 1)
        with self.assertRaises(ValueError):
            human_labels(self.cases, self.sources, [first, first])
        second['labels']['fidelity'] = 'unknown'
        agreed, unresolved = human_labels(self.cases, self.sources, [first, second])
        self.assertFalse(agreed)
        self.assertEqual(unresolved, [self.cases[0].id])

    def test_stale_reviews_and_prediction_versions_are_rejected(self):
        review = self.review()
        review['case_version'] = 'old'
        with self.assertRaises(ValueError):
            human_labels(self.cases, self.sources, [review])
        with self.assertRaises(ValueError):
            score(self.sources, self.cases, [{'case_id': self.cases[0].id, 'case_version': 'old', 'labels': {}}])

    async def test_runner_calls_real_verifier_and_analyst_without_label_leakage(self):
        model = FakeModel()
        predictions = await run_cases(self.sources, self.cases, model)
        self.assertEqual(len(predictions), 9)
        self.assertTrue(all(p['calls'] == 1 for p in predictions))
        self.assertFalse(any('reference' in p or 'annotation_status' in p for p in model.requests))
        result = score(self.sources, self.cases, predictions)
        self.assertEqual(result['reference_draft']['predicted_cases'], 9)
        self.assertEqual(result['expert_consensus']['cases'], 0)

class CompletionRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_claim_in_edited_paper_reuses_verifier(self):
        class TwoClaims(FakeModel):
            async def chat(self, messages, **kwargs):
                response = await super().chat(messages, **kwargs)
                if json.loads(messages[-1]['content'])['role'] == 'paper_reader':
                    value = json.loads(response.content)
                    another = {**value['claims'][0], 'statement': '第二个独立判断'}
                    value['claims'].append(another)
                    return LLMResponse(content=json.dumps(value))
                return response
        helper = IncrementalTests()
        with tempfile.TemporaryDirectory() as d:
            before, after, model = await helper.revision_run(d, model=TwoClaims(), cross=False, review=False)
            self.assertEqual(len(after['cards'][0]['claims']), 2)
            self.assertEqual([p['role'] for p in model.requests], ['evidence_verifier'])
            self.assertEqual(after['incremental']['reused']['claim_checks'], 1)
            self.assertEqual(before['cards'][0]['verifications'][1], after['cards'][0]['verifications'][1])

    async def test_scanned_transcription_reuse_is_bound_to_original_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'scan.pdf'
            scanned_pdf(path)
            paper = item()
            paper['full_text']['source_path'] = str(path)
            tasks = ModelTasks(VisualModel(), ResearchOptions())
            evidence, _ = await prepare_visual(tasks, paper, [])
            second = ModelTasks(VisualModel(), ResearchOptions())
            second.previous_visual_evidence = [e.model_dump() for e in evidence]
            same, audit = await prepare_visual(second, paper, [])
            self.assertEqual(same, evidence)
            self.assertEqual(second.calls, 0)
            self.assertTrue(audit['pages'][0]['transcription_reused'])
            image = Image.new('RGB', (600, 800), 'black')
            image.save(path, format='PDF')
            changed, _ = await prepare_visual(second, paper, [])
            self.assertEqual(second.calls, 1)
            self.assertNotEqual(changed[0].version, evidence[0].version)

    async def test_unread_scan_pages_prevent_complete_paper_scores(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'scan.pdf'
            scanned_pdf(path, 2)
            paper = item()
            paper['full_text']['source_path'] = str(path)
            result = await run_research_node()(state(VisualModel(), [paper], visual_page_limit=1))
            card = result['research_report']['cards'][0]
            self.assertIn('p1:unread-visual-page:2', card['omitted_evidence_ids'])
            self.assertTrue(card['warnings'])

    def test_live_evaluation_without_model_has_clear_error(self):
        from unittest.mock import patch
        from src.research.evaluation import main
        with patch('sys.argv', ['evaluation', 'run']), patch('src.agents.readAgent.load_read_agent_llm', return_value=None):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
