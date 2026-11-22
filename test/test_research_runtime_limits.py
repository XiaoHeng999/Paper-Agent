"""Regression cases from the Sol-Attn local Codex run."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse
from src.research.agents import BudgetExceeded, ModelTasks, verify
from src.research.context import ScopedEvidence
from src.research.image_retrieval import GmeIndex
from src.research.models import Claim, EvidenceSpan, PaperCard, ResearchOptions, VerificationDraft
from src.research.scheduling import reading_schedule
from src.research.workflow import render_card, run_research_node
from test.test_research_workflow import FakeModel, item, state


def evidence_case():
    span = EvidenceSpan(id='e', paper_id='p', version='v', text='The method works on small samples.', source_level='full_text')
    claim = Claim(id='c', paper_id='p', version='v', kind='result', statement='The method works.', evidence_ids=['e'])
    return claim, ScopedEvidence([span], {'p'})


class RuntimeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_time_and_call_limits_are_distinct_and_do_not_call_model(self):
        model = FakeModel()
        tasks = ModelTasks(model, ResearchOptions(max_calls=2))
        with patch('src.research.agents.time.monotonic', return_value=tasks.deadline + 1):
            with self.assertRaisesRegex(BudgetExceeded, '时间预算'):
                await tasks.ask('paper_reader', '', {}, VerificationDraft)
        tasks.calls = 2
        with self.assertRaisesRegex(BudgetExceeded, '调用预算'):
            await tasks.ask('paper_reader', '', {}, VerificationDraft)
        self.assertEqual(model.requests, [])

    async def test_reserved_time_stops_reading_but_allows_synthesis(self):
        tasks = ModelTasks(FakeModel(), ResearchOptions(timeout_seconds=1800))
        tasks.reading_time_reserve = 600
        with patch('src.research.agents.time.monotonic', return_value=tasks.deadline - 500):
            with self.assertRaisesRegex(BudgetExceeded, '报告综合与复审'):
                tasks.remaining_budget('evidence_verifier')
            self.assertEqual(tasks.remaining_budget('research_synthesis'), 500)

    async def test_exhausted_verification_does_not_retrieve_or_claim_evidence_failure(self):
        claim, scope = evidence_case()
        scope.adversarial = AsyncMock(side_effect=AssertionError('retrieved after budget exhaustion'))
        tasks = ModelTasks(FakeModel(), ResearchOptions(max_calls=1))
        tasks.calls = 1
        result = await verify(tasks, claim, scope)
        self.assertEqual(result.execution_status, 'budget_exhausted')
        scope.adversarial.assert_not_called()
        card = PaperCard(paper_id='p', title='Paper', version='v', claims=[claim], verifications=[result])
        rendered = render_card(card, scope)
        self.assertIn('核查：未完成核查', rendered)
        self.assertNotIn('核查：证据不足', rendered)

    async def test_timeout_is_identified_and_underlying_call_is_cancelled(self):
        cancelled = asyncio.Event()
        class Slow:
            async def chat(self, *args, **kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
        tasks = ModelTasks(Slow(), ResearchOptions())
        with patch.object(tasks, 'remaining_budget', return_value=0.01):
            with self.assertRaisesRegex(TimeoutError, 'research_synthesis 模型调用超时'):
                await tasks.ask('research_synthesis', '', {}, VerificationDraft)
        self.assertTrue(cancelled.is_set())

    async def test_invalid_output_is_failed_not_completed_verification(self):
        claim, scope = evidence_case()
        tasks = ModelTasks(FakeModel(malformed=True), ResearchOptions())
        result = await verify(tasks, claim, scope)
        self.assertEqual(result.execution_status, 'failed')
        self.assertIn('本次语义核查未完成', result.unchecked)

    async def test_fact_citation_schema_limits_clauses_without_mutating_shared_schema(self):
        schemas = []
        class Capture:
            async def chat(self, messages, **kwargs):
                schemas.append(json.loads(messages[0]['content'].split('JSON：', 1)[1]))
                return LLMResponse(content=json.dumps({'verdict':'insufficient_evidence','reason':'Missing evidence','evidence_ids':[]}))
        tasks = ModelTasks(Capture(), ResearchOptions())
        await tasks.ask('evidence_verifier', '', {'checkable_clauses':['Only this clause']}, VerificationDraft)
        await tasks.ask('evidence_verifier', '', {'checkable_clauses':[]}, VerificationDraft)
        self.assertEqual(schemas[0]['$defs']['FactCitation']['properties']['clause']['enum'], ['Only this clause'])
        self.assertEqual(schemas[1]['properties']['fact_citations']['maxItems'], 0)
        self.assertNotIn('enum', VerificationDraft.model_json_schema()['$defs']['FactCitation']['properties']['clause'])

    async def test_report_preserves_incomplete_status_and_independent_image_config(self):
        class TimeoutModel(FakeModel):
            async def chat(self, messages, **kwargs):
                if json.loads(messages[-1]['content'])['role'] == 'evidence_verifier':
                    raise TimeoutError()
                return await super().chat(messages, **kwargs)
        task = state(TimeoutModel())
        task['research_evidence_index'] = None
        result = await run_research_node()(task)
        report = result['research_report']
        self.assertEqual(report['verification_execution'], {'timed_out':1})
        self.assertEqual(report['retrieval']['status'], 'not_configured')
        self.assertEqual(report['image_retrieval']['status'], 'no_candidates')
        self.assertIn('gme-Qwen2-VL', report['image_retrieval']['model'])
        self.assertIn('未配置文本向量索引', result['research_markdown'])
        self.assertIn('核查：未完成核查', result['research_markdown'])

    async def test_exhausted_reader_records_all_remaining_batches_once(self):
        model = FakeModel()
        task = state(model, [item(abstract='Source text about the method. ' * 600)],
                     research_context_chars=4000, research_max_calls=1)
        task['research_evidence_index'] = None
        report = (await run_research_node()(task))['research_report']
        card = report['cards'][0]
        self.assertEqual(report['calls'], 1)
        first_source = model.requests[0]['primary_evidence_ids']
        remaining = {s['id'] for s in report['evidence']} - set(first_source)
        self.assertTrue(remaining)
        self.assertTrue(remaining <= set(card['omitted_evidence_ids']))
        self.assertEqual(sum('调用预算' in warning for warning in card['warnings']), 1)
        self.assertEqual(card['verifications'][0]['execution_status'], 'budget_exhausted')

    async def test_image_diagnostics_distinguish_no_trigger_from_no_candidates(self):
        index = GmeIndex()
        self.assertEqual(index.diagnostics()['status'], 'not_triggered')
        from src.research.models import ReadingSubquery
        query = ReadingSubquery(id='q', source='user', lens='method_mechanism', question='Which figure has green arrows?',
                               retrieval_queries=['green arrows'], evidence_requirements=[{'kind':'figure'}])
        await index.search(query, ScopedEvidence([], {'p'}), {})
        self.assertEqual(index.diagnostics()['status'], 'no_candidates')
        self.assertEqual(index.diagnostics()['queries'], 1)
        self.assertEqual(index.diagnostics()['encoded_crops'], 0)

    def test_reading_schedule_keeps_core_and_spreads_remaining_kinds_without_dropping_claims(self):
        claim, _ = evidence_case()
        claims = [claim.model_copy(update={'id':str(i),'kind':kind}) for i,kind in enumerate(
            ['method','method','method','result','result','limitation'])]
        ordered = reading_schedule(claims, ['0'])
        self.assertEqual([c.id for c in ordered], ['0','1','3','5','2','4'])
        self.assertEqual({c.id for c in ordered}, {c.id for c in claims})
