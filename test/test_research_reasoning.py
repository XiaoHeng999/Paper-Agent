import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.llm.base import LLMResponse
from src.paper_retrieval.models import PaperDocument
from src.research.agents import ModelTasks, read_claims
from src.research.context import ScopedEvidence
from src.research.models import Claim, EvidenceSpan, ResearchOptions
from src.research.visual import verification_images
from src.research.workflow import run_research_node
from src.research.reasoning import synthesize
from src.research.models import PaperCard
from src.research.supplement import supplement_sources
from test.test_research_workflow import FakeModel, item, state


class ReasoningModel(FakeModel):
    def __init__(self, *, gap=True, bad_citation=False, incomplete_review=False, foreign_argument=False):
        super().__init__()
        self.gap, self.bad_citation = gap, bad_citation
        self.incomplete_review, self.foreign_argument = incomplete_review, foreign_argument

    async def chat(self, messages, **kwargs):
        p = json.loads(messages[-1]['content'])
        role = p['role']
        if role not in {'argument_builder', 'research_synthesis', 'reasoning_verifier'}:
            return await super().chat(messages, **kwargs)
        self.requests.append(p)
        if role == 'argument_builder':
            c = p['claims'][0]
            argument = {'claim_id': c['id'], 'evidence_ids': c['evidence_ids'], 'thesis': c['statement'],
                'inference': '同预算消融支持该机制贡献，但未排除实现因素', 'conditions': 'matched budget',
                'alternative_explanation': 'kernel implementation', 'falsification': 'matched kernels remove gain',
                'conclusion': '有条件支持，依据 '+c['evidence_ids'][0], 'gaps': [{'question':'是否有独立同预算消融', 'why_material':'区分机制与实现收益',
                    'search_query':'matched budget routing correction ablation', 'stopping_rule':'找到同预算消融或保留未知'}] if self.gap else []}
            value = {'arguments':[argument]}
        elif role == 'research_synthesis':
            args = p['arguments']
            external = [e for e in p['evidence'] if e['paper_id'] in p['external_candidate_ids']]
            support = [{'argument_index':0, 'evidence_ids':[external[0]['id']], 'relation':'supports',
                'comparability':'same task, budget unknown', 'contribution':'independent mechanism observation'}] if external else []
            value = {'arguments':args, 'external_support':support, 'insights':[{
                'claim_ids':[args[0]['claim_id']], 'argument_indices':[0], 'kind':'boundary', 'epistemic_status':'hypothesis',
                'statement':'共享代理信号可能产生共同失效条件', 'derivation':'路由和补偿复用同一信号，因此可能共同受噪声影响',
                'evidence_ids': ['invented'] if self.bad_citation else args[0]['evidence_ids'],
                'uncertainty':'未做噪声干预', 'distinguishing_test':'固定预算注入代理噪声并分别关闭路由和补偿'}]}
            if self.foreign_argument:
                value['insights'][0]['argument_indices'] = [99]
        else:
            groups = {'argument':'arguments','insight':'insights','external_support':'external_support'}
            value = {'checks':[{'target_type':kind,'index':i,'verdict':'supported','reason':'raw evidence matches derivation',
                'evidence_ids':x['evidence_ids']} for kind, field in groups.items() for i,x in enumerate(p['synthesis'][field])]}
            if self.incomplete_review:
                value['checks'] = value['checks'][:1]
        return LLMResponse(content=json.dumps(value,ensure_ascii=False))


class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    def task(self, model, **constraints):
        return state(model, research_reasoning_enabled=True, **constraints)

    async def test_single_paper_reasoning_and_insights_are_delivered_without_search(self):
        model = ReasoningModel()
        task = self.task(model)
        service = SimpleNamespace(async_search=AsyncMock())
        task['research_search_service'] = service
        result = await run_research_node()(task)
        r = result['research_report']
        self.assertFalse(r['cross_paper_executed'])
        self.assertEqual(r['reasoning'][0]['status'], 'reviewed')
        self.assertEqual(r['reasoning'][0]['insights'][0]['epistemic_status'], 'hypothesis')
        self.assertIn('推理桥梁', result['research_markdown'])
        self.assertIn('验证实验', result['research_markdown'])
        self.assertIn('p1:abstract:',r['reasoning'][0]['arguments'][0]['conclusion'])
        self.assertEqual(r['research_questions'][0]['stopping_rule'], '找到同预算消融或保留未知')
        service.async_search.assert_not_awaited()
        # Abstracts never acquire a fully supported reasoning label.
        self.assertTrue(all(c['verdict']=='qualified' for c in r['reasoning'][0]['checks']))

    async def test_external_corroboration_does_not_start_pairwise_comparison(self):
        model = ReasoningModel()
        task = self.task(model, allow_external_search=True)
        paper = PaperDocument(id='ext',paperId='ext',title='Independent ablation',abstract='same task, different budget')
        service = SimpleNamespace(async_search=AsyncMock(return_value=SimpleNamespace(errors={},papers=[paper])))
        task['research_search_service'] = service
        with patch('src.research.supplement.async_download_paper_fulltext', AsyncMock(return_value=SimpleNamespace(file_path=None))):
            r = (await run_research_node()(task))['research_report']
        self.assertFalse(r['cross_paper_executed'])
        self.assertFalse(r['relations'])
        self.assertEqual(len(r['cards']),1)
        self.assertEqual(r['primary_paper_ids'],['p1'])
        self.assertEqual(len(r['reasoning'][0]['external_support']),1)
        self.assertEqual(service.async_search.call_args.kwargs['query'],'matched budget routing correction ablation')
        self.assertNotIn('cross_paper_analyst',[p['role'] for p in model.requests])
        support_check = next(c for c in r['reasoning'][0]['checks'] if c['target_type']=='external_support')
        self.assertEqual(support_check['verdict'],'qualified')
        self.assertEqual(len(r['references']),2)

    async def test_payment_rejection_stops_further_calls_and_blocks_search(self):
        class PaymentRejected(ReasoningModel):
            async def chat(self, messages, **kwargs):
                self.requests.append(json.loads(messages[-1]['content']))
                return LLMResponse(content='', finish_reason='error', error_status_code=402)
        model=PaymentRejected()
        task=self.task(model,allow_external_search=True)
        service=SimpleNamespace(async_search=AsyncMock());task['research_search_service']=service
        result=await run_research_node()(task)
        r=result['research_report']
        self.assertEqual(len(model.requests),1)
        self.assertEqual(r['calls'],1)
        self.assertEqual(r['external_search']['status'],'blocked')
        self.assertIn('HTTP 402',r['external_search']['warnings'][0])
        self.assertEqual(r['reasoning'][0]['status'],'unavailable')
        service.async_search.assert_not_called()
        self.assertTrue(result['research_markdown'])

    async def test_failed_planning_is_not_reported_as_no_need_for_evidence(self):
        class MalformedPlanning(ReasoningModel):
            async def chat(self,messages,**kwargs):
                if json.loads(messages[-1]['content'])['role']=='argument_builder':
                    return LLMResponse(content='invalid json')
                return await super().chat(messages,**kwargs)
        task=self.task(MalformedPlanning(),allow_external_search=True)
        service=SimpleNamespace(async_search=AsyncMock());task['research_search_service']=service
        r=(await run_research_node()(task))['research_report']
        self.assertEqual(r['external_search']['status'],'planning_unavailable')
        service.async_search.assert_not_called()

    async def test_primary_arxiv_version_is_not_accepted_as_external_corroboration(self):
        task=self.task(ReasoningModel(),allow_external_search=True)
        task['research_challenges']=[{'claim_id':'c','statement':'S','premise':'P','required_test':'T','paper_id':'arxiv:2607.24027'}]
        original={'paper':{'id':'2607.24027','paperId':'arxiv:2607.24027','title':'Primary'},'full_text':{}}
        same=PaperDocument(id='2607.24027v2',paperId='2607.24027v2',title='Primary newer version')
        task['research_search_service']=SimpleNamespace(async_search=AsyncMock(return_value=SimpleNamespace(errors={},papers=[same])))
        with patch('src.research.supplement.async_download_paper_fulltext',AsyncMock()) as download:
            items,audit=await supplement_sources(task,[original],ModelTasks(ReasoningModel(),ResearchOptions(allow_external_search=True)))
        self.assertEqual(len(items),1);self.assertFalse(audit['added_paper_ids']);download.assert_not_called()

    async def test_no_material_gap_means_no_search_even_when_allowed(self):
        task=self.task(ReasoningModel(gap=False),allow_external_search=True)
        service=SimpleNamespace(async_search=AsyncMock());task['research_search_service']=service
        r=(await run_research_node()(task))['research_report']
        self.assertEqual(r['external_search']['status'],'not_needed')
        service.async_search.assert_not_called()

    async def test_invalid_insight_citation_or_argument_index_cannot_be_published(self):
        for kwargs in ({'bad_citation':True},{'foreign_argument':True}):
            with self.subTest(kwargs=kwargs):
                r=(await run_research_node()(self.task(ReasoningModel(**kwargs))))['research_report']['reasoning'][0]
                self.assertEqual(r['status'],'unavailable')
                self.assertFalse(r['insights'])
                self.assertTrue(r['warnings'])

    async def test_incomplete_independent_review_keeps_draft_unreviewed(self):
        r=(await run_research_node()(self.task(ReasoningModel(incomplete_review=True))))['research_report']['reasoning'][0]
        self.assertEqual(r['status'],'unreviewed')
        self.assertFalse(r['checks'])
        self.assertTrue(r['insights'])

    async def test_primary_paper_context_does_not_include_other_primary_paper(self):
        task=state(ReasoningModel(),[item('p1','UNIQUE_A'),item('p2','UNIQUE_B')],research_reasoning_enabled=True)
        model=task['research_node_llm']
        await run_research_node()(task)
        for p in model.requests:
            if p['role']=='research_synthesis':
                self.assertEqual({e['paper_id'] for e in p['evidence']},{p['primary_paper_id']})

    async def test_budget_exhaustion_preserves_explicit_unavailable_reasoning(self):
        r=(await run_research_node()(self.task(ReasoningModel(),research_max_calls=1)))['research_report']
        self.assertLessEqual(r['calls'],1)
        self.assertEqual(r['reasoning'][0]['status'],'unavailable')
        self.assertTrue(r['reasoning'][0]['warnings'])

    async def test_unconfirmed_supported_inference_is_relabelled_as_hypothesis(self):
        class Overconfident(ReasoningModel):
            async def chat(self,messages,**kwargs):
                response=await super().chat(messages,**kwargs)
                p=json.loads(messages[-1]['content'])
                if p['role']=='research_synthesis':
                    value=json.loads(response.content)
                    value['insights'][0]['epistemic_status']='supported_inference'
                    response.content=json.dumps(value)
                return response
        r=(await run_research_node()(self.task(Overconfident())))['research_report']['reasoning'][0]
        self.assertEqual(r['insights'][0]['epistemic_status'],'hypothesis')
        self.assertEqual(r['insights'][0]['proposed_epistemic_status'],'supported_inference')

    async def test_reasoning_reuse_requires_all_current_source_dependencies(self):
        r=(await run_research_node()(self.task(ReasoningModel())))['research_report']
        card=PaperCard.model_validate(r['cards'][0])
        evidence=[EvidenceSpan.model_validate(e) for e in r['evidence']]
        tasks=ModelTasks(ReasoningModel(),ResearchOptions())
        kwargs={'previous':r['reasoning'][0], 'source_manifests':r['source_manifests']}
        reused=await synthesize(tasks,'Q',card,r['argument_plans'][0],evidence,set(),r['execution_policy'],**kwargs)
        self.assertTrue(reused['reused']);self.assertEqual(tasks.calls,0)
        evidence.append(EvidenceSpan(id='new',paper_id=card.paper_id,version='new',text='new result',source_level='full_text'))
        changed=await synthesize(tasks,'Q',card,r['argument_plans'][0],evidence,set(),r['execution_policy'],**kwargs)
        self.assertNotIn('reused',changed);self.assertEqual(tasks.calls,2)
        self.assertNotEqual(changed['version'],reused['version'])

    async def test_reader_overflow_keeps_valid_claims_and_marks_coverage_incomplete(self):
        class Overflow(FakeModel):
            async def chat(self,messages,**kwargs):
                p=json.loads(messages[-1]['content'])
                return LLMResponse(content=json.dumps({'claims':[{'statement':str(i),'evidence_ids':[p['evidence'][0]['id']]} for i in range(27)]}))
        span=EvidenceSpan(id='e',paper_id='p',version='v',text='controlled trial',source_level='full_text')
        tasks=ModelTasks(Overflow(),ResearchOptions())
        claims=await read_claims(tasks,'Q','p',ScopedEvidence([span],{'p'}),['e'])
        self.assertEqual(len(claims),25)
        self.assertEqual(tasks.reading_gaps['p'],{'e'})

    async def test_targeted_repair_requests_one_claim_schema(self):
        span=EvidenceSpan(id='e',paper_id='p',version='v',text='small sample',source_level='full_text')
        model=FakeModel();tasks=ModelTasks(model,ResearchOptions())
        claims=await read_claims(tasks,'Q','p',ScopedEvidence([span],{'p'}),['e'],feedback='限定小样本')
        self.assertEqual(len(claims),1)
        self.assertEqual(claims[0].conditions,'小样本')

    def test_incidental_counterevidence_image_does_not_block_text_claim(self):
        tasks=ModelTasks(FakeModel(),ResearchOptions())
        a=EvidenceSpan(id='a',paper_id='p',version='v',text='method definition',source_level='full_text',page_start=1)
        b=EvidenceSpan(id='b',paper_id='p',version='v',text='Figure 9 ablation',source_level='full_text',page_start=9)
        target=Claim(id='c',paper_id='p',version='v',statement='method definition',evidence_ids=['a'])
        images,_,missing=verification_images(tasks,[a,b],target)
        self.assertFalse(images);self.assertFalse(missing)
        target.evidence_ids=['b']
        self.assertTrue(verification_images(tasks,[a,b],target)[2])
