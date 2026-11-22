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
from src.repositories.sessions.sqlite import SQLiteSessionRepository
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
        if role == 'local_gap_review':
            self.requests.append(p)
            return LLMResponse(content=json.dumps({'decision': 'external_needed', 'reason':'本文未给独立同预算实验',
                                                    'evidence_ids':[p['evidence'][0]['id']]}))
        if role not in {'argument_builder', 'research_synthesis', 'reasoning_verifier'}:
            return await super().chat(messages, **kwargs)
        self.requests.append(p)
        if role == 'argument_builder':
            c = p['claims'][0]
            argument = {'claim_id': c['id'], 'evidence_ids': c['evidence_ids'], 'thesis': c['statement'],
                'inference': '同预算消融支持该机制贡献，但未排除实现因素', 'conditions': 'matched budget',
                'alternative_explanation': 'kernel implementation', 'falsification': 'matched kernels remove gain',
                'conclusion': '有条件支持，依据 '+c['evidence_ids'][0], 'gaps': [{'question':'是否有独立同预算消融', 'why_material':'区分机制与实现收益',
                    'hypothesis':'补偿能减少误差', 'prediction':'匹配预算时误差降低',
                    'weakening_observation':'匹配预算后误差没有降低','comparable_conditions':'相同核与稀疏率',
                    'search_query':'matched budget routing correction ablation', 'stopping_rule':'找到同预算消融或保留未知'}] if self.gap else []}
            value = {'arguments':[argument]}
        elif role == 'research_synthesis':
            args = p['arguments']
            external = [e for e in p['evidence'] if e['paper_id'] in p['external_candidate_ids']]
            support = [{'gap_id':p['evidence_questions'][0]['gap_id'], 'argument_index':0, 'evidence_ids':[external[0]['id']], 'relation':'supports',
                'comparability':'same task, budget unknown', 'contribution':'independent mechanism observation'}] if external else []
            guide = [{'title':'研究问题', 'explanation':'先比较相同预算下的误差。',
                      'claim_ids':[args[0]['claim_id']], 'evidence_ids': args[0]['evidence_ids']}]
            value = {'reading_guide':guide, 'arguments':args, 'external_support':support, 'insights':[{
                'claim_ids':[args[0]['claim_id']], 'argument_indices':[0], 'kind':'boundary', 'epistemic_status':'hypothesis',
                'statement':'共享代理信号可能产生共同失效条件', 'derivation':'路由和补偿复用同一信号，因此可能共同受噪声影响',
                'evidence_ids': ['invented'] if self.bad_citation else args[0]['evidence_ids'],
                'prediction':'噪声增大时两者误差同步升高','weakening_observation':'干预后两者误差独立变化','uncertainty':'未做噪声干预', 'distinguishing_test':'固定预算注入代理噪声并分别关闭路由和补偿'}]}
            if self.foreign_argument:
                value['insights'][0]['argument_indices'] = [99]
        else:
            groups = {'argument':'arguments','insight':'insights','external_support':'external_support', 'reading_passage':'reading_guide'}
            value = {'checks':[{'target_type':kind,'index':i,'verdict':'supported','reason':'raw evidence matches derivation',
                'evidence_ids':x['evidence_ids']} for kind, field in groups.items() for i,x in enumerate(p['synthesis'][field])]}
            if self.incomplete_review:
                value['checks'] = value['checks'][:1]
        return LLMResponse(content=json.dumps(value,ensure_ascii=False))


class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    def task(self, model, **constraints):
        return state(model, research_reasoning_enabled=True, **constraints)

    async def test_method_reconstruction_is_reviewed_against_primary_fulltext(self):
        class MethodModel(ReasoningModel):
            async def chat(model, messages, **kwargs):
                response = await super(MethodModel, model).chat(messages, **kwargs)
                payload = json.loads(messages[-1]['content'])
                value = json.loads(response.content)
                if payload['role'] == 'research_synthesis':
                    value['method_steps'] = [{
                        'operation':'池化后再路由', 'input':'查询块与键块', 'output':'候选块进入精算',
                        'design_reason':'减少排序开销', 'basis':'approximation',
                        'justification':'用块均值近似，依赖块内差异较小', 'failure_or_test':'固定掩码增大块内差异',
                        'evidence_ids':[payload['evidence'][0]['id']]}]
                    value['flow_edges'] = [
                        {'source': a, 'target': b, 'transfer': label, 'claim_ids':[value['arguments'][0]['claim_id']],
                         'evidence_ids': [payload['evidence'][0]['id']]}
                        for a, b, label in [('查询', '路由', '块均值'), ('路由', '精算', '入选索引'), ('精算', '输出', '加权值')]]
                    value['reading_guide'][0]['figure_number'] = 4
                if payload['role'] == 'reasoning_verifier' and payload['synthesis'].get('method_steps'):
                    for i, edge in enumerate(payload['synthesis']['flow_edges']):
                        value['checks'].append({'target_type':'flow_edge', 'index':i,
                                                'verdict':'qualified' if i == disputed else 'supported',
                                                'reason':'检查路由流向', 'evidence_ids':edge['evidence_ids']})
                    value['checks'].append({'target_type':'method_step','index':0,'verdict':'qualified',
                                            'reason':'需要控制块内差异', 'evidence_ids':payload['synthesis']['method_steps'][0]['evidence_ids']})
                response.content = json.dumps(value)
                return response

        for level, disputed in [('full_text', None), ('full_text', 0), ('full_text', 1), ('abstract', None)]:
            source = EvidenceSpan(id='s', paper_id='p1', version='1', source_level=level, page_start=5, page_end=5,
                                  text='Figure 4 | 使用池化键路由')
            with self.subTest(level=level, disputed=disputed), patch('src.research.workflow.load_evidence', return_value=[source]):
                output = await run_research_node()(self.task(MethodModel()))
                result = output['research_report']['reasoning'][0]
                self.assertEqual(result['status'], 'reviewed')
                if level == 'full_text':
                    if disputed == 0:
                        self.assertEqual(result['flow_scope'], 'partial')
                        self.assertEqual([e['review_index'] for e in result['flow_edges']], [1, 2])
                    self.assertEqual(result['reading_figures'][0]['page'], 5)
                    self.assertEqual('对应图中路径' in output['research_markdown'], disputed != 1)
                    self.assertEqual('```mermaid' in output['research_markdown'], disputed != 1)
                    self.assertEqual(result['method_steps'][0]['evidence_ids'], ['s'])
                    self.assertEqual(next(c['verdict'] for c in result['checks'] if c['target_type']=='method_step'), 'qualified')
                    self.assertIn('近似及其条件', output['research_markdown'])
                    self.assertLess(output['research_markdown'].index('算法如何工作'), output['research_markdown'].index('推理桥梁'))
                else:
                    self.assertFalse(result['method_steps'])
                    self.assertNotIn('算法如何工作', output['research_markdown'])
                    self.assertTrue(result['arguments'])

    async def test_reading_guide_precedes_analysis_and_scopes_its_conclusions(self):
        class GuideModel(ReasoningModel):
            async def chat(self, messages, **kwargs):
                response = await super().chat(messages, **kwargs)
                payload = json.loads(messages[-1]['content'])
                value = json.loads(response.content)
                if payload['role'] == 'research_synthesis':
                    arg = value['arguments'][0]
                    value['reading_guide'] = [
                        {'title': title, 'explanation': explanation, 'claim_ids': [arg['claim_id']],
                         'evidence_ids': arg['evidence_ids']}
                        for title, explanation in [('预算与误差', '本文在固定预算下观察误差。'),
                                                   ('同预算对照', '相同预算才能解释两组误差之差。')]]
                elif payload['role'] == 'reasoning_verifier':
                    for check in value['checks']:
                        if check['target_type'] == 'reading_passage':
                            check.update(verdict='qualified', reason='预算匹配仍待核查')
                response.content = json.dumps(value)
                return response

        source = EvidenceSpan(id='s', paper_id='p1', version='1', source_level='full_text', text='固定预算下比较误差')
        with patch('src.research.workflow.load_evidence', return_value=[source]):
            output = await run_research_node()(self.task(GuideModel()))
        result = output['research_report']['reasoning'][0]
        self.assertEqual(result['status'], 'reviewed')
        report = output['research_markdown']
        self.assertLess(report.index('论文总结 · 预算与误差'), report.index('讲解 1 · 同预算对照'))
        self.assertLess(report.index('讲解 1 · 同预算对照'), report.index('推理桥梁'))
        self.assertEqual(report.count('承接前文：预算与误差、同预算对照'), 2)
        insight = next(c for c in result['checks'] if c['target_type'] == 'insight')
        self.assertEqual(insight['verdict'], 'qualified')
        self.assertIn('关联讲解', insight['reason'])

    async def test_reading_guide_with_an_invented_source_is_withheld(self):
        class InvalidGuide(ReasoningModel):
            async def chat(self, messages, **kwargs):
                response = await super().chat(messages, **kwargs)
                if json.loads(messages[-1]['content'])['role'] == 'research_synthesis':
                    value = json.loads(response.content)
                    value['reading_guide'][0]['evidence_ids'] = ['unknown-source']
                    response.content = json.dumps(value)
                return response

        output = await run_research_node()(self.task(InvalidGuide()))
        result = output['research_report']['reasoning'][0]
        self.assertFalse(result['reading_guide'])
        self.assertTrue(result['arguments'])
        self.assertIn('部分论证缺少对应的前文讲解。', result['warnings'])
        self.assertNotIn('论文总结 ·', output['research_markdown'])

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
        self.assertIn('保留判断', support_check['judgment_update'])
        self.assertEqual(r['reasoning'][0]['external_support'][0]['gap_id'], r['research_questions'][0]['gap_id'])
        self.assertEqual(len(r['references']),2)

    async def test_local_answer_uncertainty_and_invalid_citations_do_not_trigger_search(self):
        class LocalReview(ReasoningModel):
            async def chat(model, messages, **kwargs):
                payload = json.loads(messages[-1]['content'])
                if payload['role'] == 'local_gap_review':
                    eid = payload['evidence'][0]['id'] if decision != 'external_needed' else 'unseen'
                    return LLMResponse(content=json.dumps({'decision': decision, 'reason':'已回查消融条件',
                                                          'evidence_ids':[eid]}))
                return await super(LocalReview, model).chat(messages, **kwargs)
        for decision, expected in [('answered_here','not_needed'), ('unresolved','unresolved'),
                                   ('external_needed','unresolved')]:
            with self.subTest(decision=decision):
                task = self.task(LocalReview(), allow_external_search=True)
                search = AsyncMock()
                task['research_search_service'] = SimpleNamespace(async_search=search)
                output = await run_research_node()(task)
                report = output['research_report']
                search.assert_not_called()
                self.assertEqual(report['external_search']['status'], expected)
                gap = report['reasoning'][0]['evidence_questions'][0]
                self.assertFalse(gap['candidate_ids'])
                self.assertTrue(gap['comparable_conditions'])
                self.assertIn(report['reasoning'][0]['insights'][0]['weakening_observation'],
                              output['research_markdown'])

    async def test_reviewed_fulltext_can_strengthen_or_weaken_a_specific_hypothesis(self):
        class LinkedInsight(ReasoningModel):
            async def chat(model, messages, **kwargs):
                p = json.loads(messages[-1]['content'])
                response = await super(LinkedInsight, model).chat(messages, **kwargs)
                if p['role'] == 'research_synthesis':
                    data = json.loads(response.content)
                    support = data['external_support'][0]
                    support['relation'] = relation
                    insight = data['insights'][0]
                    if relation == 'rejected':
                        support['relation'] = 'supports'
                        insight['epistemic_status'] = 'supported_inference'
                    insight['gap_ids'] = [support['gap_id']]
                    insight['evidence_ids'].extend(support['evidence_ids'])
                    response.content = json.dumps(data)
                if p['role'] == 'reasoning_verifier' and relation == 'rejected':
                    data = json.loads(response.content)
                    for check in data['checks']:
                        if check['target_type'] == 'external_support':
                            check['verdict'] = 'unsupported'
                    response.content = json.dumps(data)
                return response
        def fulltext(item):
            pid = item['paper']['paperId']
            return [EvidenceSpan(id=pid+'_span', paper_id=pid, version='v1', text='Matched ablation',
                                 source_level='full_text')]
        for relation, update in [('supports','增强判断'), ('challenges','削弱判断'),
                                 ('different_conditions','条件不同，保留判断'),
                                 ('rejected','佐证尚未充分通过核查，保留判断')]:
            with self.subTest(relation=relation):
                task = self.task(LinkedInsight(), allow_external_search=True)
                task['research_search_service'] = SimpleNamespace(async_search=AsyncMock(return_value=SimpleNamespace(
                    errors={}, papers=[PaperDocument(id='ext', paperId='ext', title='Independent experiment')])))
                with patch('src.research.workflow.load_evidence', side_effect=fulltext), patch(
                        'src.research.supplement.async_download_paper_fulltext', AsyncMock(return_value=SimpleNamespace(file_path=None))):
                    output = await run_research_node()(task)
                result = output['research_report']['reasoning'][0]
                gid = result['evidence_questions'][0]['gap_id']
                self.assertEqual(result['insights'][0]['gap_ids'], [gid])
                self.assertIn('ext_span', result['insights'][0]['evidence_ids'])
                self.assertEqual(result['external_support'][0]['gap_id'], gid)
                if relation == 'rejected':
                    self.assertEqual(result['insights'][0]['epistemic_status'], 'hypothesis')
                    self.assertEqual(next(c['verdict'] for c in result['checks'] if c['target_type'] == 'insight'), 'qualified')
                self.assertIn('对判断的影响：'+update, output['research_markdown'])

    async def test_shared_query_and_transient_failure_preserve_candidates_on_resume(self):
        gaps = [{'claim_id': 'claim', 'gap_id': gid, 'paper_id': 'p1', 'search_query': query}
                for gid, query in [('g1','query one'), ('g2','query  one'), ('g3','query two')]]
        task = self.task(ReasoningModel(), allow_external_search=True)
        task['research_challenges'] = gaps
        paper = PaperDocument(id='ext', title='First match')
        match = SimpleNamespace(papers=[paper], errors={})
        empty = SimpleNamespace(papers=[], errors={})
        search = AsyncMock(side_effect=[match, TimeoutError(), empty])
        task['research_search_service'] = SimpleNamespace(async_search=search)
        tasks = ModelTasks(ReasoningModel(), ResearchOptions(allow_external_search=True))
        download = AsyncMock(return_value=SimpleNamespace(file_path=None))
        with patch('src.research.supplement.async_download_paper_fulltext', download):
            items, audit = await supplement_sources(task, [item()], tasks)
            self.assertEqual(audit['status'], 'partial')
            self.assertEqual(audit['queries'], ['query one','query two'])
            self.assertEqual(items[-1]['acquisition_gap_ids'], ['g1','g2'])
            task['read_resume_checkpoint'] = {'external_search': audit}
            items, audit = await supplement_sources(task, items, tasks)
            self.assertEqual(audit['status'], 'completed')
            self.assertEqual(audit['added_paper_ids'], ['ext'])
            task['read_resume_checkpoint'] = {'external_search': audit}
            await supplement_sources(task, items, tasks)
        self.assertEqual(search.await_count, 3)
        self.assertEqual(search.await_args.kwargs['query'], 'query two')
        self.assertEqual(audit['candidates_by_gap'], {'g1':['ext'], 'g2':['ext']})
        download.assert_awaited_once()
        # A new search policy may exclude an old candidate even with unchanged gap IDs.
        task['request'].constraints['year_from'] = 2025
        search.side_effect = [empty, empty]
        _, filtered = await supplement_sources(task, items, tasks)
        self.assertEqual(filtered['candidates_by_gap'], {})
        self.assertEqual(filtered['added_paper_ids'], [])

    async def test_two_gaps_cannot_exchange_candidates_even_within_one_argument(self):
        class Swapped(ReasoningModel):
            async def chat(model, messages, **kwargs):
                p = json.loads(messages[-1]['content'])
                response = await super(Swapped, model).chat(messages, **kwargs)
                data = json.loads(response.content)
                if p['role'] == 'argument_builder':
                    gaps = data['arguments'][0]['gaps']
                    gaps.append({**gaps[0], 'question': '换硬件后如何？', 'search_query': 'matched hardware'})
                if p['role'] == 'research_synthesis':
                    by_id = {e['id']: e for e in p['evidence']}
                    support = data['external_support'][0]
                    source = by_id[support['evidence_ids'][0]]['paper_id']
                    support['gap_id'] = next(q['gap_id'] for q in p['evidence_questions']
                                             if source not in q['candidate_ids'])
                response.content = json.dumps(data)
                return response
        task = self.task(Swapped(), allow_external_search=True, external_paper_limit=2)
        service = SimpleNamespace(async_search=AsyncMock(side_effect=[
            SimpleNamespace(errors={}, papers=[PaperDocument(id=pid, paperId=pid, title=pid, abstract='comparison')])
            for pid in ('ext1', 'ext2')]))
        task['research_search_service'] = service
        with patch('src.research.supplement.async_download_paper_fulltext', AsyncMock(return_value=SimpleNamespace(file_path=None))):
            report = (await run_research_node()(task))['research_report']
        self.assertEqual(service.async_search.await_count, 2)
        result = report['reasoning'][0]
        self.assertEqual([q['candidate_ids'] for q in result['evidence_questions']], [['ext1'], ['ext2']])
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['external_support'])
        self.assertIn('不匹配', result['warnings'][-1])

    async def test_repeated_claim_does_not_spend_last_synthesis_calls(self):
        class DuplicateReader(ReasoningModel):
            async def chat(model, messages, **kwargs):
                p = json.loads(messages[-1]['content'])
                response = await super(DuplicateReader, model).chat(messages, **kwargs)
                if p['role'] == 'paper_reader':
                    data = json.loads(response.content)
                    data['claims'] *= 2
                    response.content = json.dumps(data)
                return response
        model = DuplicateReader(gap=False)
        report = (await run_research_node()(self.task(model, research_max_calls=6)))['research_report']
        self.assertEqual(len(report['cards'][0]['claims']), 1)
        self.assertEqual(sum(r['role'] == 'evidence_verifier' for r in model.requests), 1)
        self.assertEqual(report['reasoning'][0]['status'], 'reviewed')

    async def test_core_question_is_reviewed_and_saved_before_secondary_verification(self):
        class RankedReader(ReasoningModel):
            async def chat(model, messages, **kwargs):
                payload = json.loads(messages[-1]['content'])
                if payload['role'] == 'paper_reader':
                    model.requests.append(payload)
                    eid = payload['primary_evidence_ids'][0]
                    return LLMResponse(content=json.dumps({
                        'claims': [{'statement': text, 'evidence_ids': [eid]}
                                   for text in ('次要信息', '首要结果', '适用范围')],
                        'core_questions': [{'claim_index': 1, 'question': '实验支持主要结果吗？'},
                                           {'claim_index': 2, 'question': '适用范围是什么？'}]}))
                if payload['role'] == 'evidence_verifier' and payload['target']['statement'] == '次要信息':
                    # Inspect persistent session state while the pipeline is still running.
                    ready = [a for a in repo.get(session.key).artifacts if a['artifact_type'] == 'core_argument']
                    self.assertEqual(len(ready), 2)
                    model.requests.append(payload)
                    return LLMResponse(content='', finish_reason='error', error_status_code=402)
                return await super(RankedReader, model).chat(messages, **kwargs)

        for limit, expected in [(4, 1), (40, 2)]:
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                repo = SQLiteSessionRepository(storage_root=directory)
                session = repo.create()
                model = RankedReader()
                task = self.task(model, research_max_calls=limit, research_max_repair_rounds=0)
                task.update(session_repo=repo, session_key=session.key, turn_id='core')
                output = await run_research_node()(task)
                report = output['research_report']
                self.assertLessEqual(report['calls'], limit)
                snapshots = report['core_reviews']
                self.assertEqual(len(snapshots), expected)
                first = snapshots[0]
                self.assertEqual(first['question'], '实验支持主要结果吗？')
                self.assertEqual(first['claim']['statement'], '首要结果')
                self.assertEqual(first['review']['verdict'], 'qualified')
                self.assertEqual(report['first_core_review_seconds'], first['elapsed_seconds'])
                artifacts = [a for a in repo.get(session.key).artifacts if a['artifact_type'] == 'core_argument']
                self.assertEqual(len(artifacts), expected)
                saved = repo.read_artifact_path(session.key, artifacts[0]['id']).read_text()
                self.assertIn(first['question'], saved)
                self.assertIn(first['sources'][0]['text'], saved)
                self.assertIn(first['question'], output['research_markdown'])
                self.assertEqual([p['role'] for p in model.requests[:4]],
                                 ['paper_reader', 'evidence_verifier', 'argument_builder', 'reasoning_verifier'])

    async def test_citable_paraphrase_requires_scoped_review_and_a_matching_passage(self):
        # Controlled figure-7 fixture: the first latency is the system baseline;
        # the second already includes fusion and caching.
        excerpt = "Experiment E1, HunyuanVideo on B200: baseline 866.9 s, fused and cached 328.4 s, full system 170.6 s."
        sentence = "作者报告，在 B200 的 HunyuanVideo 实验中，完整系统相对初始基线的端到端加速为 5.08 倍。"
        span = EvidenceSpan(id='p1:figure7', paper_id='p1', version='v1', source_level='full_text',
                            section='System integration', page_start=8, text=excerpt)

        class CitationModel(ReasoningModel):
            async def chat(model, messages, **kwargs):
                p = json.loads(messages[-1]['content'])
                response = await super(CitationModel, model).chat(messages, **kwargs)
                data = json.loads(response.content)
                if p['role'] == 'paper_reader':
                    data['claims'][0]['statement'] = sentence
                if p['role'] == 'evidence_verifier':
                    for proof in data.get('fact_citations', []):
                        if '5.08' in proof['clause']:
                            proof['ratio'] = {'numerator': '866.9 s', 'denominator': '170.6 s'}
                if p['role'] == 'argument_builder':
                    arg = data['arguments'][0]
                    arg['citation_draft'] = {
                        'sentence': sentence if case != 'wrong attribution' else sentence.replace('完整系统', '单独使用 Sol-Attn'),
                        'scope': 'B200 的 HunyuanVideo 实验',
                        'comparison': {'baseline': '上一配置' if case == 'missing baseline' else '初始基线',
                                       'metric': '端到端加速'},
                        'overreach': '若归于 Sol-Attn 单项，比较起点应改为已融合和缓存的配置。',
                        'excerpt': 'unreported outcome' if case == 'invented passage' else excerpt,
                        'evidence_id': arg['evidence_ids'][0]}
                if p['role'] == 'reasoning_verifier':
                    for check in data['checks']:
                        if check['target_type'] == 'argument':
                            check['citation_verdict'] = 'withhold' if case == 'wrong attribution' else 'comparison'
                response.content = json.dumps(data)
                return response

        for case in ('valid', 'missing baseline', 'invented passage', 'wrong attribution', 'abstract only'):
            with self.subTest(case=case):
                source = span.model_copy(update={'source_level': 'abstract'}) if case == 'abstract only' else span
                with patch('src.research.workflow.load_evidence', return_value=[source]):
                    output = await run_research_node()(self.task(CitationModel()))
                report = output['research_report']
                saved = report['core_reviews'][0]['review']['citation']
                final = next(c for c in report['reasoning'][0]['checks'] if c['target_type'] == 'argument')['citation']
                self.assertEqual(saved, final)
                if case == 'valid':
                    self.assertEqual(final['sentence'], sentence)
                    self.assertEqual((final['page_start'], final['excerpt']), (8, excerpt))
                    self.assertIn('可引用转述：' + sentence, output['research_markdown'])
                    self.assertIn(final['overreach'], output['research_markdown'])
                else:
                    self.assertIsNone(final)
                    self.assertNotIn('可引用转述：', output['research_markdown'])

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
        reused=await synthesize(tasks,'方法有效吗',card,r['argument_plans'][0],evidence,set(),r['execution_policy'],**kwargs)
        self.assertTrue(reused['reused']);self.assertEqual(tasks.calls,0)
        evidence.append(EvidenceSpan(id='new',paper_id=card.paper_id,version='new',text='new result',source_level='full_text'))
        changed=await synthesize(tasks,'方法有效吗',card,r['argument_plans'][0],evidence,set(),r['execution_policy'],**kwargs)
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
