"""Paper synthesis, review recovery and report rendering."""
import re
from functools import lru_cache
from inspect import getsource

from . import publication
from .agents import BudgetExceeded
from .publication import publish, reviewed_citation, validate_review
from .synthesis_tasks import ReasoningCheck, ReasoningReview, generate_draft, review_draft
# The original reasoning module remains the import path for existing research scripts.
from .argumentation import (
    CITATION_TASK, METHOD_BASIS, Argument, ArgumentBatch, CitationDraft, ComparisonTerms,
    EvidenceGap, ExternalSupport, FlowEdge, Insight, LocalGapReview, MethodStep, ReadingPassage,
    Synthesis, TestablePrediction, ask_scoped, check_local_gap, check_summary, connected_flow,
    plan_arguments, research_questions, select_evidence, validate_arguments,
)
from .models import version_of


@lru_cache(maxsize=1)
def stage_versions():
    return {'draft': version_of([getsource(generate_draft), getsource(synthesize), Synthesis.model_json_schema()]),
            'review': version_of([getsource(review_draft), getsource(validate_review), ReasoningReview.model_json_schema()]),
            'publication': version_of(getsource(publication))}


async def synthesize(tasks, question, card, plan, evidence, external_ids, policy, *, previous=None, source_manifests=None, argument_only=False, store=None):
    local = [e for e in evidence if e.paper_id == card.paper_id]
    external = [e for e in evidence if e.paper_id in external_ids]
    question_map = {q['gap_id']: q for q in plan.get('evidence_questions', [])}
    preferred = [eid for a in plan['arguments'] for eid in a['evidence_ids']]
    if not argument_only:
        preferred += [eid for c in card.claims if c.kind == 'method' for eid in c.evidence_ids]
    preferred += [eid for q in question_map.values() for eid in q['local_review']['evidence_ids']]
    # Give external candidates space without replacing the primary paper's evidence.
    selected = select_evidence(local, preferred, tasks.options.context_chars * (2 if external else 3) // 3)
    terms = set(re.findall(r'[a-z]{3,}', ' '.join(g['search_query'] for a in plan['arguments'] for g in a['gaps']).lower()))
    external.sort(key=lambda e: (-sum(term in e.text.lower() for term in terms), e.source_level != 'full_text'))
    selected += select_evidence(external, [], tasks.options.context_chars - sum(len(e.text) for e in selected))
    supplied = {e.id: e for e in selected}
    planned_ids = {a['claim_id'] for a in plan['arguments']}
    claims = {c.id: c for c in card.claims if c.id in planned_ids}
    result = {'paper_id': card.paper_id, 'status': 'unavailable', 'arguments': plan['arguments'],
              'evidence_questions': list(question_map.values()), 'insights': [], 'external_support': [], 'checks': [], 'warnings': list(plan['warnings']),
              'omitted_evidence_ids': [e.id for e in [*local, *external] if e.id not in supplied],
              'dependencies': {'policy': policy, 'question': question, 'claims': {c.id: c.version for c in card.claims},
                 'evidence': {e.id: e.version for e in selected}, 'claim_checks': version_of([v.model_dump() for v in card.verifications]),
                 'plan': version_of(plan), 'argument_only': argument_only, 'sources': {pid: (source_manifests or {}).get(pid) for pid in {card.paper_id, *external_ids}},
                 'evidence_order': list(supplied), 'available_evidence': {e.id: e.version for e in [*local, *external]}}}
    trace = {'paper_id': card.paper_id}
    if not argument_only:
        tasks.recovery_trace.append(trace)
    policies = stage_versions()
    result['stage_policies'] = policies
    if previous and previous.get('stage_policies') == policies and previous.get('status') == 'reviewed' and previous.get('dependencies') == result['dependencies']:
        covered = {cid for p in previous.get('reading_guide', []) for cid in p['claim_ids']}
        if argument_only or all(a['claim_id'] in covered for a in previous['arguments']):
            trace['publication'] = 'matching_parent_report'
            return {**previous, 'reused': True}
    if not plan['arguments']:
        result['version'] = version_of(result)
        return result
    try:
        draft_key = version_of([question, card.paper_id, result['dependencies'], policies['draft']])
        cached_draft = await store.load(draft_key, 'reasoning_draft') if store else None
        if cached_draft and cached_draft.get('version') != version_of([draft_key, cached_draft.get('draft')]):
            cached_draft = None
        trace['draft'] = 'matched' if cached_draft else 'no_matching_draft'
        if cached_draft:
            draft = Synthesis.model_validate(cached_draft['draft'])
            result['warnings'] = list(cached_draft['warnings'])
        elif argument_only:
            draft = Synthesis(arguments=plan['arguments'])
        else:
            draft = await generate_draft(tasks, question, card, plan, claims, supplied, question_map, external_ids)
        if not draft.arguments:
            raise ValueError('综合步骤未返回论证，保留原计划待审')
        primary_fulltext = {e.id for e in selected if e.paper_id == card.paper_id and e.source_level == 'full_text'}
        if any(not set(step.evidence_ids) <= primary_fulltext for step in draft.method_steps):
            result['warnings'].append('方法重建未展示：至少一步缺少本次提供的主论文全文依据')
            draft.method_steps = []
        argument_claims = {a.claim_id for a in draft.arguments}
        if draft.flow_edges and (not connected_flow(draft.flow_edges) or not draft.method_steps or not draft.reading_guide or any(
                edge.source == edge.target or not set(edge.evidence_ids) <= primary_fulltext
                or not set(edge.claim_ids) <= argument_claims for edge in draft.flow_edges)):
            draft.flow_edges = []
            result['warnings'].append('教学图缺少可定位的方法或原文依据，已省略。')
        primary_ids = {e.id for e in selected if e.paper_id == card.paper_id}
        if any(not set(p.claim_ids) <= argument_claims or not set(p.evidence_ids) <= primary_ids for p in draft.reading_guide):
            draft.reading_guide = []
            result['warnings'].append('讲解引用了未关联的主张或主论文以外的片段，已撤下讲解。')
        covered = {cid for p in draft.reading_guide for cid in p.claim_ids}
        if not argument_only and argument_claims - covered:
            result['warnings'].append('部分论证缺少对应的前文讲解。')
        validate_arguments(draft.arguments, claims, supplied)
        for insight in draft.insights:
            if insight.prediction.casefold() == insight.weakening_observation.casefold():
                raise ValueError('预测与削弱判断的观察必须可区分')
            if not set(insight.claim_ids) <= claims.keys() or not set(insight.evidence_ids) <= supplied.keys() or any(i < 0 or i >= len(draft.arguments) for i in insight.argument_indices):
                raise ValueError('insight引用了未知论证、主张或证据')
            if not set(insight.claim_ids) <= {draft.arguments[i].claim_id for i in insight.argument_indices}:
                raise ValueError('insight主张与所关联论证不一致')
            if any(gid not in question_map or question_map[gid]['claim_id'] not in insight.claim_ids for gid in insight.gap_ids):
                raise ValueError('见解关联了其他主张的证据缺口')
        for support in draft.external_support:
            if not 0 <= support.argument_index < len(draft.arguments) or not set(support.evidence_ids) <= supplied.keys() or not any(supplied[eid].paper_id in external_ids for eid in support.evidence_ids):
                raise ValueError('外部佐证必须引用已读取的外部原文且绑定论证')
            gap = question_map.get(support.gap_id)
            external_sources = {supplied[eid].paper_id for eid in support.evidence_ids} & external_ids
            if (not gap or gap['claim_id'] != draft.arguments[support.argument_index].claim_id
                    or not external_sources <= set(gap['candidate_ids'])):
                raise ValueError('外部证据与原检索缺口或主张不匹配')
        for index, argument in enumerate(draft.arguments):
            linked = {eid for support in draft.external_support if support.argument_index == index for eid in support.evidence_ids}
            if any(supplied[eid].paper_id in external_ids and eid not in linked for eid in argument.evidence_ids):
                raise ValueError('论证使用外部证据但未说明其贡献')
        for insight in draft.insights:
            linked = {eid for support in draft.external_support if support.gap_id in insight.gap_ids
                      and support.argument_index in insight.argument_indices for eid in support.evidence_ids}
            if any(supplied[eid].paper_id in external_ids and eid not in linked for eid in insight.evidence_ids):
                raise ValueError('见解的外部证据未关联对应缺口的判断')
        result['warnings'] = list(dict.fromkeys(result['warnings']))
        draft_version = version_of([draft_key, draft.model_dump()])
        if store and not cached_draft:
            accepted = await store.accept('reasoning_draft', {
                'paper_id': card.paper_id, 'question': question, 'version': draft_version, 'policy': policies['draft'],
                'dependencies': result['dependencies'], 'evidence_ids': list(supplied),
                'omitted_evidence_ids': result['omitted_evidence_ids'], 'warnings': result['warnings'],
                'draft': draft.model_dump()}, draft_key)
            draft = Synthesis.model_validate(accepted['draft'])
            draft_version = accepted['version']
            result['warnings'] = list(accepted['warnings'])
        result['draft_version'] = draft_version
        review_key = version_of([draft_version, policies['review']])
        result.update(draft.model_dump(exclude={'flow_edges'}), status='unreviewed')
        cached_review = await store.load(review_key, 'reasoning_review') if store else None
        if cached_review and (cached_review.get('draft_version') != draft_version or
                              cached_review.get('version') != version_of([review_key, cached_review.get('review')])):
            cached_review = None
        trace['review'] = 'matched' if cached_review else 'no_matching_review'
        if cached_review:
            review = ReasoningReview.model_validate(cached_review['review'])
        else:
            review = await review_draft(tasks, card, draft, question_map, claims, supplied)
        raw_review = review.model_dump()
        groups, missing_edges = validate_review(draft, review, supplied)
        review_version = version_of([review_key, raw_review])
        if store and not cached_review:
            accepted = await store.accept('reasoning_review', {'draft_version': draft_version, 'version': review_version,
                                          'policy': policies['review'], 'review': raw_review}, review_key)
            if accepted['version'] != review_version:
                review_version = accepted['version']
                review = ReasoningReview.model_validate(accepted['review'])
                groups, missing_edges = validate_review(draft, review, supplied)
        result['review_version'] = review_version
        publication_key = version_of([review_version, policies['publication']])
        cached_publication = await store.load(publication_key, 'reasoning_publication') if store else None
        if cached_publication:
            trace['publication'] = 'matched'
            return cached_publication
        trace['publication'] = 'no_matching_publication'
        publish(result, draft, review, groups, missing_edges, claims, supplied, card, external_ids)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        if result.get('draft_version'):
            if tasks.terminal_error:
                reason = 'provider_blocked'
            elif isinstance(exc, BudgetExceeded):
                reason = 'budget_exhausted'
            elif isinstance(exc, TimeoutError):
                reason = 'timeout'
            else:
                reason = 'invalid_or_failed_review'
            result['pending_review'] = reason
        result['warnings'].append(str(exc) or type(exc).__name__)
    publication.qualify_insights(result)
    result['version'] = version_of(result)
    if store and result['status'] == 'reviewed':
        return await store.accept('reasoning_publication', result, publication_key)
    return result


def render_citation(check):
    citation = check.get('citation')
    if not citation:
        return ['引用状态：尚无通过核查的转述']
    location = citation['section'] or '原文片段'
    if citation['page_start']:
        location = f"第 {citation['page_start']} 页 · {location}"
    return ['可引用转述：' + citation['sentence'], '引用边界：' + citation['overreach'],
            f"决定性原文（{location}，{citation['evidence_id']}）：{citation['excerpt']}"]


def render_flow(edges, partial=False):
    # Only generated node identifiers and arrows are syntax. User/model text stays literal.
    def label(text):
        return ''.join(c if c.isalnum() or c == ' ' else f'#{ord(c)};' for c in text)
    names = list(dict.fromkeys(n for edge in edges for n in (edge['source'], edge['target'])))
    graph = ['```mermaid', 'flowchart TD']
    graph.extend(f'  N{i}["{label(name)}"]' for i, name in enumerate(names))
    for edge in edges:
        graph.append(f'  N{names.index(edge["source"])} -->|"{label(edge["transfer"])}"| N{names.index(edge["target"])}')
    return ['### ' + ('局部数据流' if partial else '算法数据流') + ' · 系统教学重绘', '\n'.join([*graph, '```']),
            *[f"{i}. {e['source']} → {e['target']} · {e['transfer']}（依据：{'、'.join(e['evidence_ids'])}）"
              for i, e in enumerate(edges, 1)]]


def render_reasoning(results):
    lines = ['# 论文讲解、核心论证与研究见解', '系统推论与研究建议不等于作者结论；核查通过也不表示独立复现。']
    for result in results:
        lines.append('## 主论文：' + result['paper_id'])
        lines.extend('> ' + w for w in result['warnings'])
        reviews = {(c['target_type'], c['index']): c for c in result['checks']}
        guide = result.get('reading_guide', [])
        passage_sources = []
        for i, passage in enumerate(guide):
            check = reviews.get(('reading_passage', i), {'verdict': 'unreviewed', 'reason': '尚未完成独立核查'})
            heading = '论文总结' if i == 0 else f'讲解 {i}'
            lines.extend([f"### {heading} · {passage['title']}", passage['explanation']])
            if check['verdict'] != 'supported':
                lines.append(f"> {check['verdict']} · {check['reason']}")
            for figure in result.get('reading_figures', []):
                if figure['passage_index'] == i:
                    caption = f"作者图 {figure['number']} · 第 {figure['page']} 页"
                    lines.append(caption + ' · ' + figure['caption_excerpt'])
                    if figure.get('url'):
                        lines.append(f"![{caption}]({figure['url']})")
            if i == 0 and result.get('flow_edges'):
                lines.extend(render_flow(result['flow_edges'], result.get('flow_scope') == 'partial'))
            passage_sources.extend([f"**{heading} · {passage['title']}**", '原文：' + '、'.join(passage['evidence_ids']),
                                    f"讲解核查（{check['verdict']}）：{check['reason']}"])
        if result.get('method_steps'):
            lines.append('## 算法如何工作')
        for i, step in enumerate(result.get('method_steps', [])):
            check = reviews.get(('method_step', i), {'verdict': 'unreviewed', 'reason': '尚未完成独立核查'})
            lines.extend([f"### {i+1}. {step['operation']}", '输入：' + step['input'], '输出与流向：' + step['output'],
                          '设计取舍：' + step['design_reason'], METHOD_BASIS[step['basis']] + '：' + step['justification'],
                          '失效条件或检验：' + step['failure_or_test'], '原文：' + '、'.join(step['evidence_ids']),
                          f"步骤核查（{check['verdict']}）：{check['reason']}"])
        for kind, values in [('argument', result['arguments']), ('external_support', result['external_support']), ('insight', result['insights'])]:
            for i, value in enumerate(values):
                check = reviews.get((kind, i), {'verdict': 'unreviewed', 'reason': '尚未完成独立核查'})
                labels = {'thesis':'主张','inference':'推理桥梁','conditions':'成立条件','alternative_explanation':'替代解释','falsification':'反驳条件','conclusion':'当前结论','statement':'见解','derivation':'推导','uncertainty':'不确定性','prediction':'预期观察','weakening_observation':'会削弱判断的观察','distinguishing_test':'验证实验','comparability':'可比条件','contribution':'证据贡献','relation':'佐证关系','epistemic_status':'认识状态'}
                lines.append(f"### {dict(argument='论证', external_support='外部佐证', insight='Insight')[kind]} {i+1}（{check['verdict']}）")
                if kind in {'argument', 'insight'}:
                    ids = [value['claim_id']] if kind == 'argument' else value['claim_ids']
                    for index in value.get('flow_edge_indices', []):
                        edge = result['flow_edges'][index]
                        lines.append(f"对应图中路径 {index + 1} · {edge['source']} → {edge['target']}")
                    titles = [p['title'] for p in guide if set(ids).intersection(p['claim_ids'])]
                    if titles:
                        lines.append('承接前文：' + '、'.join(titles))
                lines.extend(f'{label}：{value[key]}' for key, label in labels.items() if key in value)
                if kind == 'insight':
                    for gap in result.get('evidence_questions', []):
                        if gap['gap_id'] in value.get('gap_ids', []):
                            lines.append('对应证据缺口：' + gap['question'])
                if kind == 'external_support':
                    gap = next((q for q in result.get('evidence_questions', []) if q['gap_id'] == value.get('gap_id')), None)
                    if gap:
                        lines.append('对应问题：' + gap['question'])
                    lines.append('对判断的影响：' + check.get('judgment_update', '待核查'))
                if kind == 'argument':
                    lines.extend(render_citation(check))
                lines.append('证据：' + '、'.join(value['evidence_ids']))
                lines.append('核查：' + check['reason'])
        for gap in result.get('evidence_questions', []):
            if gap['local_review']['decision'] == 'external_needed' and not gap['candidate_ids']:
                lines.append('该缺口尚无可用外部候选，保留当前判断。')
            lines.extend(['### 待检验问题：' + gap['question'], '假设：' + gap['hypothesis'],
                          '预期观察：' + gap['prediction'], '会削弱判断的观察：' + gap['weakening_observation'],
                          '比较条件：' + gap['comparable_conditions'], '外查停止条件：' + gap['stopping_rule'],
                          '本文核查与外查决定：' + gap['local_review']['reason'],
                          '本文依据：' + ('、'.join(gap['local_review']['evidence_ids']) or '无定向核查引文')])
        if not result['insights']:
            lines.append('未形成可交付的 insight；不强行补齐。')
        if passage_sources:
            lines.extend(['### 讲解原文与核查记录', *passage_sources])
        if result['omitted_evidence_ids']:
            lines.append(f"本阶段未覆盖 {len(result['omitted_evidence_ids'])} 个证据片段，完整记录见结构化产物。")
    return '\n\n'.join(lines)


async def review_core_question(tasks, question, claim, check, card, evidence, policy, manifests):
    """Review a single versioned argument before processing secondary claims."""
    focused = card.model_copy(update={'claims': [claim], 'verifications': [check]})
    by_id = {e.id: e for e in evidence}
    preferred = dict.fromkeys([*claim.evidence_ids, *check.counterevidence.evidence_ids,
                               *check.experimental_support.evidence_ids, *check.evidence_ids, *by_id])
    evidence = [by_id[eid] for eid in preferred if eid in by_id]
    plan = await plan_arguments(tasks, question, focused, evidence)
    plan['arguments'] = plan['arguments'][:1]
    result = await synthesize(tasks, question, focused, plan, evidence, set(), policy,
                              source_manifests=manifests, argument_only=True)
    if result['status'] != 'reviewed':
        return None, result['warnings'] or ['未形成可独立审查的核心论证']
    citations = set(claim.evidence_ids + check.evidence_ids)
    for entry in result['arguments'] + result['checks']:
        citations.update(entry['evidence_ids'])
    return {'question': question, 'paper_id': card.paper_id, 'claim': claim.model_dump(),
            'verification': check.model_dump(), 'argument': result['arguments'][0],
            'review': result['checks'][0],
            'sources': [e.model_dump() for e in evidence if e.id in citations]}, []


def render_core_review(snapshot):
    arg, review = snapshot['argument'], snapshot['review']
    verdict = {'supported': '有证据支持', 'qualified': '需限定或补证', 'unsupported': '证据不支持'}[review['verdict']]
    lines = ['## ' + snapshot['question'], arg['thesis'], '原文主张核查：' + snapshot['verification']['reason'], '依据与推理：' + arg['inference'],
             '成立条件：' + arg['conditions'], '替代解释：' + arg['alternative_explanation'],
             '可反驳条件：' + arg['falsification'], '阶段结论：' + arg['conclusion'],
             f"独立审查（{verdict}）：{review['reason']}"]
    lines.extend(render_citation(review))
    for source in snapshot['sources']:
        location = f"第 {source['page_start']} 页 " if source['page_start'] else ''
        lines.extend([f"**{location}{source['section']}** `{source['id']}`", source['text']])
    return '\n\n'.join(lines)
