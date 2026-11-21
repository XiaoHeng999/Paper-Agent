"""Apply evidence and publication rules to an independently reviewed, versioned draft."""
from .argumentation import connected_flow


def reviewed_citation(argument, review, primary_check, evidence, paper_id):
    draft = argument.citation_draft
    if (not draft or review.citation_verdict == 'withhold' or review.verdict == 'unsupported'
            or not primary_check or primary_check.verdict not in {'verified', 'qualified'}
            or primary_check.fidelity.status not in {'supported', 'qualified'}):
        return None
    if (review.citation_verdict == 'comparison') != (draft.comparison is not None):
        return None
    span = evidence.get(draft.evidence_id)
    if (not span or span.paper_id != paper_id
            or span.source_level != 'full_text' or span.modality != 'text'
            or span.id not in argument.evidence_ids or span.id not in review.evidence_ids):
        return None
    # Allow PDF line wrapping while preserving word order and punctuation.
    if ' '.join(draft.excerpt.split()) not in ' '.join(span.text.split()):
        return None
    required = [draft.scope]
    if draft.comparison:
        required.extend([draft.comparison.baseline, draft.comparison.metric])
    if any(term not in draft.sentence for term in required):
        return None
    return {**draft.model_dump(), 'section': span.section, 'page_start': span.page_start,
            'page_end': span.page_end}


def validate_review(draft, review, supplied):
    groups = {'flow_edge': draft.flow_edges, 'reading_passage': draft.reading_guide, 'method_step': draft.method_steps, 'argument': draft.arguments, 'insight': draft.insights, 'external_support': draft.external_support}
    expected = {(kind, i) for kind, values in groups.items() for i in range(len(values))}
    seen = set()
    for check in review.checks:
        key = (check.target_type, check.index)
        if key not in expected or key in seen or not set(check.evidence_ids) <= supplied.keys():
            raise ValueError('论证核查引用越界或重复')
        if check.verdict == 'supported' and not check.evidence_ids:
            raise ValueError('通过论证核查必须有原文依据')
        seen.add(key)
    missing_edges = {key for key in expected - seen if key[0] == 'flow_edge'}
    if seen | missing_edges != expected:
        raise ValueError('论证核查遗漏对象，保留待审状态')
    return groups, missing_edges


def publish(result, draft, review, groups, missing_edges, claims, supplied, card, external_ids):
    for check in review.checks:
        target = groups[check.target_type][check.index]
        # A fulltext claim cannot be upgraded solely on an abstract or on unrelated papers.
        if check.verdict == 'supported' and (any(supplied[eid].source_level == 'abstract' for eid in [*target.evidence_ids, *check.evidence_ids])
                or not {supplied[eid].paper_id for eid in target.evidence_ids} <= {supplied[eid].paper_id for eid in check.evidence_ids}):
            check.verdict = 'qualified'
            check.reason += '；摘要或核查来源覆盖不足，不能视为完整验证'
        if (check.target_type in {'method_step', 'reading_passage', 'flow_edge'} and check.verdict == 'supported'
                and not set(check.evidence_ids).intersection(target.evidence_ids)):
            check.verdict = 'qualified'
            check.reason += '；核查未定位该段引用的原文'
        dependent_ids = ([target.claim_id] if check.target_type == 'argument' else target.claim_ids if check.target_type == 'insight' else [])
        if check.verdict == 'supported' and (check.target_type == 'argument' or (check.target_type == 'insight' and target.epistemic_status == 'supported_inference')):
            if any(not any(v.target_id == cid and v.target_version == claims[cid].version and v.verdict == 'verified' for v in card.verifications) for cid in dependent_ids):
                check.verdict = 'qualified'
                check.reason += '；依赖的主论文判断尚未完整通过核查'
    # A rejected external link cannot establish a conclusion that relies on that link.
    approved_links = set()
    for check in review.checks:
        if check.target_type == 'external_support' and check.verdict == 'supported':
            support = draft.external_support[check.index]
            approved_links.update((support.argument_index, support.gap_id, eid) for eid in support.evidence_ids)
    for check in review.checks:
        if check.verdict != 'supported' or check.target_type in {'external_support', 'method_step', 'reading_passage', 'flow_edge'}:
            continue
        target = groups[check.target_type][check.index]
        if check.target_type == 'argument':
            usable = {eid for i, _, eid in approved_links if i == check.index}
        elif target.epistemic_status == 'supported_inference':
            usable = {eid for i, gid, eid in approved_links if i in target.argument_indices and gid in target.gap_ids}
        else:
            continue
        if any(supplied[eid].paper_id in external_ids and eid not in usable for eid in target.evidence_ids):
            check.verdict = 'qualified'
            check.reason += '；依赖的外部佐证尚未通过独立核查'
    passage_checks = {c.index: c.verdict for c in review.checks if c.target_type == 'reading_passage'}
    for check in review.checks:
        if check.target_type not in {'argument', 'insight'} or check.verdict != 'supported' or not draft.reading_guide:
            continue
        target = groups[check.target_type][check.index]
        ids = [target.claim_id] if check.target_type == 'argument' else target.claim_ids
        for cid in ids:
            # The overview introduces the topic. Prefer the detailed premises when available.
            links = [i for i, p in enumerate(draft.reading_guide[1:], 1) if cid in p.claim_ids]
            if not links and cid in draft.reading_guide[0].claim_ids:
                links = [0]
            if not links or any(passage_checks[i] != 'supported' for i in links):
                check.verdict = 'qualified'
                check.reason += '；关联讲解的依据仍需限定或补证'
                break
    checked = []
    for check in review.checks:
        record = check.model_dump(exclude={'citation_verdict'})
        if check.target_type == 'argument':
            arg = draft.arguments[check.index]
            claim = claims[arg.claim_id]
            primary_check = next((v for v in card.verifications if v.target_id == arg.claim_id
                                  and v.target_version == claims[arg.claim_id].version), None)
            record['citation'] = (reviewed_citation(arg, check, primary_check, supplied, card.paper_id)
                                  if claim.attribution == 'author_report' else None)
        if check.target_type == 'external_support':
            support = draft.external_support[check.index]
            record['judgment_update'] = (
                {'supports': '增强判断', 'challenges': '削弱判断', 'different_conditions': '条件不同，保留判断',
                 'inconclusive': '证据不足，保留判断'}[support.relation]
                if check.verdict == 'supported' else '佐证尚未充分通过核查，保留判断')
        checked.append(record)
    approved_edges = [(c.index, draft.flow_edges[c.index]) for c in review.checks
                      if c.target_type == 'flow_edge' and c.verdict == 'supported']
    approved_edges.sort(key=lambda entry: entry[0])
    kept = {i for i, _ in approved_edges}
    omitted = [edge for i, edge in enumerate(draft.flow_edges) if i not in kept]
    # A disputed input can be left outside the drawing. Never remove an internal branch.
    inputs_only = all(sum(edge.source == other.source for other in draft.flow_edges) == 1
                      and all(edge.source != other.target for other in draft.flow_edges) for edge in omitted)
    if (draft.reading_guide and not missing_edges and inputs_only
            and connected_flow([edge for _, edge in approved_edges])):
        result['flow_edges'] = [dict(edge.model_dump(), review_index=i) for i, edge in approved_edges]
        if omitted:
            result['flow_scope'] = 'partial'
            result['warnings'].append('教学图从已核实的输入开始，省略待核实的输入路径：' +
                                      '、'.join(e.source + ' → ' + e.target for e in omitted))
        for target in [*result['arguments'], *result['insights']]:
            ids = set(target.get('claim_ids', [target.get('claim_id')]))
            cited = set(target['evidence_ids'])
            matches = []
            for index, (_, edge) in enumerate(approved_edges):
                shared = cited.intersection(edge.evidence_ids)
                if ids.intersection(edge.claim_ids) and shared:
                    matches.append((len(shared) / len(set(edge.evidence_ids)), len(shared), index))
            matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
            target['flow_edge_indices'] = [index for _, _, index in matches[:2]]
    elif draft.flow_edges:
        result['warnings'].append('教学图有箭头未通过核查，保留文字讲解。')
    result.update(checks=checked, status='reviewed')


def qualify_insights(result):
    supported_insights = {c['index'] for c in result['checks'] if c['target_type'] == 'insight' and c['verdict'] == 'supported'}
    for i, value in enumerate(result['insights']):
        if i not in supported_insights and value['epistemic_status'] == 'supported_inference':
            value['proposed_epistemic_status'] = 'supported_inference'
            value['epistemic_status'] = 'hypothesis'
