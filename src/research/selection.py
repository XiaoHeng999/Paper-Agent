"""Auditable hybrid retrieval and bounded evidence-packet selection.

Similarity is used only to find entry points.  Deterministic relations and requirement coverage
then assemble a small packet; the packet remains candidate evidence until Reader/Verifier checks it.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .multimodal import find_object_labels
from .models import (
    EvidenceCandidateTrace, EvidencePacket, IncrementJudgmentBatch, NodeJudgmentBatch,
    ReadingPlan, ReadingSubquery, RetrievalHop, version_of,
)


_RRF_OFFSET = 60
_POLICY = "rrf-relation-greedy-v2"

_REQUIREMENT_TERMS = {
    "problem": ("problem", "motivation", "challenge", "问题", "动机", "挑战"),
    "claim": ("contribution", "we show", "we find", "结论", "贡献", "发现"),
    "method_step": ("method", "algorithm", "module", "architecture", "方法", "算法", "模块", "架构"),
    "equation": ("equation", "theorem", "formula", "公式", "定理", "证明"),
    "assumption": ("assume", "assumption", "condition", "假设", "前提", "条件"),
    "dataset": ("dataset", "corpus", "benchmark", "数据集", "语料", "基准"),
    "baseline": ("baseline", "compared with", "state-of-the-art", "基线", "对比方法"),
    "metric": ("metric", "accuracy", "precision", "recall", "f1", "bleu", "指标", "准确率"),
    "budget": ("budget", "parameter", "flop", "cost", "预算", "参数量", "计算量", "成本"),
    "precision": ("fp16", "bf16", "fp32", "fp8", "fp64", "int8", "int4", "mixed precision", "混合精度"),
    "hardware": ("gpu", "cpu", "tpu", "hardware", "a100", "v100", "硬件", "显卡"),
    "result": ("result", "performance", "outperform", "improve", "结果", "性能", "提升"),
    "ablation": ("ablation", "remove", "variant", "消融", "移除", "变体"),
    "negative_result": ("failure", "degrade", "worse", "negative result", "失败", "下降", "负面结果"),
    "limitation": ("limitation", "weakness", "future work", "局限", "不足", "未来工作"),
    "implementation": ("implementation", "code", "hyperparameter", "appendix", "实现", "代码", "超参数", "附录"),
    "figure": ("figure", "fig.", "图"),
    "table": ("table", "tab.", "表"),
}

_LENS_SECTION_TERMS = {
    "problem_and_scope": ("abstract", "introduction", "conclusion", "摘要", "引言", "结论"),
    "method_mechanism": ("method", "approach", "algorithm", "architecture", "方法", "算法", "架构"),
    "assumptions": ("background", "theory", "analysis", "assumption", "背景", "理论", "假设"),
    "experimental_design": ("experiment", "evaluation", "setup", "result", "实验", "评估", "结果"),
    "causal_validation": ("ablation", "analysis", "discussion", "消融", "分析", "讨论"),
    "limitations_reproducibility": ("limitation", "discussion", "appendix", "implementation", "局限", "附录", "实现"),
}


async def build_evidence_packets(scope, plan: ReadingPlan, tasks, *, per_packet_chars: int = 4200,
        max_nodes: int = 5) -> list[EvidencePacket]:
    """Build one bounded packet per subquery from scoped authoritative evidence."""

    packets = []
    for subquery in sorted(plan.subqueries, key=lambda item: -item.priority):
        ranked, paths = await _hybrid_candidates(scope, subquery, tasks)
        packet = _select_packet(scope, subquery, ranked, per_packet_chars, max_nodes, paths)
        expanded = _expand_reference_gaps(scope, packet, ranked, paths)
        if expanded:
            ranked = expanded
            packet = _select_packet(scope, subquery, ranked, per_packet_chars, max_nodes, paths)
        repair_rounds = getattr(getattr(tasks, "options", None), "max_repair_rounds", 0)
        if packet.missing_requirements and packet.repair_queries and repair_rounds > 0:
            gap_hits = await scope.search(packet.repair_queries[0], tasks, 8)
            if gap_hits:
                ranked = _merge_gap_hits(ranked, gap_hits)
                packet = _select_packet(scope, subquery, ranked, per_packet_chars, max_nodes, paths)
        packets.append(packet)
    return packets


def packet_priority(packets: list[EvidencePacket], scope) -> list:
    """Return packet evidence in plan order without trusting serialized copies."""

    ids = list(dict.fromkeys(evidence_id for packet in packets for evidence_id in packet.evidence_ids))
    return scope.read(ids) if ids else []


async def refine_packets_with_model(tasks, scope, packets: list[EvidencePacket]) -> list[EvidencePacket]:
    """Optionally run two bounded listwise judgments; never replace factual verification."""

    if not getattr(tasks.options, "evidence_reranking_enabled", False):
        return packets
    candidates = []
    for packet in packets:
        trace = {item.evidence_id: item for item in packet.candidate_trace}
        for evidence_id in packet.evidence_ids:
            span = scope.spans.get(evidence_id)
            if span is None or evidence_id not in trace:
                continue
            candidates.append({"subquery_id": packet.subquery_id, "question": packet.question,
                "evidence_id": evidence_id, "text": span.text[:3000], "section": span.section,
                "view_type": span.view_type, "visual_object_ids": span.visual_object_ids,
                "retrieval_channels": trace[evidence_id].retrieval_channels,
                "candidate_requirements": trace[evidence_id].matched_requirements})
    candidates = candidates[:40]
    if not candidates:
        return packets
    allowed = {(item["subquery_id"], item["evidence_id"]) for item in candidates}
    try:
        nodes = await tasks.ask("node_verifier",
            "逐项判断候选能否直接帮助回答对应子问题。0=无关，1=仅背景，2=部分相关，3=直接支持，4=关键直接证据。"
            "只判断相关性和可能覆盖的证据需求，不判断论文主张真假；不得新增候选、需求或事实。", {
                "candidates": candidates,
            }, NodeJudgmentBatch)
        if any((item.subquery_id, item.evidence_id) not in allowed for item in nodes.judgments):
            raise ValueError("Node Verifier 返回了未提供的候选")
        anchors = {}
        node_lookup = {(item.subquery_id, item.evidence_id): item for item in nodes.judgments}
        for packet in packets:
            available = [evidence_id for evidence_id in packet.evidence_ids
                if (packet.subquery_id, evidence_id) in node_lookup]
            anchors[packet.subquery_id] = max(available,
                key=lambda evidence_id: node_lookup[(packet.subquery_id, evidence_id)].direct_support,
                default=packet.anchor_id)
        increments_payload = []
        for item in candidates:
            anchor_id = anchors.get(item["subquery_id"], "")
            if anchor_id and item["evidence_id"] != anchor_id:
                increments_payload.append({**item, "anchor_id": anchor_id,
                    "anchor_text": scope.spans[anchor_id].text[:3000]})
        increments = await tasks.ask("increment_verifier",
            "相对同一子问题的 Anchor，逐项判断候选的增量角色。conflict 和 clarification 即使相似度低也应明确标出；"
            "redundant 表示没有新增信息，irrelevant 表示不能帮助当前问题。只返回已提供的组合。", {
                "candidates": increments_payload[:40],
            }, IncrementJudgmentBatch) if increments_payload else IncrementJudgmentBatch()
        allowed_increments = {(item["subquery_id"], item["anchor_id"], item["evidence_id"])
            for item in increments_payload[:40]}
        if any((item.subquery_id, item.anchor_id, item.evidence_id) not in allowed_increments
                for item in increments.judgments):
            raise ValueError("Increment Verifier 返回了未提供的候选关系")
        return _apply_model_judgments(packets, nodes, increments, anchors)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        paper_id = next(iter(scope.allowed), "unknown")
        tasks.reranking_warnings.setdefault(paper_id, []).append(
            f"证据候选模型重排未完成，保留确定性 Packet：{exc}")
        return packets


async def _hybrid_candidates(scope, subquery: ReadingSubquery, tasks):
    query = " ".join([subquery.question, *subquery.retrieval_queries])
    channels = {}
    dense = await scope.search(query, tasks, 8)
    lexical = _rank_lexical(scope.spans.values(), query, 12)
    section = _rank_sections(scope.spans.values(), subquery, 12)
    exact = _rank_exact_references(scope.spans.values(), query)
    requirement = _rank_requirements(scope.spans.values(), subquery, 16)
    rankings = {"dense": dense, "lexical": lexical, "section": section,
        "exact_reference": exact, "requirement": requirement}
    scores = defaultdict(float)
    for channel, spans in rankings.items():
        for rank, span in enumerate(spans, start=1):
            scores[span.id] += 1 / (_RRF_OFFSET + rank)
            channels.setdefault(span.id, set()).add(channel)

    if not scores:
        for rank, span in enumerate(list(scope.spans.values())[:3], start=1):
            scores[span.id] = 1 / (_RRF_OFFSET + rank)
            channels[span.id] = {"fallback"}

    seed_ids = [item[0] for item in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:4]]
    paths = {}
    seed_scores = dict(scores)
    for seed_id in seed_ids:
        seed = scope.spans[seed_id]
        related = []
        if seed.visual_object_ids:
            related.extend(span.id for span in scope.spans.values() if span.id != seed.id
                and set(seed.visual_object_ids) & set(span.visual_object_ids))
        related.extend([seed.previous_id, seed.next_id])
        if seed.parent_id:
            related.extend(span.id for span in scope.spans.values()
                if span.parent_id == seed.parent_id and span.id != seed.id)
        for related_id in list(dict.fromkeys(value for value in related if value))[:4]:
            if related_id not in scope.spans:
                continue
            target = scope.spans[related_id]
            if target.paper_id != seed.paper_id or related_id == seed_id:
                continue
            shared = sorted(set(seed.visual_object_ids) & set(target.visual_object_ids))
            relation = "same_object" if shared else _relation(seed, target)
            provenance = ", ".join(shared) if shared else (
                seed.parent_id if relation == "same_parent" else "previous_id/next_id")
            paths.setdefault(related_id, [RetrievalHop(source_id=seed.id, target_id=related_id,
                relation=relation, confidence=1, provenance=provenance)])
            scores[related_id] = max(scores[related_id], seed_scores[seed_id] * 0.72)
            channels.setdefault(related_id, set()).add("relation")

    return [(scope.spans[evidence_id], score, sorted(channels[evidence_id]))
        for evidence_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:40]], paths


def _expand_reference_gaps(scope, packet, ranked, paths):
    gaps = set(packet.missing_requirements) & {
        "hardware", "precision", "baseline_precision", "budget", "implementation", "ablation", "limitation"}
    if not gaps:
        return []
    entries = {span.id: (span, score, channels) for span, score, channels in ranked}
    hits = {}
    for bridge_id in packet.evidence_ids:
        if bridge_id not in paths:
            continue
        bridge = scope.spans[bridge_id]
        if bridge.crop_version:
            continue
        labels = set(find_object_labels(bridge.text))
        if not labels:
            continue
        for span in scope.spans.values():
            if span.crop_version or span.paper_id != bridge.paper_id or span.id in packet.evidence_ids or span.id in hits:
                continue
            if span.id == paths[bridge_id][0].source_id or not _matched_requirements(span, gaps):
                continue
            shared = labels.intersection(find_object_labels(span.text))
            if len(shared) == 1:
                label, _ = next(iter(shared))
                hits[span.id] = (span, bridge_id, label)
                if len(hits) == 4:
                    break
        if len(hits) == 4:
            break
    for span, bridge_id, label in hits.values():
        paths[span.id] = [*paths[bridge_id], RetrievalHop(source_id=bridge_id,
            target_id=span.id, relation="explicit_reference", confidence=1,
            provenance=f"shared explicit reference {label}")]
        _, score, channels = entries.get(span.id, (span, 0, []))
        entries[span.id] = (span, max(score, entries[bridge_id][1] * 0.72),
            sorted(set(channels) | {"relation"}))
    return sorted(entries.values(), key=lambda item: (-item[1], item[0].id))[:40] if hits else []


def _select_packet(scope, subquery, ranked, char_budget, max_nodes, paths):
    required = [item.kind for item in subquery.evidence_requirements if item.required]
    matches = {span.id: _matched_requirements(span, required) for span, _, _ in ranked}
    anchor_entry = max(ranked, key=lambda item: (item[1] * 100 +
        (1.0 if "dense" in item[2] else 0.8 if "exact_reference" in item[2] else 0), item[0].id)) if ranked else None
    selected = [anchor_entry] if anchor_entry else []
    covered = set(matches[anchor_entry[0].id]) if anchor_entry else set()
    total = len(anchor_entry[0].text) if anchor_entry else 0
    remaining = [item for item in ranked if item != anchor_entry]
    while remaining and len(selected) < max_nodes:
        best = None
        best_value = float("-inf")
        for span, fused_score, channels in remaining:
            if total + len(span.text) > char_budget and selected:
                continue
            new_requirements = set(matches[span.id]) - covered
            limitation_bonus = 1.5 if {"limitation", "negative_result"} & set(matches[span.id]) else 0
            redundancy = max((_text_overlap(span.text, item[0].text) for item in selected), default=0)
            value = fused_score * 100 + len(new_requirements) * 3 + limitation_bonus - redundancy * 2
            if value > best_value:
                best, best_value = (span, fused_score, channels), value
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        total += len(best[0].text)
        covered.update(matches[best[0].id])
        if set(required) <= covered:
            break

    anchor = selected[0][0] if selected else None
    trace = []
    for span, score, channels in ranked:
        relation = _relation(anchor, span)
        trace.append(EvidenceCandidateTrace(evidence_id=span.id,
            retrieval_path=paths.get(span.id, []), retrieval_channels=channels, fused_score=score,
            matched_requirements=matches[span.id], relation=relation,
            anchor_id=anchor.id if anchor and span.id != anchor.id else ""))
    selected_ids = [item[0].id for item in selected]
    covered_ordered = [kind for kind in required if kind in covered]
    missing = [kind for kind in required if kind not in covered]
    repair_queries = _repair_queries(subquery, missing)
    payload = {"subquery_id": subquery.id, "question": subquery.question,
        "evidence_ids": selected_ids, "anchor_id": anchor.id if anchor else "",
        "covered_requirements": covered_ordered, "missing_requirements": missing,
        "repair_queries": repair_queries, "candidate_trace": [item.model_dump() for item in trace],
        "selection_policy": _POLICY}
    version = version_of(payload)
    return EvidencePacket(id=f"packet:{version}", version=version, **payload)


def _merge_gap_hits(ranked, hits):
    values = {span.id: [span, score, set(channels)] for span, score, channels in ranked}
    for rank, span in enumerate(hits, start=1):
        current = values.setdefault(span.id, [span, 0.0, set()])
        current[1] += 1 / (_RRF_OFFSET + rank)
        current[2].add("gap_dense")
    return [(span, score, sorted(channels)) for span, score, channels in
        sorted(values.values(), key=lambda item: (-item[1], item[0].id))[:40]]


def _apply_model_judgments(packets, nodes, increments, anchors):
    node_lookup = {(item.subquery_id, item.evidence_id): item for item in nodes.judgments}
    increment_lookup = {(item.subquery_id, item.evidence_id): item for item in increments.judgments}
    refined = []
    role_order = {"conflict": 0, "clarification": 1, "new_fact": 2,
        "corroboration": 3, None: 4, "redundant": 5, "irrelevant": 6}
    for packet in packets:
        total_requirements = list(dict.fromkeys([*packet.covered_requirements, *packet.missing_requirements]))
        trace = []
        for item in packet.candidate_trace:
            node = node_lookup.get((packet.subquery_id, item.evidence_id))
            increment = increment_lookup.get((packet.subquery_id, item.evidence_id))
            trace.append(item.model_copy(update={
                "direct_support": node.direct_support if node else None,
                "verifier_requirements": [value for value in (node.supported_requirements if node else [])
                    if value in total_requirements],
                "incremental_role": increment.role if increment else None,
                "anchor_id": anchors.get(packet.subquery_id, packet.anchor_id)
                    if item.evidence_id != anchors.get(packet.subquery_id, packet.anchor_id) else "",
                "relation": "anchor" if item.evidence_id == anchors.get(packet.subquery_id, packet.anchor_id)
                    else item.relation,
            }))
        selected_trace = [item for item in trace if item.evidence_id in packet.evidence_ids]
        anchor_id = anchors.get(packet.subquery_id, packet.anchor_id)
        keep = [item for item in selected_trace if item.evidence_id == anchor_id or
            item.incremental_role not in {"irrelevant", "redundant"} or
            {"limitation", "negative_result"} & set(item.matched_requirements)]
        keep.sort(key=lambda item: (item.evidence_id != anchor_id,
            role_order[item.incremental_role], -(item.direct_support or 0), -item.fused_score))
        evidence_ids = [item.evidence_id for item in keep]
        covered = list(dict.fromkeys(value for item in keep
            for value in [*item.matched_requirements, *item.verifier_requirements]
            if value in total_requirements))
        missing = [value for value in total_requirements if value not in covered]
        payload = packet.model_dump(exclude={"id", "version"})
        payload.update(evidence_ids=evidence_ids, anchor_id=anchor_id,
            covered_requirements=covered, missing_requirements=missing,
            repair_queries=packet.repair_queries if missing else [],
            candidate_trace=[item.model_dump() for item in trace],
            selection_policy=packet.selection_policy + "+listwise-node-increment-v1")
        version = version_of(payload)
        refined.append(EvidencePacket(id=f"packet:{version}", version=version, **payload))
    return refined


def _matched_requirements(span, required):
    text = f"{span.section} {span.text}".casefold()
    matched = [kind for kind in required if any(term in text for term in _REQUIREMENT_TERMS.get(kind, ()))]
    cell = span.table_cell
    precision_row = cell and re.fullmatch(r"baseline|基线", cell.row_header, re.I) and re.fullmatch(
        r"precision|dtype|数值精度|精度", cell.column_header, re.I) and re.fullmatch(
        r"(?:fp(?:8|16|32|64)|bf16|int[48])", cell.value, re.I)
    if "baseline_precision" in required and (precision_row or re.search(
            r"(?:\bbaseline(?: precision)?|基线(?:精度)?)\s*(?:uses?|used|runs? (?:in|at)|was|is|采用|使用|为)?"
            r"\s*[:：|=]?\s*(?:fp(?:8|16|32|64)|bf16|int[48])(?![a-z0-9])", span.text, re.I)):
        matched.append("baseline_precision")
    return matched


def _rank_lexical(spans, query, limit):
    terms = _terms(query)
    ranked = []
    for position, span in enumerate(spans):
        text = f"{span.section} {span.text}".casefold()
        score = sum(1 for term in terms if term in text)
        if score:
            ranked.append((-score, position, span))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:limit]]


def _rank_sections(spans, subquery, limit):
    terms = _LENS_SECTION_TERMS[subquery.lens]
    matched = [span for span in spans if any(term in span.section.casefold() for term in terms)]
    return matched[:limit]


def _rank_exact_references(spans, query):
    references = {label for label, _ in find_object_labels(query)}
    if not references:
        return []
    matches = []
    for span in spans:
        if span.crop_version:
            labels = {oid.removeprefix(f"{span.paper_id}:visual-object:") for oid in span.visual_object_ids}
        else:
            labels = {label for label, _ in find_object_labels(span.text)}
        if references & labels:
            matches.append(span)
    return matches


def _rank_requirements(spans, subquery, limit):
    required = [item.kind for item in subquery.evidence_requirements if item.required]
    ranked = []
    for position, span in enumerate(spans):
        count = len(_matched_requirements(span, required))
        if count:
            ranked.append((-count, position, span))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:limit]]


def _relation(anchor, span):
    if anchor is None:
        return "independent"
    if span.id == anchor.id:
        return "anchor"
    if span.id in {anchor.previous_id, anchor.next_id} or anchor.id in {span.previous_id, span.next_id}:
        return "adjacent"
    if anchor.parent_id and anchor.parent_id == span.parent_id:
        return "same_parent"
    if set(anchor.visual_object_ids) & set(span.visual_object_ids):
        return "same_object"
    return "independent"


def _repair_queries(subquery, missing):
    if not missing:
        return []
    labels = " ".join(missing)
    return [f"{subquery.question} {labels}", f"{labels} appendix setup limitation"]


def _terms(text):
    lowered = text.casefold()
    latin = re.findall(r"[a-z0-9_\-]{3,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    return list(dict.fromkeys([*latin, *(value[i:i + 2] for value in chinese for i in range(len(value) - 1))]))


def _text_overlap(left, right):
    left_terms, right_terms = set(_terms(left)), set(_terms(right))
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 0
