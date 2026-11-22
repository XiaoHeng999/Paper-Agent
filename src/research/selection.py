"""Auditable hybrid retrieval and bounded evidence-packet selection.

Similarity is used only to find entry points.  Deterministic relations and requirement coverage
then assemble a small packet; the packet remains candidate evidence until Reader/Verifier checks it.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .models import (
    EvidenceCandidateTrace, EvidencePacket, ReadingPlan, ReadingSubquery, version_of,
)


_RRF_OFFSET = 60
_POLICY = "rrf-relation-greedy-v1"

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
        ranked = await _hybrid_candidates(scope, subquery, tasks)
        packet = _select_packet(scope, subquery, ranked, per_packet_chars, max_nodes)
        repair_rounds = getattr(getattr(tasks, "options", None), "max_repair_rounds", 0)
        if packet.missing_requirements and packet.repair_queries and repair_rounds > 0:
            gap_hits = await scope.search(packet.repair_queries[0], tasks, 8)
            if gap_hits:
                ranked = _merge_gap_hits(ranked, gap_hits)
                packet = _select_packet(scope, subquery, ranked, per_packet_chars, max_nodes)
        packets.append(packet)
    return packets


def packet_priority(packets: list[EvidencePacket], scope) -> list:
    """Return packet evidence in plan order without trusting serialized copies."""

    ids = list(dict.fromkeys(evidence_id for packet in packets for evidence_id in packet.evidence_ids))
    return scope.read(ids) if ids else []


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
    for seed_id in seed_ids:
        seed = scope.spans[seed_id]
        related = [seed.previous_id, seed.next_id]
        if seed.parent_id:
            related.extend(span.id for span in scope.spans.values()
                if span.parent_id == seed.parent_id and span.id != seed.id)
        for related_id in list(dict.fromkeys(value for value in related if value))[:4]:
            if related_id not in scope.spans:
                continue
            scores[related_id] = max(scores[related_id], scores[seed_id] * 0.72)
            channels.setdefault(related_id, set()).add("relation")

    return [(scope.spans[evidence_id], score, sorted(channels[evidence_id]))
        for evidence_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:40]]


def _select_packet(scope, subquery, ranked, char_budget, max_nodes):
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
            retrieval_channels=channels, fused_score=score,
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


def _matched_requirements(span, required):
    text = f"{span.section} {span.text}".casefold()
    return [kind for kind in required if any(term in text for term in _REQUIREMENT_TERMS.get(kind, ()))]


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
    references = re.findall(r"(?:fig(?:ure)?|table|eq(?:uation)?)\.?\s*\d+[a-z]?", query, re.I)
    if not references:
        return []
    return [span for span in spans if any(reference.casefold() in span.text.casefold() for reference in references)]


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
