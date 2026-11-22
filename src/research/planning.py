"""Deterministic first-stage reading plans and query-aware evidence prioritization.

The planner deliberately does not assert paper conclusions.  It turns a raw user prompt into
bounded, typed questions that retrieval and reader calls can audit.  A later model-backed planner
can refine these questions without changing the data contract or the safe fallback.
"""
from __future__ import annotations

import re
from collections import Counter

from .models import (
    Claim, EvidencePacket, EvidenceRequirement, ReadingPlan, ReadingSubquery, SubqueryCoverage,
    version_of,
)


_GENERIC_PROMPTS = {
    "read", "阅读", "阅读论文", "阅读这篇论文", "分析", "分析论文", "分析这篇论文",
    "总结", "总结论文", "总结这篇论文", "看看这篇论文", "论文分析",
}

_LENS_SPECS = {
    "problem_and_scope": (
        "论文解决的具体问题、相对已有工作的贡献及适用范围是什么？",
        ("problem", "claim", "limitation"),
        "research problem contribution scope motivation conclusion limitation",
    ),
    "method_mechanism": (
        "方法沿同一输入依次执行哪些关键操作，各步输出如何被下一步使用？",
        ("method_step", "equation", "figure", "implementation"),
        "method architecture algorithm equation input output implementation",
    ),
    "assumptions": (
        "方法成立依赖哪些理论、数据、环境或实现假设，假设失效时有什么边界？",
        ("assumption", "equation", "limitation"),
        "assumption approximation condition boundary failure limitation",
    ),
    "experimental_design": (
        "实验使用哪些数据、基线、指标、预算和硬件，比较条件是否一致？",
        ("dataset", "baseline", "metric", "budget", "hardware", "result", "table"),
        "experimental setup dataset baseline metric budget hardware result table",
    ),
    "causal_validation": (
        "实验和消融是否支持作者提出的机制，是否存在未排除的替代解释？",
        ("result", "ablation", "negative_result", "figure", "table"),
        "ablation mechanism causal validation alternative explanation negative result",
    ),
    "limitations_reproducibility": (
        "论文报告了哪些局限、失败场景和复现细节，还有哪些关键条件未报告？",
        ("limitation", "negative_result", "implementation", "budget"),
        "limitation failure reproducibility implementation detail negative result",
    ),
}

_KIND_TO_LENS = {
    "question": "problem_and_scope",
    "method": "method_mechanism",
    "premise": "assumptions",
    "result": "experimental_design",
    "limitation": "limitations_reproducibility",
}


def build_reading_plan(user_prompt: str, paper_id: str, title: str, abstract: str = "") -> ReadingPlan:
    """Build a bounded plan without inventing facts from the target paper."""

    prompt = " ".join(str(user_prompt or "").split())
    questions = _user_questions(prompt)
    mode = _reading_mode(prompt, questions)
    paper_type = _paper_type(title, abstract)
    subqueries = []
    for question in questions:
        lens = _infer_lens(question)
        subqueries.append(_subquery(paper_id, "user", lens, question, 5))

    # Default lenses provide coverage; user questions keep the highest priority and are never
    # overwritten.  Eight is the hard schema limit, so four user questions leave four defaults.
    for lens in _lenses_for_type(paper_type):
        if len(subqueries) >= 8:
            break
        if any(item.lens == lens and item.question == _LENS_SPECS[lens][0] for item in subqueries):
            continue
        question = _LENS_SPECS[lens][0]
        subqueries.append(_subquery(paper_id, "default_lens", lens, question, 3))

    warnings = []
    if not prompt or mode == "open_ended":
        warnings.append("用户未提供可直接检索的具体问题，已使用论文阅读默认分析维度")
    return ReadingPlan(paper_id=paper_id, mode=mode, paper_type=paper_type,
        user_prompt=prompt, user_questions=questions, subqueries=subqueries, warnings=warnings)


def assign_subqueries(kind: str, statement: str, plan: ReadingPlan, requested: list[str]) -> list[str]:
    """Validate model-provided IDs or conservatively bind a claim to one planned lens."""

    allowed = {item.id for item in plan.subqueries}
    if requested:
        if not set(requested) <= allowed:
            raise ValueError("Reader 引用了当前阅读计划之外的子问题")
        return list(dict.fromkeys(requested))
    preferred = _KIND_TO_LENS.get(kind, "problem_and_scope")
    candidates = [item for item in plan.subqueries if item.lens == preferred]
    user_matches = [item for item in plan.subqueries if item.source == "user"
        and _overlap(statement, item.question) > 0]
    selected = user_matches[:1] or candidates[:1] or plan.subqueries[:1]
    return [item.id for item in selected]


async def prioritize_for_plan(scope, plan: ReadingPlan, tasks, per_query: int = 3):
    """Round-robin semantic and lexical hits so one broad question cannot occupy every slot."""

    per_subquery = []
    for item in sorted(plan.subqueries, key=lambda value: -value.priority):
        query = " ".join([item.question, *item.retrieval_queries])
        semantic = await scope.search(query, tasks, per_query)
        lexical = _lexical_hits(scope.spans.values(), query, per_query)
        unique = []
        seen = set()
        for span in [*semantic, *lexical]:
            if span.id not in seen:
                seen.add(span.id)
                unique.append(span)
        per_subquery.append(unique)
    ordered = []
    for offset in range(per_query * 2):
        for hits in per_subquery:
            if offset < len(hits) and hits[offset] not in ordered:
                ordered.append(hits[offset])
    return ordered


def coverage_for_plan(plan: ReadingPlan, claims: list[Claim], packets: list[EvidencePacket] | None = None) -> list[SubqueryCoverage]:
    """Combine verified extraction linkage with packet-level candidate coverage."""

    packet_by_subquery = {packet.subquery_id: packet for packet in packets or []}
    result = []
    for item in plan.subqueries:
        linked = [claim for claim in claims if item.id in claim.subquery_ids]
        packet = packet_by_subquery.get(item.id)
        covered = packet.covered_requirements if packet else []
        missing = packet.missing_requirements if packet else [requirement.kind for requirement in item.evidence_requirements]
        # Packet matching identifies candidate evidence types, while a linked Reader claim is
        # still required before a question can be reported as covered.
        status = "unanswered" if not linked else "covered" if packet and not missing else "partial"
        result.append(SubqueryCoverage(subquery_id=item.id,
            status=status,
            claim_ids=[claim.id for claim in linked],
            covered_requirements=covered, missing_requirements=missing))
    return result


def _subquery(paper_id, source, lens, question, priority):
    _, requirements, retrieval = _LENS_SPECS[lens]
    if re.search(r"fp(?:16|32|64|8)|bf16|int[48]|量化|数值精度|低精度|混合精度|基线精度|mixed precision|lower precision|numerical precision|baseline precision", question, re.I):
        requirements = (*requirements, "precision")
        if re.search(r"基线|对比|加速|优势|速度|baseline|speedup|faster|latency", question, re.I):
            requirements = (*requirements, "baseline_precision")
    value = [paper_id, source, lens, question]
    return ReadingSubquery(id=f"rq:{version_of(value)}", source=source, lens=lens,
        question=question, retrieval_queries=[question, retrieval], priority=priority,
        evidence_requirements=[EvidenceRequirement(kind=kind) for kind in requirements])


def _reading_mode(prompt, questions):
    normalized = re.sub(r"[\s，。！？!?；;：:]", "", prompt).casefold()
    if not normalized or normalized in _GENERIC_PROMPTS:
        return "open_ended"
    if questions and ("?" in prompt or "？" in prompt or any(term in prompt.casefold()
            for term in ("为什么", "如何", "是否", "what", "why", "how", "whether"))):
        return "targeted"
    return "guided"


def _user_questions(prompt):
    if not prompt:
        return []
    normalized = re.sub(r"[\s，。！？!?；;：:]", "", prompt).casefold()
    if normalized in _GENERIC_PROMPTS:
        return []
    parts = [part.strip(" ，。！？!?；;\n") for part in re.split(r"[？?；;\n]+", prompt)]
    return list(dict.fromkeys(part for part in parts if part))[:4]


def _paper_type(title, abstract):
    text = f"{title} {abstract}".casefold()
    rules = (
        ("survey", ("survey", "review", "综述")),
        ("dataset_benchmark", ("dataset", "benchmark", "corpus", "数据集", "基准")),
        ("theory", ("theorem", "proof", "理论", "定理", "证明")),
        ("system", ("system", "serving", "runtime", "compiler", "系统", "框架")),
        ("empirical_study", ("empirical study", "observational", "randomized", "实证", "调查")),
    )
    return next((kind for kind, terms in rules if any(term in text for term in terms)), "algorithm")


def _lenses_for_type(paper_type):
    if paper_type == "theory":
        return ("problem_and_scope", "method_mechanism", "assumptions",
            "causal_validation", "limitations_reproducibility")
    if paper_type == "survey":
        return ("problem_and_scope", "method_mechanism", "assumptions",
            "experimental_design", "limitations_reproducibility")
    return tuple(_LENS_SPECS)


def _infer_lens(question):
    text = question.casefold()
    keyword_lenses = (
        ("experimental_design", ("实验", "基线", "数据集", "指标", "硬件", "速度", "性能", "experiment", "baseline", "latency")),
        ("causal_validation", ("消融", "归因", "有效", "ablation", "why", "为什么")),
        ("limitations_reproducibility", ("局限", "失败", "复现", "可靠", "limitation", "failure", "reproduc")),
        ("assumptions", ("假设", "前提", "条件", "assumption", "premise")),
        ("method_mechanism", ("方法", "机制", "架构", "如何", "method", "mechanism", "how")),
    )
    return next((lens for lens, terms in keyword_lenses if any(term in text for term in terms)), "problem_and_scope")


def _terms(text):
    lowered = text.casefold()
    latin = re.findall(r"[a-z0-9_\-]{3,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    bigrams = [value[i:i + 2] for value in chinese for i in range(len(value) - 1)]
    return list(dict.fromkeys([*latin, *bigrams]))


def _overlap(left, right):
    return len(set(_terms(left)) & set(_terms(right)))


def _lexical_hits(spans, query, limit):
    wanted = Counter(_terms(query))
    ranked = []
    for position, span in enumerate(spans):
        haystack = f"{span.section} {span.text}".casefold()
        score = sum(weight for term, weight in wanted.items() if term in haystack)
        if score:
            ranked.append((-score, position, span))
    ranked.sort(key=lambda value: (value[0], value[1]))
    return [span for _, _, span in ranked[:limit]]
