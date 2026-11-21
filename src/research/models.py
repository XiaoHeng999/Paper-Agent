from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def version_of(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ResearchOptions:
    output_mode: str = "paper_report"
    cross_paper_enabled: bool = False
    max_calls: int = 80
    max_repair_rounds: int = 1
    timeout_seconds: int = 600
    context_chars: int = 24000
    allow_external_search: bool = False
    external_paper_limit: int = 2
    reasoning_enabled: bool = True
    visual_verification: bool = True
    visual_page_limit: int = 6
    scoring_goal: str = ""
    score_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_constraints(cls, constraints: dict) -> ResearchOptions:
        if not isinstance(constraints, dict):
            raise ValueError("constraints 必须是对象")
        mode = constraints.get("output_mode", "paper_report")
        if not isinstance(mode, str) or mode not in {"paper_cards", "paper_report", "comparison", "review"}:
            raise ValueError("output_mode 必须是 paper_cards、paper_report、comparison 或 review")
        enabled = constraints.get("cross_paper_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("cross_paper_enabled 必须是布尔值")
        papers = constraints.get("papers")
        if papers is not None:
            if not isinstance(papers, list) or not 1 <= len(papers) <= 200:
                raise ValueError("papers 必须包含 1 到 200 篇论文")
            identities, input_ids = set(), set()
            for paper in papers:
                if not isinstance(paper, dict) or not isinstance(paper.get("id"), str) or not paper["id"].strip() or not isinstance(paper.get("title"), str) or not paper["title"].strip():
                    raise ValueError("每篇指定论文必须包含非空字符串 id 和 title")
                identity = str(paper.get("paperId") or paper.get("doi") or paper["id"]).strip()
                if identity in identities or paper["id"] in input_ids:
                    raise ValueError("指定论文包含重复的论文标识")
                identities.add(identity)
                input_ids.add(paper["id"])
        external = constraints.get("allow_external_search", False)
        if type(external) is not bool:
            raise ValueError("allow_external_search 必须是布尔值")
        reasoning = constraints.get("research_reasoning_enabled", True)
        if type(reasoning) is not bool:
            raise ValueError("research_reasoning_enabled 必须是布尔值")
        external_limit = constraints.get("external_paper_limit", 2)
        if type(external_limit) is not int or not 1 <= external_limit <= 3:
            raise ValueError("external_paper_limit 必须是 1 到 3 的整数")
        goal = constraints.get("scoring_goal", "")
        weights = constraints.get("score_weights", {})
        if not isinstance(goal, str) or not isinstance(weights, dict):
            raise ValueError("评分目标必须是文字，权重必须是对象")
        import math
        for key, value in weights.items():
            if key not in DIMENSIONS or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("评分权重仅支持已定义维度，且必须为有限正数")
        if weights and not goal.strip():
            raise ValueError("计算总分前必须明确 scoring_goal")
        visual = constraints.get("visual_verification", True)
        page_limit = constraints.get("visual_page_limit", 6)
        if type(visual) is not bool or type(page_limit) is not int or not 1 <= page_limit <= 20:
            raise ValueError("visual_verification 必须是布尔值，visual_page_limit 必须为 1–20")
        limits = {}
        for key, default, low, high in (
            ("max_calls", 80, 1, 500), ("max_repair_rounds", 1, 0, 2),
            ("timeout_seconds", 600, 1, 3600), ("context_chars", 24000, 4000, 60000),
        ):
            value = constraints.get("research_" + key, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"research_{key} 必须是 {low} 到 {high} 的整数")
            limits[key] = value
        return cls(reasoning_enabled=reasoning, visual_verification=visual, visual_page_limit=page_limit, output_mode=mode, cross_paper_enabled=enabled, allow_external_search=external, external_paper_limit=external_limit, scoring_goal=goal.strip(), score_weights=weights, **limits)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


DIMENSIONS = {
    "evidence_sufficiency": "证据充分性", "comparison_fairness": "比较公平性",
    "method_increment": "方法增量", "reproducibility": "复现透明度", "scope_clarity": "适用范围清晰度",
}
AXES = ("question", "task", "method", "dataset", "baseline", "metric", "budget", "scope")


class JudgmentCheck(Record):
    status: Literal["supported", "qualified", "contradicted", "unknown"] = "unknown"
    reason: str = "未完成独立判断"
    evidence_ids: list[str] = Field(default_factory=list)


class DimensionAssessment(Record):
    dimension: Literal["evidence_sufficiency", "comparison_fairness", "method_increment", "reproducibility", "scope_clarity"]
    score: int | None = Field(default=None, ge=0, le=4, strict=True)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    scope: Literal["paper_internal", "compared_papers"] = "paper_internal"

    @model_validator(mode="after")
    def require_evidence(self):
        if self.score is not None and not self.evidence_ids:
            raise ValueError("有分值的维度必须引用证据，缺失信息应为 unknown")
        return self


class ComparisonAxis(Record):
    axis: Literal["question", "task", "method", "dataset", "baseline", "metric", "budget", "scope"]
    left: str = "unknown"
    right: str = "unknown"
    status: Literal["matched", "different", "unknown"] = "unknown"
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceSpan(Record):
    id: str
    paper_id: str
    version: str
    text: str
    source_level: Literal["full_text", "abstract"]
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""
    modality: Literal["text", "visual_transcription"] = "text"
    image_version: str = ""
    transcription_uncertainties: list[str] = Field(default_factory=list)


class StudyContext(Record):
    problem: str = "unknown"
    mechanism: str = "unknown"
    task: str = "unknown"
    dataset: str = "unknown"
    baseline: str = "unknown"
    metric: str = "unknown"
    budget: str = "unknown"
    sample_size: str = "unknown"
    ablation: str = "unknown"
    negative_results: str = "unknown"
    implementation: str = "unknown"


class ClaimDraft(Record):
    statement: str = Field(min_length=1)
    kind: Literal["question", "method", "result", "premise", "limitation"] = "result"
    conditions: str = "unknown"
    experiment: str = "unknown"
    study: StudyContext = Field(default_factory=StudyContext)
    attribution: Literal["author_report", "system_inference"] = "author_report"
    evidence_ids: list[str] = Field(min_length=1)


class Claim(ClaimDraft):
    id: str
    paper_id: str
    version: str


class ClaimBatch(Record):
    uncovered_evidence_ids: list[str] = Field(default_factory=list)
    claims: list[ClaimDraft] = Field(max_length=25)


class CoreQuestion(Record):
    question: str = Field(min_length=1)
    claim_index: int = Field(ge=0)


class FocusedClaimBatch(ClaimBatch):
    core_questions: list[CoreQuestion] = Field(default_factory=list, max_length=2)


class RelationDraft(Record):
    claim_ids: list[str] = Field(min_length=2, max_length=2)
    relation: Literal["equivalent", "extends", "supports", "conflicts", "different_scope", "incomparable"]
    conditions: str
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)
    axes: list[ComparisonAxis] = Field(default_factory=list, max_length=8)
    unresolved_questions: list[str] = Field(default_factory=list)


class AnalysisRelation(RelationDraft):
    id: str
    version: str


class RelationBatch(Record):
    relations: list[RelationDraft] = Field(max_length=1)


class VisualObservation(Record):
    paper_id: str
    page: int = Field(ge=1)
    image_version: str
    locator: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    value: str = "unknown"
    unit: str = "unknown"
    conditions: str = "unknown"
    status: Literal["supports", "contradicts", "unclear"]


class Challenge(Record):
    premise: str = Field(min_length=1)
    alternative_explanation: str = Field(min_length=1)
    required_test: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class VerificationDraft(Record):
    visual_observations: list[VisualObservation] = Field(default_factory=list, max_length=20)
    challenges: list[Challenge] = Field(default_factory=list, max_length=5)
    verdict: Literal["verified", "qualified", "insufficient_evidence", "contradicted", "needs_human_review"]
    reason: str = Field(min_length=1)
    evidence_ids: list[str]
    repair_request: str = ""
    unchecked: list[str] = Field(default_factory=list)
    fidelity: JudgmentCheck = Field(default_factory=JudgmentCheck)
    experimental_support: JudgmentCheck = Field(default_factory=JudgmentCheck)
    counterevidence: JudgmentCheck = Field(default_factory=JudgmentCheck)
    dimensions: list[DimensionAssessment] = Field(default_factory=list, max_length=5)


class VerificationResult(VerificationDraft):
    visual_pages: list[dict] = Field(default_factory=list)
    target_id: str
    target_version: str
    scope: Literal["paper", "cross_paper"]
    source_levels: list[str]
    inspected_evidence_ids: list[str] = Field(default_factory=list)
    omitted_evidence_ids: list[str] = Field(default_factory=list)


class PaperCard(Record):
    paper_id: str
    title: str
    version: str
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    omitted_evidence_ids: list[str] = Field(default_factory=list)


class ParagraphDraft(Record):
    relation_ids: list[str] = Field(default_factory=list)
    text: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)


class ReportDraft(Record):
    paragraphs: list[ParagraphDraft] = Field(max_length=20)


class ReportParagraph(Record):
    relation_ids: list[str] = Field(default_factory=list)
    relation_context: dict[str, str] = Field(default_factory=dict)
    id: str
    version: str
    statement: str
    claim_ids: list[str]
    evidence_ids: list[str]


def validate_checkpoint_options(checkpoint, constraints=None):
    """Validate restored options before creating a run or calling a model."""
    if checkpoint is None:
        return
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("request"), dict):
        raise ValueError("恢复现场必须包含原始 request")
    original = checkpoint["request"].get("constraints", {})
    ResearchOptions.from_constraints(original)
    for key, value in (constraints or {}).items():
        default = {"output_mode": "paper_report", "cross_paper_enabled": False}.get(key)
        if value != original.get(key, default):
            raise ValueError("恢复任务不能修改原始约束；请创建新任务应用新的范围或输出模式")
