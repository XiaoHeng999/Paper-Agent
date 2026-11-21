from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from src.agents.Prompts import READ_DEEP_READ_SELECTION_PROMPT
from src.agents.readAgent import ReadAgentModelUnavailableError, build_read_agent, load_read_agent_llm
from src.utils.read_utils.read_fulltext import async_convert_fulltext_to_markdown, markdown_source_is_current
from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance, READ_RELEVANCE_STATUSES
from src.repositories.node_persistence.read_persistence import ReadPersistenceSink
from src.graph.checkpoint_halt import halt_with_checkpoint
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import WorkflowRuntimeResources
from src.graph.state_models import JsonObject, State
from src.research.models import ResearchOptions
from src.llm import ProviderSnapshot, SystemConfig
from src.paper_retrieval.download import async_download_paper_fulltext
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository
from src.utils.read_utils.chunkers import async_build_chunks_file, chunks_are_current


_FULL_TEXT_COMPLETED_STATUSES = frozenset({"indexed", "chunks_saved"})


class ReadResourceUnavailableError(RuntimeError):
    """表示阅读节点依赖的外部资源不可用，需要保存现场后等待用户处理。"""

    recovery_status = "waiting_resource"
    current_step = "read_waiting_resource"
    failure_stage = "read_resource_unavailable"
    diagnostic_key = "read_resource_unavailable"

    def __init__(self, message: str, *, details: JsonObject | None = None):
        """保存可展示错误信息和用于恢复现场的结构化细节。"""

        super().__init__(message)
        # 中文注释：details 只保存能写进 JSON 的普通数据，不保存 provider、repo
        # 这类运行时对象，避免 checkpoint 写盘或前端传递时无法序列化。
        self.details = dict(details or {})


class ReadModelUnavailableError(ReadResourceUnavailableError):
    """表示阅读模型当前不可用，需要保存现场后等待用户处理。

    这个异常只用于“用户修好模型后可以继续”的场景，例如：
    1. 没有配置阅读模型；
    2. 模型接口返回 HTTP/鉴权/限流/服务端错误；
    3. 调用模型时发生网络、超时、鉴权等异常。

    它和普通论文处理异常刻意区分：普通异常只跳过当前论文；模型返回了内容但
    内容质量不好时，也不算模型不可用，阅读节点会改用保守笔记继续处理。
    """

    recovery_status = "waiting_model"
    current_step = "read_waiting_model"
    failure_stage = "read_model_unavailable"
    diagnostic_key = "read_model_unavailable"


@dataclass(slots=True)
class PaperTaskInput:
    """保存一篇论文并发处理时需要的固定输入和恢复现场。"""

    paper: PaperDocument
    position: int
    restored_status: JsonObject
    restored_result: PaperReadResult | None = None


class CompletedPaperCounter:
    """记录已经完成的论文数量，方便并发任务实时上报进度。"""

    def __init__(self, initial_count: int):
        """用已经恢复的完成数量初始化计数器。"""

        self._count = max(0, int(initial_count))

    def current(self) -> int:
        """返回当前已经完成的论文数量。"""

        return self._count

    def increment(self) -> int:
        """在一篇论文真正完成并落盘后加一，并返回新值。"""

        self._count += 1
        return self._count


def run_read_node():
    """生成执行图中的阅读节点，直接返回异步节点实现。"""

    async def _node(state: State) -> State:
        """阅读节点已经具备 async 主流程，图层改成 ainvoke 后这里直接 await 即可。"""

        owned = "read_node_llm" not in state
        llm = _resolve_llm(state, SystemConfig.load().read.agent_name)
        try:
            return await _run_read_node_async({**state, "read_node_llm": llm})
        finally:
            if owned and llm is not None:
                await llm.aclose()

    return _node


async def _run_read_node_async(state: State) -> State:
    """并发处理多篇论文，并在资源不可用时按论文状态保存恢复现场。"""

    request = state.get("request")
    if request is None:
        raise ValueError("阅读节点缺少用户请求，无法判断论文主题")
    system_config = SystemConfig.load()
    config = system_config.read
    checkpoint = _checkpoint_from_state(state)
    papers = list(state.get("search_results") or _papers_from_payload(checkpoint.get("search_results")))
    if not request.constraints.get("papers"):
        papers = _deduplicate_papers(papers)

    reporter = _resolve_reporter(state)
    sink = _resolve_sink(state)
    runtime_resources = _resolve_runtime_resources(state)
    llm = _resolve_llm(state, config.agent_name)
    recovered_results = _restore_read_results(state, papers, checkpoint)
    results_by_paper_id = {result.paper.id: result for result in recovered_results}
    artifact_refs: list[JsonObject] = list(state.get("read_artifact_refs") or checkpoint.get("read_artifact_refs") or [])
    paper_runtime_statuses = _restore_paper_runtime_statuses(papers, checkpoint, recovered_results)
    # 中文说明：恢复时，正在进行全文处理的论文会保存在每篇论文的运行状态中。
    # 先把它们放回结果字典，后面统一重新计算名额，不沿用旧的先到先得计数。
    for paper in papers:
        restored_result = _paper_result_from_runtime_status(paper_runtime_statuses.get(paper.id) or {}, paper)
        if restored_result is not None:
            results_by_paper_id.setdefault(paper.id, restored_result)
    deep_read_limit = _deep_read_limit(request.constraints, len(papers))
    completed_counter = CompletedPaperCounter(len(results_by_paper_id))

    if reporter is not None:
        reporter.started(
            f"准备阅读 {len(papers)} 篇论文",
            stage="read_start",
            total=len(papers),
            completed=len(recovered_results),
            resumed_paper_count=len(recovered_results),
        )

    try:
        deep_read_count = await _process_papers_concurrently(
            papers=papers,
            topic=request.topic,
            constraints=request.constraints,
            config=config,
            llm=llm,
            reporter=reporter,
            sink=sink,
            runtime_resources=runtime_resources,
            results_by_paper_id=results_by_paper_id,
            artifact_refs=artifact_refs,
            paper_runtime_statuses=paper_runtime_statuses,
            completed_counter=completed_counter,
            deep_read_limit=deep_read_limit,
        )
    except ReadResourceUnavailableError as exc:
        deep_read_count = int(getattr(exc, "deep_read_count", 0))
        _halt_for_resource_unavailable(
            state,
            papers=papers,
            results=_ordered_results(papers, results_by_paper_id),
            artifact_refs=artifact_refs,
            position=_next_pending_position(papers, paper_runtime_statuses),
            error=exc,
            sink=sink,
            reporter=reporter,
            deep_read_count=deep_read_count,
            deep_read_limit=deep_read_limit,
            paper_runtime_statuses=paper_runtime_statuses,
        )

    results = _ordered_results(papers, results_by_paper_id)
    summary = _build_summary(results, deep_read_count)
    if sink is not None:
        try:
            persisted = await asyncio.to_thread(sink.persist_summary, summary, results)
            artifact_refs.extend(persisted.artifacts)
            summary["manifest"] = persisted.artifacts[0] if persisted.artifacts else {}
            if reporter is not None:
                for artifact in persisted.artifacts:
                    reporter.artifact(artifact, stage="read_artifact_ready")
        except Exception as exc:
            summary["persistence_error"] = str(exc)

    if reporter is not None:
        reporter.completed(
            "论文阅读节点已完成",
            stage="read_done",
            total=len(papers),
            completed=len(results),
            deep_read_paper_count=summary["deep_read_paper_count"],
            deep_read_papers=summary["deep_read_papers"],
            discarded_paper_count=summary["discarded_paper_count"],
            prepared_paper_count=summary["prepared_paper_count"],
        )
    # 中文注释：软丢弃的论文不进下游——汇总和产物里保留完整记录（含丢弃理由），
    # 但写回 state 的 read_results 只包含未被丢弃的论文，分析/写作节点自然收窄。
    kept_results = [result for result in results if not result.relevance.discarded]
    updated = dict(state)
    updated.update(
        read_results=[result.to_dict() for result in kept_results],
        read_summary=summary,
        read_artifact_refs=artifact_refs,
        read_paper_statuses=_ordered_paper_runtime_statuses(papers, paper_runtime_statuses),
        current_step="read",
    )
    if checkpoint:
        updated["search_results"] = papers
    updated.pop("read_resume_checkpoint", None)
    return cast(State, updated)


def _halt_for_resource_unavailable(
    state: State,
    *,
    papers: list[PaperDocument],
    results: list[PaperReadResult],
    artifact_refs: list[JsonObject],
    position: int,
    error: ReadResourceUnavailableError,
    sink: ReadPersistenceSink | None,
    reporter: Any,
    deep_read_count: int,
    deep_read_limit: int,
    paper_runtime_statuses: dict[str, JsonObject],
) -> NoReturn:
    """保存阅读节点现场并抛出明确错误，等待用户修好外部资源后恢复。"""

    # 中文注释：checkpoint 必须包含“继续执行所需的最小信息”：原始请求、全部论文、
    # 已完成结果、产物引用、当前位置和精读计数。具体字段由阅读节点负责构造，
    # 通用模块只负责写盘、上报、写回 state 和抛错。
    checkpoint = _build_resume_checkpoint(
        state,
        papers=papers,
        results=results,
        artifact_refs=artifact_refs,
        position=position,
        error=error,
        deep_read_count=deep_read_count,
        deep_read_limit=deep_read_limit,
        paper_runtime_statuses=paper_runtime_statuses,
    )
    halt_with_checkpoint(
        state,
        checkpoint=checkpoint,
        error=error,
        persist_checkpoint=sink.persist_checkpoint if sink is not None else None,
        reporter=reporter,
        results_payload=[result.to_dict() for result in results],
        artifact_refs=artifact_refs,
        checkpoint_key="read_resume_checkpoint",
        results_key="read_results",
        artifact_refs_key="read_artifact_refs",
        diagnostics_key=error.diagnostic_key,
        current_step=error.current_step,
        failure_stage=error.failure_stage,
        recovery_status=error.recovery_status,
        total=len(papers),
        completed=len(results),
        next_position=position,
    )


def _checkpoint_from_state(state: State) -> JsonObject:
    """读取调用方注入的阅读恢复现场。"""

    checkpoint = state.get("read_resume_checkpoint")
    return dict(checkpoint) if isinstance(checkpoint, dict) else {}


def _restore_read_results(state: State, papers: list[PaperDocument], checkpoint: JsonObject | None = None) -> list[PaperReadResult]:
    """从状态或 checkpoint 恢复已完成论文，避免重跑已保存的阅读结果。"""

    checkpoint = checkpoint or {}
    # 中文注释：优先使用 state 中较新的 read_results；如果恢复运行直接从
    # checkpoint 进入阅读节点，state 可能还没被 graph 合并，则退回 checkpoint。
    raw_results = list(state.get("read_results") or checkpoint.get("read_results") or [])
    papers_by_id = {paper.id: paper for paper in papers}
    restored: list[PaperReadResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        paper_payload = item.get("paper")
        paper = _paper_from_payload(paper_payload) if isinstance(paper_payload, dict) else None
        if paper is None:
            continue
        # 中文注释：如果当前检索结果里已经有同 ID 论文，使用当前对象，避免
        # checkpoint 中的旧 metadata 覆盖新版检索节点补充的字段。
        paper = papers_by_id.get(paper.id, paper)
        restored.append(
            PaperReadResult(
                paper=paper,
                note=_read_note_from_payload(item.get("note")),
                relevance=_read_relevance_from_payload(item.get("relevance")),
                full_text=_full_text_from_payload(item.get("full_text")),
                warnings=_string_list(item.get("warnings")),
            )
        )
    return restored[: len(papers)]


def _read_result_from_payload(value: Any, papers: list[PaperDocument]) -> PaperReadResult | None:
    """把 checkpoint 中单篇论文阅读结果恢复为 PaperReadResult。"""

    if not isinstance(value, dict):
        return None
    paper_payload = value.get("paper")
    paper = _paper_from_payload(paper_payload) if isinstance(paper_payload, dict) else None
    if paper is None:
        return None
    # 中文注释：优先使用当前论文列表中的对象，避免 checkpoint 里的旧字段覆盖新检索结果。
    papers_by_id = {item.id: item for item in papers}
    paper = papers_by_id.get(paper.id, paper)
    return PaperReadResult(
        paper=paper,
        note=_read_note_from_payload(value.get("note")),
        relevance=_read_relevance_from_payload(value.get("relevance")),
        full_text=_full_text_from_payload(value.get("full_text")),
        warnings=_string_list(value.get("warnings")),
    )


def _papers_from_payload(value: Any) -> list[PaperDocument]:
    """把 checkpoint 里的论文 JSON 恢复成 PaperDocument 列表。"""

    if not isinstance(value, list):
        return []
    papers: list[PaperDocument] = []
    for item in value:
        paper = _paper_from_payload(item)
        if paper is not None:
            papers.append(paper)
    return papers


def _paper_from_payload(value: Any) -> PaperDocument | None:
    """把论文 JSON 安全恢复为领域对象。"""

    if not isinstance(value, dict):
        return None
    paper_id = str(value.get("id") or "").strip()
    title = str(value.get("title") or "").strip()
    if not paper_id or not title:
        return None
    return PaperDocument(
        id=paper_id,
        title=title,
        authors=_string_list(value.get("authors")),
        abstract=str(value.get("abstract")) if value.get("abstract") is not None else None,
        year=_optional_int(value.get("year")),
        venue=str(value.get("venue")) if value.get("venue") is not None else None,
        url=str(value.get("url")) if value.get("url") is not None else None,
        pdf_url=str(value.get("pdf_url")) if value.get("pdf_url") is not None else None,
        doi=str(value.get("doi")) if value.get("doi") is not None else None,
        source=str(value.get("source")) if value.get("source") is not None else None,
        paperId=str(value.get("paperId")) if value.get("paperId") is not None else None,
        publication_date=str(value.get("publication_date") or ""),
        journal_conference=str(value.get("journal_conference") or value.get("journal/conference") or ""),
        volume=str(value.get("volume") or ""),
        issue=str(value.get("issue") or ""),
        language=str(value.get("language") or ""),
        metadata=dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), dict) else {},
    )


def _read_note_from_payload(value: Any) -> ReadNote:
    """把 checkpoint 中的笔记 JSON 恢复为 ReadNote。"""

    payload = value if isinstance(value, dict) else {}
    return ReadNote(
        main_question=_text_value(payload.get("main_question")),
        methods=_string_list(payload.get("methods")),
        datasets=_string_list(payload.get("datasets")),
        contributions=_string_list(payload.get("contributions")),
        limitations=_string_list(payload.get("limitations")),
        main_results=_string_list(payload.get("main_results")),
        short_summary=_text_value(payload.get("short_summary")),
        evidence_level=_text_value(payload.get("evidence_level")) or "metadata",
    )


def _read_relevance_from_payload(value: Any) -> ReadRelevance:
    """把 checkpoint 中的相关性 JSON 恢复为 ReadRelevance。

    中文注释：新版结构是 {relevant, reason, discarded, uncertain, status}。
    relevant 缺失时按"拿不准判有关"保底；旧版 checkpoint 里的 match_levels
    和 score 字段不再解析，统一按默认值恢复。
    """

    payload = value if isinstance(value, dict) else {}
    relevant = payload.get("relevant")
    if not isinstance(relevant, bool):
        relevant = True
    status = _text_value(payload.get("status")) or "pending"
    if status not in READ_RELEVANCE_STATUSES:
        status = "pending"
    return ReadRelevance(
        relevant=relevant,
        reason=_text_value(payload.get("reason")),
        discarded=bool(payload.get("discarded")),
        uncertain=bool(payload.get("uncertain")),
        status=status,
    )


def _full_text_from_payload(value: Any) -> FullTextStatus:
    """把 checkpoint 中的全文状态 JSON 恢复为 FullTextStatus。"""

    payload = value if isinstance(value, dict) else {}
    return FullTextStatus(
        status=_text_value(payload.get("status")) or "not_requested",
        reason=_text_value(payload.get("reason")),
        source_url=_optional_text(payload.get("source_url")),
        source_path=_optional_text(payload.get("source_path")),
        markdown_path=_optional_text(payload.get("markdown_path")),
        page_count=_optional_int(payload.get("page_count")),
        chunk_count=max(0, _optional_int(payload.get("chunk_count")) or 0),
    )


def _optional_text(value: Any) -> str | None:
    """把可选字段恢复为非空字符串或 None。"""

    text = str(value).strip() if isinstance(value, str) else ""
    return text or None


def _optional_int(value: Any) -> int | None:
    """把可选数字字段恢复为整数或 None。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_agent_from_llm(llm: ProviderSnapshot | None, usage_callback: Any | None = None):
    """根据节点已经解析出的模型快照创建阅读 Agent。"""

    return build_read_agent(llm, usage_callback=usage_callback)


DEFAULT_DEEP_READ_LIMIT = 10


def _deep_read_limit(constraints: JsonObject, total: int) -> int:
    """读取用户可选的精读数量限制，未提供时默认最多精读 10 篇。"""

    raw = constraints.get("deep_read_limit", constraints.get("max_deep_read", DEFAULT_DEEP_READ_LIMIT))
    try:
        return max(0, min(int(raw), total))
    except (TypeError, ValueError):
        return min(DEFAULT_DEEP_READ_LIMIT, total)


def _deduplicate_papers(papers: list[PaperDocument]) -> list[PaperDocument]:
    """按 DOI 优先、标题其次去除重复论文，保持检索节点原有排序。"""

    seen: set[str] = set()
    unique: list[PaperDocument] = []
    for paper in papers:
        key = f"doi:{paper.doi.strip().lower()}" if (paper.doi or "").strip() else f"title:{paper.title.strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique

def _resolve_runtime_resources(state: State) -> WorkflowRuntimeResources | None:
    """从运行时上下文里取出单次 run 共享的资源对象。"""

    runtime = state.get("runtime_context")
    if not isinstance(runtime, WorkflowRuntimeContext):
        return None
    return runtime.resources if isinstance(runtime.resources, WorkflowRuntimeResources) else None


def _resolve_llm(state: State, agent_name: str) -> ProviderSnapshot | None:
    """优先使用状态注入的阅读模型，正式运行时再从本地配置加载默认模型。"""

    injected = state.get("read_node_llm")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if "read_node_llm" not in state:
        return load_read_agent_llm(agent_name)
    return None


def _resolve_sink(state: State) -> ReadPersistenceSink | None:
    """仅在会话信息完整时启用阅读产物写盘，普通脚本调用仍可直接运行。"""

    repo = state.get("session_repo")
    session_key = str(state.get("session_key") or "").strip()
    turn_id = str(state.get("turn_id") or "").strip()
    if not isinstance(repo, SessionRepository) or not session_key or not turn_id:
        return None
    return ReadPersistenceSink(repo, session_key=session_key, turn_id=turn_id)


def _resolve_reporter(state: State):
    """从运行上下文中取出阅读节点上报器，没有前端同步接口时返回空值。"""

    runtime = state.get("runtime_context")
    if not isinstance(runtime, WorkflowRuntimeContext) or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("read", "论文阅读")


def _build_summary(results: list[PaperReadResult], deep_read_count: int) -> JsonObject:
    """汇总每篇论文的处理结果，让后续节点不必重新遍历全部阅读详情。"""

    global_statistics = {
        "total_papers_received": len(results),
        "passed_abstract_filter": sum(not item.relevance.discarded for item in results),
        "fulltext_downloaded": sum(bool(item.full_text.source_path) for item in results),
        "errors": _read_errors(results),
    }
    return {
        "total_paper_count": len(results),
        "deep_read_attempt_count": deep_read_count,
        "deep_read_candidate_count": sum(not item.relevance.discarded for item in results),
        # 中文说明：只要被统一排序选中，就保留在精读清单中；下载、解析或建立
        # 原文分块失败不会改变它已经获得全文名额这一事实。
        "deep_read_paper_count": sum(item.relevance.status == "selected_for_deep_read" for item in results),
        "deep_read_papers": [
            _deep_read_paper_item(item)
            for item in results
            if item.relevance.status == "selected_for_deep_read"
        ],
        # 中文说明：软丢弃清单单独汇总，方便审计"哪些论文被 AI 判无关、为什么"。
        "discarded_paper_count": sum(item.relevance.discarded for item in results),
        "discarded_papers": [
            {
                "paperId": item.paper.paperId or item.paper.id,
                "title": item.paper.title,
                "reason": item.relevance.reason,
            }
            for item in results
            if item.relevance.discarded
        ],
        "uncertain_paper_count": sum(item.relevance.uncertain for item in results),
        "prepared_paper_count": sum(item.full_text.status in _FULL_TEXT_COMPLETED_STATUSES for item in results),
        "failed_fulltext_count": sum(item.full_text.status in {"download_failed", "parse_failed"} for item in results),
        "subtopics": _read_subtopics(results),
        "global_statistics": global_statistics,
    }


def _text_value(value: Any) -> str:
    """把可能为空的模型字段安全整理为字符串。"""

    return str(value).strip() if isinstance(value, str) else ""


def _read_subtopics(results: list[PaperReadResult]) -> list[JsonObject]:
    """按检索子主题整理阅读结果。

    中文注释：检索节点会把论文来自哪个子主题放进 metadata.search_subtopics。
    如果没有这个信息，就放到“综合阅读”里，保证输出结构稳定。
    """

    grouped: dict[str, list[JsonObject]] = {}
    for result in results:
        names = _paper_subtopic_names(result.paper) or ["综合阅读"]
        for name in names:
            grouped.setdefault(name, []).append(_summary_paper_item(result))
    return [{"subtopic": name, "papers": papers} for name, papers in grouped.items()]


def _paper_subtopic_names(paper: PaperDocument) -> list[str]:
    """从论文 metadata 中取出子主题名称。"""

    origins = paper.metadata.get("search_subtopics") if isinstance(paper.metadata, dict) else []
    if not isinstance(origins, list):
        return []
    names: list[str] = []
    for origin in origins:
        if not isinstance(origin, dict):
            continue
        name = str(origin.get("subtopic") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _summary_paper_item(result: PaperReadResult) -> JsonObject:
    """整理 read_summary.subtopics[].papers[] 中的单篇论文。"""

    return {
        "paperId": result.paper.paperId or result.paper.id,
        "title": result.paper.title,
        "year": result.paper.year,
        "processing_status": "completed" if result.full_text.status in _FULL_TEXT_COMPLETED_STATUSES else result.full_text.status,
    }


def _deep_read_paper_item(result: PaperReadResult) -> JsonObject:
    """整理已经进入全文精读流程的论文，供汇总信息和前端详情展示。"""

    # 中文说明：这里同时保留全文状态，方便用户看出“已精读但未成功建立索引”的情况。
    return {
        "paperId": result.paper.paperId or result.paper.id,
        "title": result.paper.title,
        "year": result.paper.year,
        "relevance_reason": result.relevance.reason,
        "uncertain": result.relevance.uncertain,
        "selection_status": result.relevance.status,
        "full_text_status": result.full_text.status,
    }


def _paper_completion_runtime_status(result: PaperReadResult) -> str:
    """根据单篇论文结果判断卡片最终显示完成、丢弃还是失败。"""

    # 中文注释：软丢弃优先判断——被 AI 判无关的论文卡片显示"已丢弃"，
    # 不算失败（判断本身是正常业务结果），也不算完成（没有进入下游）。
    if result.relevance.discarded:
        return "discarded"
    # 中文注释：下载、转换、索引失败不再让整批阅读中断，
    # 但对这篇论文来说确实有阶段失败，所以卡片用 failed 更醒目。
    if result.full_text.status in {"download_failed", "no_url", "parse_failed"}:
        return "failed"
    if result.full_text.reason and result.full_text.status not in _FULL_TEXT_COMPLETED_STATUSES | {"not_requested"}:
        return "failed"
    if any("失败" in warning or "无法" in warning for warning in result.warnings):
        return "failed"
    return "completed"


def _paper_completion_message(result: PaperReadResult) -> str:
    """把单篇论文最终结果整理成用户能看懂的一句话。"""

    # 中文注释：丢弃卡片直接展示 AI 给出的判断理由，方便用户核对是否误杀。
    if result.relevance.discarded:
        return f"已丢弃：{result.relevance.reason}" if result.relevance.reason else "已丢弃：判定与主题无关"
    status = result.full_text.status
    if status in {"download_failed", "no_url"}:
        return "全文下载失败，已保留摘要阅读结果"
    if status == "parse_failed":
        return "Markdown 转换失败，已保留摘要阅读结果"
    if result.full_text.reason and status not in _FULL_TEXT_COMPLETED_STATUSES | {"not_requested"}:
        return "原文准备失败，已保留阅读结果"
    if status == "not_requested":
        return result.full_text.reason or "摘要阅读完成，当前论文不需要全文精读"
    return "论文阅读完成"


def _paper_completion_error_message(result: PaperReadResult) -> str:
    """提取单篇论文最终失败原因，没有失败时返回空字符串。"""

    # 中文注释：软丢弃不是错误，不产生 error 信息。
    if result.relevance.discarded:
        return ""
    if result.full_text.status in {"download_failed", "no_url", "parse_failed"}:
        return result.full_text.reason or result.full_text.status
    if result.full_text.reason and result.full_text.status not in _FULL_TEXT_COMPLETED_STATUSES | {"not_requested"}:
        return result.full_text.reason
    for warning in result.warnings:
        if "失败" in warning or "无法" in warning:
            return warning
    return ""


def _read_errors(results: list[PaperReadResult]) -> list[JsonObject]:
    """把下载、解析和分块失败整理成统一错误列表。"""

    errors: list[JsonObject] = []
    for result in results:
        # 中文注释：被软丢弃的论文不算错误，跳过统计。
        if result.relevance.discarded:
            continue
        paper_id = result.paper.paperId or result.paper.id
        status = result.full_text.status
        if status in {"download_failed", "no_url"}:
            errors.append({"paperId": paper_id, "stage": "download", "reason": result.full_text.reason})
        elif status == "parse_failed":
            errors.append({"paperId": paper_id, "stage": "parse", "reason": result.full_text.reason})
        elif result.full_text.reason and status not in _FULL_TEXT_COMPLETED_STATUSES:
            errors.append({"paperId": paper_id, "stage": "source_preparation", "reason": result.full_text.reason})
    return errors


def _string_list(value: Any) -> list[str]:
    """只保留模型返回列表中的非空字符串，避免错误类型进入结构化笔记。"""

    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


_FALLBACK_PAPER_TASK_SEMAPHORE = asyncio.Semaphore(3)
_FALLBACK_READ_MODEL_SEMAPHORE = asyncio.Semaphore(2)


async def _process_papers_concurrently(
    *,
    papers: list[PaperDocument],
    topic: str,
    constraints: JsonObject,
    config: Any,
    llm: ProviderSnapshot | None,
    reporter: Any,
    sink: ReadPersistenceSink | None,
    runtime_resources: WorkflowRuntimeResources | None,
    results_by_paper_id: dict[str, PaperReadResult],
    artifact_refs: list[JsonObject],
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    deep_read_limit: int,
) -> int:
    """先并发阅读全部摘要，再统一选择全文精读论文。"""

    # 中文说明：第一阶段只调用一次摘要模型，不会下载任何全文。这样模型较早返回的论文
    # 也不能提前占用全文名额，所有论文都拿到固定分数后才会一起排序。
    abstract_items = [
        PaperTaskInput(paper=paper, position=position, restored_status={}, restored_result=None)
        for position, paper in enumerate(papers, start=1)
        if paper.id not in results_by_paper_id
    ]
    await _run_paper_task_batch(
        abstract_items,
        topic=topic,
        constraints=constraints,
        config=config,
        llm=llm,
        reporter=reporter,
        sink=sink,
        runtime_resources=runtime_resources,
        results_by_paper_id=results_by_paper_id,
        artifact_refs=artifact_refs,
        paper_runtime_statuses=paper_runtime_statuses,
        completed_counter=completed_counter,
        total_paper_count=len(papers),
        process_full_text=False,
        selected_count=0,
    )

    selected_paper_ids = await _assign_deep_read_selection(
        papers,
        results_by_paper_id,
        paper_runtime_statuses,
        deep_read_limit,
        topic=topic,
        llm=llm,
        allow_cross_selection=ResearchOptions.from_constraints(constraints).cross_paper_enabled,
    )

    # 没有获得全文名额的论文到这里已经完成，立即保存其结构化摘要。
    for position, paper in enumerate(papers, start=1):
        result = results_by_paper_id[paper.id]
        if paper.id not in selected_paper_ids:
            await _finalize_paper_result(
                result,
                paper_position=position,
                results_by_paper_id=results_by_paper_id,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=len(papers),
                reporter=reporter,
                sink=sink,
                artifact_refs=artifact_refs,
            )

    # 中文说明：第二阶段只处理统一排序选中的论文。恢复运行时，已经下载或转换过的
    # 论文会从保存的文件继续，而不是再次调用摘要模型。
    deep_read_items = [
        PaperTaskInput(
            paper=paper,
            position=position,
            restored_status=dict(paper_runtime_statuses.get(paper.id) or {}),
            restored_result=results_by_paper_id[paper.id],
        )
        for position, paper in enumerate(papers, start=1)
        if paper.id in selected_paper_ids and _deep_read_needs_processing(results_by_paper_id[paper.id])
    ]
    await _run_paper_task_batch(
        deep_read_items,
        topic=topic,
        constraints=constraints,
        config=config,
        llm=llm,
        reporter=reporter,
        sink=sink,
        runtime_resources=runtime_resources,
        results_by_paper_id=results_by_paper_id,
        artifact_refs=artifact_refs,
        paper_runtime_statuses=paper_runtime_statuses,
        completed_counter=completed_counter,
        total_paper_count=len(papers),
        process_full_text=True,
        selected_count=len(selected_paper_ids),
    )
    return len(selected_paper_ids)


async def _run_paper_task_batch(
    work_items: list[PaperTaskInput],
    *,
    topic: str,
    constraints: JsonObject,
    config: Any,
    llm: ProviderSnapshot | None,
    reporter: Any,
    sink: ReadPersistenceSink | None,
    runtime_resources: WorkflowRuntimeResources | None,
    results_by_paper_id: dict[str, PaperReadResult],
    artifact_refs: list[JsonObject],
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    process_full_text: bool,
    selected_count: int,
) -> None:
    """执行一个阶段的论文任务，并在资源不可用时保留已完成结果。"""

    if not work_items:
        return
    semaphore = runtime_resources.paper_task_semaphore if runtime_resources is not None else _FALLBACK_PAPER_TASK_SEMAPHORE
    tasks = {
        asyncio.create_task(
            _run_one_paper_task(
                item,
                topic=topic,
                constraints=constraints,
                config=config,
                llm=llm,
                reporter=reporter,
                runtime_resources=runtime_resources,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=total_paper_count,
                semaphore=semaphore,
                process_full_text=process_full_text,
            )
        ): item
        for item in work_items
    }
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        resource_error: ReadResourceUnavailableError | None = None
        for task in done:
            item = tasks[task]
            try:
                result = task.result()
            except ReadResourceUnavailableError as exc:
                resource_error = resource_error or exc
                continue
            results_by_paper_id[item.paper.id] = result
            if not process_full_text:
                _set_paper_runtime_status(
                    paper_runtime_statuses,
                    item.paper,
                    status="running",
                    current_stage="abstract_ready",
                    result=result,
                    error_message="",
                )
                continue
            await _finalize_paper_result(
                result,
                paper_position=item.position,
                results_by_paper_id=results_by_paper_id,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=total_paper_count,
                reporter=reporter,
                sink=sink,
                artifact_refs=artifact_refs,
            )
        if resource_error is not None:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            setattr(resource_error, "deep_read_count", selected_count)
            raise resource_error


async def _finalize_paper_result(
    result: PaperReadResult,
    *,
    paper_position: int,
    results_by_paper_id: dict[str, PaperReadResult],
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    reporter: Any,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
) -> None:
    """保存一篇最终结果，并更新前端看到的完成进度。"""

    results_by_paper_id[result.paper.id] = result
    completion_status = _paper_completion_runtime_status(result)
    completion_message = _paper_completion_message(result)
    completion_error = _paper_completion_error_message(result)
    _set_paper_runtime_status(
        paper_runtime_statuses,
        result.paper,
        status=completion_status,
        current_stage="paper_completed",
        result=result,
        error_message=completion_error,
    )
    completed = completed_counter.increment()
    _report_progress(
        reporter,
        result.paper,
        "paper_completed",
        completed,
        total_paper_count,
        runtime_status=completion_status,
        message=completion_message,
        current_status=result.full_text.status,
        paper_position=paper_position,
        error_message=completion_error,
    )
    _, save_error = await _persist_completed_paper_async(
        result,
        sink=sink,
        artifact_refs=artifact_refs,
        reporter=reporter,
        completed=completed,
        total=total_paper_count,
        paper_position=paper_position,
    )
    _set_paper_runtime_status(
        paper_runtime_statuses,
        result.paper,
        # 中文注释：软丢弃的论文卡片状态保持 discarded，不伪装成 completed；
        # 保存出错或阶段失败才是 failed，其余为 completed。
        status=(
            "failed"
            if save_error or completion_status == "failed"
            else "discarded"
            if completion_status == "discarded"
            else "completed"
        ),
        current_stage="paper_artifact_ready",
        result=result,
        error_message=save_error or completion_error,
    )


MIN_USABLE_PAPERS = 3


async def _assign_deep_read_selection(
    papers: list[PaperDocument],
    results_by_paper_id: dict[str, PaperReadResult],
    paper_runtime_statuses: dict[str, JsonObject],
    deep_read_limit: int,
    *,
    topic: str,
    llm: ProviderSnapshot | None,
    allow_cross_selection: bool = True,
) -> set[str]:
    """摘要阅读后统一分配去留与全文精读名额。

    中文注释：新版不再按分数排序，而是三步：
    1. 模型判无关（relevant=False）→ 软丢弃（标记 discarded，不进下游，笔记照常落盘）；
    2. 判相关的论文少于保底数量 → 从丢弃池按候选池顺序回捞，标注"相关性存疑"；
    3. 相关论文超过精读预算 → 触发一次批量比较，由模型在同一上下文里挑出精读对象；
       不超过预算则全部精读，零新增调用。
    """

    candidates: list[tuple[int, PaperReadResult]] = []
    discarded_items: list[tuple[int, PaperReadResult]] = []
    for position, paper in enumerate(papers, start=1):
        result = results_by_paper_id[paper.id]
        if not result.relevance.relevant:
            # 软丢弃：只标记，不删除对象；note.json 会带 discarded 字段照常落盘。
            result.relevance.discarded = True
            result.relevance.status = "discarded"
            result.full_text = FullTextStatus(status="not_requested", reason="判定与主题无关，已丢弃")
            discarded_items.append((position, result))
            continue
        candidates.append((position, result))

    # 保底回捞：可用论文太少时按候选池顺序从丢弃池捞回，标注存疑。
    if len(candidates) < MIN_USABLE_PAPERS and discarded_items:
        for position, result in list(discarded_items):
            if len(candidates) >= MIN_USABLE_PAPERS:
                break
            discarded_items.remove((position, result))
            result.relevance.discarded = False
            result.relevance.uncertain = True
            result.relevance.status = "pending"
            result.relevance.reason = f"{result.relevance.reason}；回捞保留，相关性存疑"
            result.full_text = FullTextStatus(status="not_requested", reason="")
            candidates.append((position, result))

    # 精读名额分配：预算内全取；超预算时一次批量比较选择。
    candidate_ids = [result.paper.id for _, result in candidates]
    if len(candidates) <= deep_read_limit or not allow_cross_selection:
        selected_paper_ids = set(candidate_ids[:deep_read_limit])
    else:
        selected_paper_ids = await _select_deep_read_by_llm(
            topic=topic,
            candidates=[result for _, result in candidates],
            limit=deep_read_limit,
            llm=llm,
        )
        # 模型比较失败时 _select_deep_read_by_llm 已回退为池内顺序取前 N。
    for _, result in candidates:
        if result.paper.id not in selected_paper_ids:
            result.relevance.status = "not_selected_due_to_limit"
            result.full_text = FullTextStatus(status="not_requested", reason="全文精读名额已分配给更贴近主题的论文")
            continue
        result.relevance.status = "selected_for_deep_read"
        current = dict(paper_runtime_statuses.get(result.paper.id) or {})
        if _deep_read_needs_processing(result) and not str(current.get("status") or "").startswith("waiting_"):
            _set_paper_runtime_status(
                paper_runtime_statuses,
                result.paper,
                status="pending",
                current_stage="selected_for_deep_read",
                result=result,
                error_message="",
                deep_read_reserved=True,
            )
    return selected_paper_ids


async def _select_deep_read_by_llm(
    *,
    topic: str,
    candidates: list[PaperReadResult],
    limit: int,
    llm: ProviderSnapshot | None,
) -> set[str]:
    """相关论文超出精读预算时，让模型在同一上下文里比较并挑出精读对象。

    中文注释：这是"一次批量比较"而不是"逐篇打分"——所有候选放进同一个请求，
    模型可以直接对比它们的相对价值。任何环节失败都回退为按候选池顺序取前 N 篇，
    保证精读流程不会因为比较调用出错而中断。
    """

    fallback_ids = {result.paper.id for result in candidates[:limit]}
    if llm is None:
        return fallback_ids
    lines = []
    for index, result in enumerate(candidates):
        summary = (result.note.short_summary or "").strip() or (result.paper.abstract or "").strip()[:150]
        lines.append(f"[{index}] {result.paper.title}\n    {summary}")
    user_prompt = (
        f"用户研究主题：{topic}\n\n"
        f"精读名额：{limit}\n\n以下是全部相关论文的编号、标题和摘要要点：\n\n" + "\n".join(lines)
    )
    messages: list[JsonObject] = [
        {"role": "system", "content": READ_DEEP_READ_SELECTION_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        content = await _call_read_llm(llm.provider, messages)
    except Exception:
        # 比较调用失败时按池内顺序回退，宁可按原排序精读，也不中断流程。
        return fallback_ids
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return fallback_ids
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return fallback_ids
    selected_indices = payload.get("selected") if isinstance(payload, dict) else None
    if not isinstance(selected_indices, list):
        return fallback_ids
    selected_ids = {
        candidates[index].paper.id
        for index in selected_indices
        if isinstance(index, int) and 0 <= index < len(candidates)
    }
    if not selected_ids:
        return fallback_ids
    return set(list(selected_ids)[:limit])


async def _call_read_llm(provider: Any, messages: list[JsonObject]) -> str:
    """调用阅读侧模型并取回文本内容，优先走异步 chat 接口。"""

    chat = getattr(provider, "chat", None)
    if callable(chat):
        result = chat(messages, temperature=0)
        if inspect.isawaitable(result):
            result = await result
        return getattr(result, "content", "") or ""
    result = await asyncio.to_thread(provider.chat_with_retry, messages, temperature=0)
    return getattr(result, "content", "") or ""


def _deep_read_needs_processing(result: PaperReadResult) -> bool:
    """判断已选中论文是否还需要继续全文流程。"""

    if result.full_text.status in _FULL_TEXT_COMPLETED_STATUSES:
        path = result.full_text.markdown_path
        source = result.full_text.source_path
        if path and source and Path(source).is_file() and not markdown_source_is_current(result.paper, Path(source), Path(path)):
            return True
        return not path or not chunks_are_current(result.paper, Path(path))
    return result.full_text.status not in {"download_failed", "parse_failed", "no_url"}


async def _run_one_paper_task(
    item: PaperTaskInput,
    *,
    topic: str,
    constraints: JsonObject,
    config: Any,
    llm: ProviderSnapshot | None,
    reporter: Any,
    runtime_resources: WorkflowRuntimeResources | None,
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    semaphore: asyncio.Semaphore,
    process_full_text: bool,
) -> PaperReadResult:
    """在线程池外层控制论文级并发，并把单篇异常隔离在当前论文内部。"""

    async with semaphore:
        try:
            return await _read_one_paper(
                item,
                topic=topic,
                constraints=constraints,
                llm=llm,
                config=config,
                reporter=reporter,
                runtime_resources=runtime_resources,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=total_paper_count,
                process_full_text=process_full_text,
            )
        except ReadResourceUnavailableError as exc:
            _mark_paper_waiting_resource(paper_runtime_statuses, item.paper, exc)
            # 中文注释：资源不可用会让整个阅读节点暂停。这里先更新当前论文卡片，
            # 让用户一眼看到具体是哪篇论文卡在模型服务上。
            _report_progress(
                reporter,
                item.paper,
                _paper_stage_from_status(paper_runtime_statuses.get(item.paper.id)),
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message=str(exc),
                paper_position=item.position,
                error_message=str(exc),
                recovery_status=exc.recovery_status,
            )
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = f"当前论文处理失败：{exc}"
            result = PaperReadResult(
                paper=item.paper,
                full_text=FullTextStatus(status="not_requested", reason="当前论文处理发生异常"),
                warnings=[message],
            )
            _set_paper_runtime_status(
                paper_runtime_statuses,
                item.paper,
                status="failed",
                current_stage=_paper_stage_from_status(paper_runtime_statuses.get(item.paper.id)),
                result=result,
                error_message=message,
            )
            _report_progress(
                reporter,
                item.paper,
                _paper_stage_from_status(paper_runtime_statuses.get(item.paper.id)),
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message="当前论文处理失败",
                current_status=result.full_text.status,
                paper_position=item.position,
                error_message=message,
            )
            return result


async def _read_one_paper(
    item: PaperTaskInput,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
    config: Any,
    reporter: Any,
    runtime_resources: WorkflowRuntimeResources | None,
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    process_full_text: bool,
) -> PaperReadResult:
    """执行摘要阶段或已选中论文的全文阶段。"""

    paper = item.paper
    title = paper.title.strip()
    paper_usage = {"input_tokens": 0, "output_tokens": 0}
    if not title:
        result = PaperReadResult(
            paper=paper,
            full_text=FullTextStatus(status="not_requested", reason="论文没有标题，无法可靠阅读"),
            warnings=["论文缺少标题，未交给模型判断"],
        )
        _report_progress(
            reporter,
            paper,
            "paper_completed",
            completed_counter.current(),
            total_paper_count,
            runtime_status="failed",
            message="论文缺少标题，无法阅读",
            current_status=result.full_text.status,
            paper_position=item.position,
            error_message="论文缺少标题，未交给模型判断",
        )
        return result

    result = item.restored_result
    if not process_full_text:
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage="reading_abstract",
            result=result,
            error_message="",
        )
        _report_progress(
            reporter,
            paper,
            "reading_abstract",
            completed_counter.current(),
            total_paper_count,
            paper_position=item.position,
        )
        def report_abstract_usage(usage: JsonObject) -> None:
            """把单篇论文摘要模型的真实 token 用量写入该论文卡片。"""

            paper_usage["input_tokens"] += int(usage.get("input_tokens") or 0)
            paper_usage["output_tokens"] += int(usage.get("output_tokens") or 0)
            _report_progress(
                reporter,
                paper,
                "reading_abstract",
                completed_counter.current(),
                total_paper_count,
                **paper_usage,
            )

        note, relevance, warnings = await _build_abstract_note(
            paper,
            topic=topic,
            constraints=constraints,
            llm=llm,
            runtime_resources=runtime_resources,
            usage_callback=report_abstract_usage,
        )
        result = PaperReadResult(paper=paper, note=note, relevance=relevance, warnings=warnings)
        if constraints.get("papers"):
            # Explicitly selected papers must not disappear through relevance screening.
            result.relevance.relevant = True
        return result

    if result is None:
        raise ValueError("全文阶段缺少摘要阅读结果")
    if result.relevance.status != "selected_for_deep_read":
        return result

    restored_stage = str(item.restored_status.get("current_stage") or "").strip()
    resume_stages = {"downloading_full_text", "converting_markdown", "chunking_full_text", "saving_chunks"}
    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="running",
        current_stage=restored_stage if restored_stage in resume_stages else "selected_for_deep_read",
        result=result,
        error_message="",
        deep_read_reserved=True,
    )

    full_text = result.full_text
    source_path = Path(full_text.source_path) if full_text.source_path else None
    if source_path is None or not source_path.is_file():
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage="downloading_full_text",
            result=result,
            error_message="",
        )
        _report_progress(
            reporter,
            paper,
            "downloading_full_text",
            completed_counter.current(),
            total_paper_count,
            paper_position=item.position,
        )
        downloaded = await async_download_paper_fulltext(
            paper,
            cache_dir=config.paper_cache_dir,
            connect_timeout_seconds=config.connect_timeout_seconds,
            download_timeout_seconds=config.download_timeout_seconds,
            max_file_size_mb=config.max_file_size_mb,
            runtime_resources=runtime_resources,
        )
        full_text = FullTextStatus(status=downloaded.status, reason=downloaded.reason, source_url=downloaded.source_url)
        if downloaded.file_path is None:
            result.full_text = full_text
            _report_progress(
                reporter,
                paper,
                "paper_completed",
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message="全文下载失败，已保留摘要阅读结果",
                current_status=result.full_text.status,
                paper_position=item.position,
                error_message=full_text.reason or "全文下载失败",
            )
            return result
        source_path = downloaded.file_path
        full_text.source_path = str(source_path)
    else:
        full_text.source_path = str(source_path)

    markdown_path = Path(full_text.markdown_path) if full_text.markdown_path else None
    if markdown_path is None or not await asyncio.to_thread(markdown_source_is_current, paper, source_path, markdown_path):
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage="converting_markdown",
            result=result,
            error_message="",
        )
        _report_progress(
            reporter,
            paper,
            "converting_markdown",
            completed_counter.current(),
            total_paper_count,
            paper_position=item.position,
        )
        converted = await async_convert_fulltext_to_markdown(
            paper,
            source_path=source_path,
            source_url=full_text.source_url,
        )
        if converted.markdown_path is None:
            result.full_text = FullTextStatus(
                status="parse_failed",
                reason="；".join(converted.warnings) or "无法将全文转换为 Markdown",
                source_url=full_text.source_url,
                source_path=str(source_path),
            )
            _report_progress(
                reporter,
                paper,
                "paper_completed",
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message="Markdown 转换失败，已保留摘要阅读结果",
                current_status=result.full_text.status,
                paper_position=item.position,
                error_message=result.full_text.reason or "无法将全文转换为 Markdown",
            )
            return result
        full_text.status = "markdown_ready"
        full_text.markdown_path = str(converted.markdown_path)
        full_text.page_count = converted.page_count
        if converted.warnings:
            result.warnings.extend(converted.warnings)
        markdown_path = converted.markdown_path
    else:
        full_text.status = "markdown_ready"
        full_text.markdown_path = str(markdown_path)

    result.full_text = full_text
    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="running",
        current_stage="chunking_full_text",
        result=result,
        error_message="",
    )
    _report_progress(
        reporter,
        paper,
        "chunking_full_text",
        completed_counter.current(),
        total_paper_count,
        paper_position=item.position,
    )
    chunk_build = await async_build_chunks_file(paper, markdown_path=markdown_path)
    full_text.chunk_count = len(chunk_build.chunks)
    result.full_text = full_text

    full_text.status = "chunks_saved"
    full_text.reason = "原文已准备，交由论文判断与核查阶段处理"
    return result


async def _build_abstract_note(
    paper: PaperDocument,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
    runtime_resources: WorkflowRuntimeResources | None,
    usage_callback: Any | None = None,
) -> tuple[ReadNote, ReadRelevance, list[str]]:
    """通过 ReadAgent 完成异步摘要阅读，节点只保留流程控制和暂停恢复逻辑。"""

    semaphore = runtime_resources.read_model_semaphore if runtime_resources is not None else _FALLBACK_READ_MODEL_SEMAPHORE
    try:
        result = await _read_agent_from_llm(llm, usage_callback=usage_callback).async_read_abstract(
            paper,
            topic=topic,
            constraints=constraints,
            semaphore=semaphore,
            usage_callback=usage_callback,
        )
        return result.as_tuple()
    except ReadAgentModelUnavailableError as exc:
        raise ReadModelUnavailableError(str(exc)) from exc


async def _persist_completed_paper_async(
    result: PaperReadResult,
    *,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
    reporter: Any,
    completed: int,
    total: int,
    paper_position: int,
) -> tuple[bool, str]:
    """在 async 阅读流程里把单篇结果落盘，并更新这篇论文自己的卡片。"""

    if sink is None:
        return True, ""
    try:
        persisted = await asyncio.to_thread(sink.persist_paper, result)
        artifact_refs.extend(persisted.artifacts)
        if reporter is not None:
            # 中文注释：产物列表仍然需要 artifact 事件，但这里关闭自动生成保存步骤卡片。
            # 保存成功这件事由下面的 _report_progress 更新当前论文卡片，避免前端多出一张卡片。
            for artifact in persisted.artifacts:
                reporter.artifact(artifact, stage="paper_artifact_ready", emit_runtime_event=False)
        completion_status = _paper_completion_runtime_status(result)
        _report_progress(
            reporter,
            result.paper,
            "paper_artifact_ready",
            completed,
            total,
            runtime_status=completion_status,
            message="单篇阅读结果已保存" if completion_status == "completed" else "单篇阅读结果已保存，但阅读过程中有阶段失败",
            current_status=result.full_text.status,
            paper_position=paper_position,
            artifact_count=len(persisted.artifacts),
            error_message=_paper_completion_error_message(result),
        )
        return True, ""
    except Exception as exc:
        message = f"阅读结果无法保存到会话目录：{exc}"
        result.warnings.append(message)
        _report_progress(
            reporter,
            result.paper,
            "paper_artifact_ready",
            completed,
            total,
            runtime_status="failed",
            message="单篇阅读结果保存失败",
            current_status=result.full_text.status,
            paper_position=paper_position,
            error_message=message,
        )
        return False, message


def _restore_paper_runtime_statuses(
    papers: list[PaperDocument],
    checkpoint: JsonObject,
    recovered_results: list[PaperReadResult],
) -> dict[str, JsonObject]:
    """从 checkpoint 中恢复每篇论文自己的处理状态，供并发任务继续运行。"""

    statuses = {paper.id: _default_paper_runtime_status(paper) for paper in papers}
    for item in list(checkpoint.get("paper_runtime_statuses") or []):
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        paper = next((candidate for candidate in papers if candidate.id == paper_id), None)
        if paper is None:
            continue
        statuses[paper_id] = _merge_runtime_status(statuses[paper_id], item, paper)

    for result in recovered_results:
        statuses[result.paper.id] = {
            **statuses.get(result.paper.id, _default_paper_runtime_status(result.paper)),
            "paper_id": result.paper.id,
            "paper_title": result.paper.title,
            # 中文注释：恢复时按论文实际完成状态回填，被软丢弃的论文保持 discarded，
            # 不能统一标成 completed，否则前端卡片和后续判断都会出错。
            "status": _paper_completion_runtime_status(result),
            "current_stage": "paper_completed",
            "source_path": result.full_text.source_path,
            "markdown_path": result.full_text.markdown_path,
            "error_message": "",
            "deep_read_reserved": _infer_deep_read_reserved(result),
            "result": result.to_dict(),
        }
    return statuses


def _build_pending_paper_tasks(
    papers: list[PaperDocument],
    paper_runtime_statuses: dict[str, JsonObject],
    results_by_paper_id: dict[str, PaperReadResult],
) -> list[PaperTaskInput]:
    """找出还没完成的论文，整理成待投递的并发任务输入。"""

    tasks: list[PaperTaskInput] = []
    for position, paper in enumerate(papers, start=1):
        if paper.id in results_by_paper_id:
            continue
        status = dict(paper_runtime_statuses.get(paper.id) or _default_paper_runtime_status(paper))
        if str(status.get("status") or "") == "completed":
            continue
        tasks.append(
            PaperTaskInput(
                paper=paper,
                position=position,
                restored_status=status,
                restored_result=_paper_result_from_runtime_status(status, paper),
            )
        )
    return tasks


def _ordered_results(papers: list[PaperDocument], results_by_paper_id: dict[str, PaperReadResult]) -> list[PaperReadResult]:
    """把并发完成的论文结果重新按原论文顺序排好，方便后续节点稳定消费。"""

    return [results_by_paper_id[paper.id] for paper in papers if paper.id in results_by_paper_id]


def _ordered_paper_runtime_statuses(
    papers: list[PaperDocument],
    paper_runtime_statuses: dict[str, JsonObject],
) -> list[JsonObject]:
    """把每篇论文的运行状态按原顺序输出，便于恢复和前端展示。"""

    return [dict(paper_runtime_statuses.get(paper.id) or _default_paper_runtime_status(paper)) for paper in papers]


def _next_pending_position(papers: list[PaperDocument], paper_runtime_statuses: dict[str, JsonObject]) -> int:
    """找出当前批次里第一篇未完成论文的位置，兼容旧前端依赖的 next_position 字段。"""

    for position, paper in enumerate(papers, start=1):
        status = str((paper_runtime_statuses.get(paper.id) or {}).get("status") or "pending")
        if status != "completed":
            return position
    return len(papers) + 1


def _default_paper_runtime_status(paper: PaperDocument) -> JsonObject:
    """给每篇论文准备一个最基础的运行状态记录。"""

    return {
        "paper_id": paper.id,
        "paper_title": paper.title,
        "status": "pending",
        "current_stage": "pending",
        "source_path": None,
        "markdown_path": None,
        "error_message": "",
        "deep_read_reserved": False,
        "result": None,
    }


def _merge_runtime_status(base: JsonObject, payload: JsonObject, paper: PaperDocument) -> JsonObject:
    """把 checkpoint 里恢复出来的论文状态整理成统一结构。"""

    result_payload = payload.get("result")
    result = _paper_result_from_runtime_status({"result": result_payload}, paper)
    return {
        **base,
        "paper_id": paper.id,
        "paper_title": paper.title,
        "status": str(payload.get("status") or base.get("status") or "pending"),
        "current_stage": str(payload.get("current_stage") or base.get("current_stage") or "pending"),
        "source_path": _optional_text(payload.get("source_path")) or (result.full_text.source_path if result is not None else base.get("source_path")),
        "markdown_path": _optional_text(payload.get("markdown_path")) or (result.full_text.markdown_path if result is not None else base.get("markdown_path")),
        "error_message": str(payload.get("error_message") or base.get("error_message") or ""),
        "deep_read_reserved": bool(payload.get("deep_read_reserved") if payload.get("deep_read_reserved") is not None else (result is not None and _infer_deep_read_reserved(result))),
        "result": result.to_dict() if result is not None else base.get("result"),
    }


def _paper_result_from_runtime_status(payload: JsonObject, paper: PaperDocument) -> PaperReadResult | None:
    """从单篇论文状态里恢复出中间结果，便于从下载后或 Markdown 后继续执行。"""

    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        return None
    merged = dict(result_payload)
    merged["paper"] = paper.to_dict()
    return _read_result_from_payload(merged, [paper])


def _set_paper_runtime_status(
    paper_runtime_statuses: dict[str, JsonObject],
    paper: PaperDocument,
    *,
    status: str,
    current_stage: str,
    result: PaperReadResult | None,
    error_message: str,
    deep_read_reserved: bool | None = None,
) -> None:
    """更新某篇论文的当前状态，保证 checkpoint 里随时都有可恢复的信息。"""

    current = dict(paper_runtime_statuses.get(paper.id) or _default_paper_runtime_status(paper))
    current.update(
        paper_id=paper.id,
        paper_title=paper.title,
        status=status,
        current_stage=current_stage,
        error_message=error_message,
    )
    if result is not None:
        current["result"] = result.to_dict()
        current["source_path"] = result.full_text.source_path
        current["markdown_path"] = result.full_text.markdown_path
    if deep_read_reserved is not None:
        current["deep_read_reserved"] = bool(deep_read_reserved)
    paper_runtime_statuses[paper.id] = current


def _paper_stage_from_status(status: JsonObject | None) -> str:
    """从论文状态里取当前阶段，缺失时给前端一个稳定的默认阶段。"""

    # 中文注释：异常处理可能发生在任意位置。这里不猜测复杂流程，
    # 只读取已经记录过的 current_stage；如果还没记录，就默认算作摘要阅读阶段。
    stage = str((status or {}).get("current_stage") or "").strip()
    return stage or "reading_abstract"


def _mark_paper_waiting_resource(
    paper_runtime_statuses: dict[str, JsonObject],
    paper: PaperDocument,
    error: ReadResourceUnavailableError,
) -> None:
    """在模型不可用时，把当前论文的中间结果和等待原因记下来。"""

    result = _paper_result_from_runtime_status(
        paper_runtime_statuses.get(paper.id) or {},
        paper,
    )
    current_stage = str((paper_runtime_statuses.get(paper.id) or {}).get("current_stage") or "reading_abstract")
    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status=error.recovery_status,
        current_stage=current_stage,
        result=result,
        error_message=str(error),
        deep_read_reserved=_infer_deep_read_reserved(result) if result is not None else bool((paper_runtime_statuses.get(paper.id) or {}).get("deep_read_reserved")),
    )


def _paper_has_reserved_deep_read(paper_runtime_statuses: dict[str, JsonObject], paper: PaperDocument) -> bool:
    """判断这篇论文是否已经占用了全文精读名额，避免恢复时重复扣减。"""

    payload = paper_runtime_statuses.get(paper.id) or {}
    if bool(payload.get("deep_read_reserved")):
        return True
    result = _paper_result_from_runtime_status(payload, paper)
    return _infer_deep_read_reserved(result)


def _infer_deep_read_reserved(result: PaperReadResult | None) -> bool:
    """只要论文已经进入全文链路，就视为这个名额已经被占用。"""

    if result is None:
        return False
    full_text = result.full_text
    return bool(full_text.source_path or full_text.markdown_path or full_text.status in _FULL_TEXT_COMPLETED_STATUSES | {"downloaded", "markdown_ready", "download_failed", "parse_failed"})


def _build_resume_checkpoint(
    state: State,
    *,
    papers: list[PaperDocument],
    results: list[PaperReadResult],
    artifact_refs: list[JsonObject],
    position: int,
    error: ReadResourceUnavailableError,
    deep_read_count: int,
    deep_read_limit: int,
    paper_runtime_statuses: dict[str, JsonObject],
) -> JsonObject:
    """新的恢复现场按论文保存状态，不再主要依赖串行位置。"""

    request = state.get("request")
    checkpoint = {
        "recovery_status": error.recovery_status,
        "current_step": error.current_step,
        "message": str(error),
        "error_details": dict(error.details),
        "next_position": position,
        "completed_count": len(results),
        "total_count": len(papers),
        "deep_read_count": deep_read_count,
        "deep_read_limit": deep_read_limit,
        "request": {
            "topic": getattr(request, "topic", ""),
            "constraints": dict(getattr(request, "constraints", {}) or {}),
            "language": getattr(request, "language", "zh"),
        },
        "search_results": [paper.to_dict() for paper in papers],
        "read_results": [result.to_dict() for result in results],
        "read_artifact_refs": list(artifact_refs),
        "paper_runtime_statuses": _ordered_paper_runtime_statuses(papers, paper_runtime_statuses),
        "resume_hint": "外部资源验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可继续未完成的论文。",
    }
    if isinstance(error, ReadModelUnavailableError):
        checkpoint["resume_hint"] = "阅读模型验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可继续未完成的论文。"
    return checkpoint

def _report_progress(reporter: Any, paper: PaperDocument, stage: str, completed: int, total: int, **extra: Any) -> None:
    """按论文维度上报实时进度，让同一篇论文始终更新同一张卡片。"""

    if reporter is None:
        return
    # 中文注释：这里的 key 一定要稳定。同一篇论文的摘要、下载、转换、保存等阶段
    # 都使用同一个 event_key，前端收到后就会更新旧卡片，而不是新增一堆零散卡片。
    messages = {
        "reading_abstract": "正在阅读论文摘要",
        "downloading_full_text": "正在下载论文全文",
        "converting_markdown": "正在转换 Markdown",
        "chunking_full_text": "正在切分全文内容",
        "saving_chunks": "正在写入全文索引",
        "paper_completed": "论文阅读完成",
        "paper_artifact_ready": "单篇阅读结果已保存",
    }
    runtime_status = str(extra.pop("runtime_status", "running") or "running")
    message = str(extra.pop("message", messages.get(stage, "正在处理论文")))
    reporter.progress(
        message,
        stage=stage,
        event_key=f"paper:{paper.id}",
        stage_title=paper.title,
        runtime_status=runtime_status,
        show_content=message,
        paper_id=paper.id,
        paper_title=paper.title,
        current_title=paper.title,
        completed=completed,
        total=total,
        **extra,
    )
