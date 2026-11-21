from __future__ import annotations

import asyncio

from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import State


def run_compose_reply_node():
    """发布已核查的研究报告，不再调用模型或旧综述流水线。"""
    async def node(state: State) -> State:
        runtime = state.get("runtime_context")
        reporter = (runtime.sync_port.for_node("compose_reply", "回复整理")
                    if isinstance(runtime, WorkflowRuntimeContext) and runtime.sync_port else None)
        markdown = str(state.get("research_markdown") or "")
        refs = list(state.get("final_artifact_refs") or [])
        if reporter:
            reporter.started("正在整理研究报告", stage="compose_start")
        repo = state.get("session_repo")
        if markdown and repo is not None:
            turn_id = str(state.get("turn_id") or "")
            record = await asyncio.to_thread(
                repo.write_artifact, state["session_key"], "final_report", "research_report.md", markdown,
                relative_path=f"artifacts/final_report/{turn_id}/research_report.md",
                metadata={"turn_id": turn_id, "format": "markdown", "artifact_version": "1.0"},
            )
            ref = {"artifact_id": str(record["id"]), **{key: record[key] for key in
                   ("artifact_type", "name", "path", "size", "created_at", "metadata")}}
            refs.append(ref)
            if reporter:
                reporter.artifact(ref, stage="final_artifact_ready")
        content = markdown or "未生成研究报告，请检查论文来源和研究阶段的执行记录。"
        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics["compose_reply"] = {"status": "ok" if markdown else "empty",
                                       "final_markdown": bool(markdown), "final_artifact_count": len(refs)}
        metadata = {key: state.get(key) for key in ("research_report", "research_artifact_refs",
                    "search_summary", "search_artifact_refs", "read_summary", "read_artifact_refs")}
        metadata.update(diagnostics=diagnostics, final_artifact_refs=refs, final_markdown=markdown)
        if reporter:
            reporter.message(role="assistant", content=content, metadata=metadata, stage="compose_reply")
            reporter.completed("研究报告整理完成", stage="compose_done", final_artifact_count=len(refs))
        return State(assistant_message=content, assistant_message_metadata=metadata,
                     final_artifact_refs=refs, diagnostics=diagnostics, current_step="reply")
    return node
