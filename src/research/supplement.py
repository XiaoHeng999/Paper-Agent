"""Bounded, explicitly enabled external evidence acquisition (not an extra agent)."""
import asyncio
import time

from src.llm import SystemConfig
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.download import async_download_paper_fulltext
from src.utils.read_utils.read_fulltext import async_convert_fulltext_to_markdown
from src.utils.read_utils.chunkers import async_build_chunks_file


async def supplement_sources(state, items, tasks):
    options = tasks.options
    previous = (state.get("read_resume_checkpoint") or {}).get("external_search")
    if previous and previous.get("attempted"):
        return items, previous
    audit = {"attempted": False, "status": "disabled", "added_paper_ids": [], "warnings": []}
    if not (options.cross_paper_enabled and options.allow_external_search) or not items:
        return items, audit
    if tasks.calls >= options.max_calls:
        audit.update(status="budget_exhausted", warnings=["研究调用预算已用尽，未继续下载无法分析的新论文"] )
        return items, audit
    if tasks.cancellation:
        tasks.cancellation.raise_if_requested()
    request = state["request"]
    challenges = state.get('research_challenges', [])[:3]
    focus = " ".join(f"{c['statement']} {c['premise']} {c['required_test']}" for c in challenges)
    query = (request.topic + " " + focus)[:1800] + " limitations failure cases ablation"
    audit['challenged_claim_ids'] = [c['claim_id'] for c in challenges]
    audit.update(attempted=True, status="searching", query=query)
    runtime = state.get("runtime_context")
    resources = getattr(runtime, "resources", None)
    seen = {str(i.get("paper", {}).get("paperId") or i.get("paper", {}).get("id")) for i in items}
    added = []
    service = state.get("research_search_service") or PaperSearchService()
    config = SystemConfig.load().read
    try:
        async with asyncio.timeout(max(0, min(60, tasks.deadline - time.monotonic()))):
            response = await service.async_search(query=query, limit=min(10, options.external_paper_limit + len(seen)),
                sources=request.constraints.get("sources"), year_from=request.constraints.get("year_from"),
                year_to=request.constraints.get("year_to"), excluded_terms=request.constraints.get("excluded_terms"),
                runtime_resources=resources)
            audit["warnings"].extend(str(x) for x in response.errors.values())
            for paper in response.papers:
                if tasks.cancellation:
                    tasks.cancellation.raise_if_requested()
                paper_id = paper.paperId or paper.id
                if paper_id in seen or len(added) >= options.external_paper_limit:
                    continue
                seen.add(paper_id)
                value = {"paper": paper.to_dict(), "full_text": {}, "source_role": "external_candidate"}
                added.append(value)  # Keep a discovered candidate even if its download times out.
                audit["added_paper_ids"].append(paper_id)
                downloaded = await async_download_paper_fulltext(paper, cache_dir=config.paper_cache_dir,
                    connect_timeout_seconds=config.connect_timeout_seconds, download_timeout_seconds=config.download_timeout_seconds,
                    max_file_size_mb=config.max_file_size_mb, runtime_resources=resources)
                if downloaded.file_path:
                    value["full_text"] = {"source_path": str(downloaded.file_path), "source_url": downloaded.source_url}
                    converted = await async_convert_fulltext_to_markdown(paper, source_path=downloaded.file_path, source_url=downloaded.source_url)
                    if converted.markdown_path:
                        chunks = await async_build_chunks_file(paper, markdown_path=converted.markdown_path)
                        value["full_text"] = {"status": "chunks_saved", "source_path": str(downloaded.file_path),
                            "markdown_path": str(converted.markdown_path), "source_url": downloaded.source_url, "chunk_count": len(chunks.chunks)}
                    else:
                        audit["warnings"].extend(converted.warnings)
                else:
                    audit["warnings"].append(f"{paper_id} 全文不可用，仅保留可得摘要")
            audit["status"] = "completed"
    except Exception as exc:
        audit["status"] = "partial" if added else "unavailable"
        audit["warnings"].append(f"外部补搜未完成：{type(exc).__name__}")
    return [*items, *added], audit
