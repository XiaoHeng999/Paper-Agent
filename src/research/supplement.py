"""Bounded, explicitly enabled external evidence acquisition (not an extra agent)."""
import asyncio
import time
import re

from src.llm import SystemConfig
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.download import async_download_paper_fulltext
from src.utils.read_utils.read_fulltext import async_convert_fulltext_to_markdown
from src.utils.read_utils.chunkers import async_build_chunks_file


def source_keys(paper):
    keys = set()
    for field in ('paperId', 'id', 'doi', 'url', 'pdf_url'):
        value = str(paper.get(field) or '').strip().lower()
        if not value:
            continue
        match = re.search(r'(?:^|arxiv:|arxiv.org/(?:abs|pdf)/)([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?(?:$|[?#.]|/)', value)
        keys.add('arxiv:' + match.group(1) if match else value.removeprefix('https://doi.org/'))
    return keys


async def supplement_sources(state, items, tasks):
    options = tasks.options
    previous = (state.get("read_resume_checkpoint") or {}).get("external_search")
    if previous and previous.get("attempted"):
        return items, previous
    audit = {"attempted": False, "status": "disabled", "added_paper_ids": [], "warnings": []}
    if not options.allow_external_search or not items:
        return items, audit
    if getattr(tasks, 'terminal_error', None):
        audit.update(status='blocked', warnings=[tasks.terminal_error])
        return items, audit
    if state.get('research_planning_failed') and not state.get('research_challenges'):
        audit.update(status='planning_unavailable', warnings=['论证规划未完成，无法判断是否需要外部补证'])
        return items, audit
    if tasks.calls >= options.max_calls:
        audit.update(status="budget_exhausted", warnings=["研究调用预算已用尽，未继续下载无法分析的新论文"] )
        return items, audit
    if tasks.cancellation:
        tasks.cancellation.raise_if_requested()
    request = state["request"]
    challenges = state.get('research_challenges', [])[:3]
    if not challenges:
        audit.update(status='not_needed', warnings=['没有会改变当前结论的明确证据缺口，未启动外部补搜'])
        return items, audit
    query = " ".join(c.get("search_query") or (c["statement"] + " " + c["required_test"]) for c in challenges)[:1800]
    audit['challenged_claim_ids'] = [c['claim_id'] for c in challenges]
    audit.update(attempted=True, status="searching", query=query, questions=challenges, stopping_rule="每个缺口定向查询一次，达到论文数/时间预算即停止；未解决缺口保留为未知")
    runtime = state.get("runtime_context")
    resources = getattr(runtime, "resources", None)
    seen = {str(i.get("paper", {}).get("paperId") or i.get("paper", {}).get("id")) for i in items}
    seen_sources = set().union(*(source_keys(i.get('paper', {})) for i in items))
    added = []
    service = state.get("research_search_service") or PaperSearchService()
    config = SystemConfig.load().read
    try:
        async with asyncio.timeout(max(0, min(60, tasks.deadline - time.monotonic()))):
            candidates = []
            queries = []
            for challenge in challenges:
                targeted_query = (challenge.get('search_query') or (challenge['statement'] + ' ' + challenge['required_test']))[:1800]
                queries.append(targeted_query)
                response = await service.async_search(query=targeted_query, limit=min(10, options.external_paper_limit + len(seen)),
                sources=request.constraints.get("sources"), year_from=request.constraints.get("year_from"),
                year_to=request.constraints.get("year_to"), excluded_terms=request.constraints.get("excluded_terms"),
                runtime_resources=resources)
                audit["warnings"].extend(str(x) for x in response.errors.values())
                candidates.extend((paper, challenge) for paper in response.papers)
            audit['queries'] = queries
            for paper, challenge in candidates:
                if tasks.cancellation:
                    tasks.cancellation.raise_if_requested()
                paper_id = paper.paperId or paper.id
                candidate_keys = source_keys(paper.to_dict())
                existing = next((i for i in added if source_keys(i['paper']) & candidate_keys), None)
                if existing is not None:
                    pid = challenge.get('paper_id')
                    if pid and pid not in existing['support_for_paper_ids']:
                        existing['support_for_paper_ids'].append(pid)
                if candidate_keys & seen_sources or paper_id in seen or len(added) >= options.external_paper_limit:
                    continue
                seen.add(paper_id)
                seen_sources.update(candidate_keys)
                value = {"paper": paper.to_dict(), "full_text": {}, "source_role": "external_candidate", "support_for_paper_ids": [challenge["paper_id"]] if challenge.get("paper_id") else [], "acquisition_question": challenge}
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
