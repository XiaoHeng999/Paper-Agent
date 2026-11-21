"""Bounded, explicitly enabled external evidence acquisition (not an extra agent)."""
import asyncio
import time
import re
from itertools import zip_longest

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
    audit = {"attempted": False, "status": "disabled", "added_paper_ids": [], "warnings": []}
    if not options.allow_external_search or not items:
        return items, audit
    challenges = state.get('research_challenges', [])[:3]
    request = state['request']
    policy = {k: request.constraints.get(k) for k in ('sources', 'year_from', 'year_to', 'excluded_terms')}
    policy['external_paper_limit'] = options.external_paper_limit
    policy['questions'] = [[q.get('gap_id') or q['claim_id'], q.get('search_query'), q.get('required_test')]
                           for q in challenges]
    audit['search_policy'] = policy
    present = {str(i['paper'].get('paperId') or i['paper'].get('id')) for i in items}
    if not (previous and previous.get('search_policy') == policy and set(previous.get('added_paper_ids', [])) <= present):
        previous = {}
    if previous.get('status') == 'completed':
        return items, previous
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
    if not challenges and state.get('unresolved_local_gaps'):
        audit.update(status='unresolved', warnings=['本文核查未能授权定向外查，缺口保留为未决'])
        return items, audit
    if not challenges:
        audit.update(status='not_needed', warnings=['没有会改变当前结论的明确证据缺口，未启动外部补搜'])
        return items, audit
    query = " ".join(c.get("search_query") or (c["statement"] + " " + c["required_test"]) for c in challenges)[:1800]
    audit['challenged_claim_ids'] = [c['claim_id'] for c in challenges]
    audit.update(attempted=True, status="searching", query=query, questions=challenges, stopping_rule="每个缺口定向查询一次，达到论文数/时间预算即停止；未解决缺口保留为未知")
    runtime = state.get("runtime_context")
    resources = getattr(runtime, "resources", None)
    seen = present
    seen_sources = set().union(*(source_keys(i.get('paper', {})) for i in items))
    added = []
    downloads = []
    service = state.get("research_search_service") or PaperSearchService()
    config = SystemConfig.load().read
    groups = {}
    for challenge in challenges:
        query = (challenge.get('search_query') or (challenge['statement'] + ' ' + challenge['required_test']))[:1800]
        groups.setdefault(' '.join(query.split()), []).append(challenge)
    audit['completed_queries'] = list(previous.get('completed_queries', []))
    audit['added_paper_ids'] = list(previous.get('added_paper_ids', []))
    audit['candidates_by_gap'] = {gid: list(pids) for gid, pids in previous.get('candidates_by_gap', {}).items()}
    audit['queries'] = []
    audit['fulltext_warnings'] = list(previous.get('fulltext_warnings', []))
    audit['warnings'].extend(audit['fulltext_warnings'])
    ranked = []
    failed_query = False
    deadline = min(tasks.deadline, time.monotonic() + 60)
    for query, matching_questions in groups.items():
        if query in audit['completed_queries']:
            continue
        if tasks.cancellation:
            tasks.cancellation.raise_if_requested()
        if time.monotonic() >= deadline:
            audit['warnings'].append('外查时间用尽，剩余问题待检索')
            failed_query = True
            break
        audit['queries'].append(query)
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                response = await service.async_search(query=query, limit=min(10, options.external_paper_limit + len(seen)),
                    sources=request.constraints.get("sources"), year_from=request.constraints.get("year_from"),
                    year_to=request.constraints.get("year_to"), excluded_terms=request.constraints.get("excluded_terms"),
                    runtime_resources=resources)
            audit['warnings'].extend(str(x) for x in response.errors.values())
            if response.errors:
                failed_query = True
            else:
                audit['completed_queries'].append(query)
            ranked.append([(paper, matching_questions) for paper in response.papers])
        except Exception as exc:
            failed_query = True
            audit['warnings'].append(f'定向检索失败：{type(exc).__name__}，问题：{query}')

    # Give each question a candidate before spending the remaining slots on lower ranks.
    existing = {key: i for i in items if i.get('source_role') == 'external_candidate' for key in source_keys(i['paper'])}
    for same_rank in zip_longest(*ranked):
        for candidate in same_rank:
            if candidate is None:
                continue
            paper, matching_questions = candidate
            paper_id = paper.paperId or paper.id
            keys = source_keys(paper.to_dict())
            value = next((existing[k] for k in keys if k in existing), None)
            if value is None:
                if keys & seen_sources or paper_id in seen or len(audit['added_paper_ids']) >= options.external_paper_limit:
                    continue
                seen.add(paper_id)
                seen_sources.update(keys)
                value = {'paper': paper.to_dict(), 'full_text': {}, 'source_role': 'external_candidate',
                         'acquisition_question': dict(matching_questions[0])}
                existing.update(dict.fromkeys(keys, value))
                added.append(value)
                downloads.append((paper, value))
            canonical = value['paper'].get('paperId') or value['paper']['id']
            if canonical not in audit['added_paper_ids']:
                if len(audit['added_paper_ids']) >= options.external_paper_limit:
                    continue
                audit['added_paper_ids'].append(canonical)
            for q in matching_questions:
                if q.get('gap_id'):
                    matches = audit['candidates_by_gap'].setdefault(q['gap_id'], [])
                    if canonical not in matches:
                        matches.append(canonical)
            for field, key in [('support_for_paper_ids', 'paper_id'), ('acquisition_gap_ids', 'gap_id')]:
                value[field] = list(dict.fromkeys([*value.get(field, []), *(q[key] for q in matching_questions if q.get(key))]))

    try:
        async with asyncio.timeout(max(0, deadline - time.monotonic())):
            for paper, value in downloads:
                if tasks.cancellation:
                    tasks.cancellation.raise_if_requested()
                paper_id = paper.paperId or paper.id
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
                        audit["fulltext_warnings"].extend(converted.warnings)
                else:
                    audit["fulltext_warnings"].append(f"{paper_id} 全文不可用，仅保留可得摘要")
            audit["status"] = ("partial" if audit['added_paper_ids'] else "unavailable") if failed_query else "completed"
    except Exception as exc:
        audit["status"] = "partial" if added else "unavailable"
        audit["fulltext_warnings"].append(f"外部全文读取未完成：{type(exc).__name__}")
    audit["warnings"] = list(dict.fromkeys([*audit["warnings"], *audit["fulltext_warnings"]]))
    return [*items, *added], audit
