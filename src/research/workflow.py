from __future__ import annotations

import asyncio
import json
import time

from src.agents.readAgent import load_read_agent_llm
from .agents import ModelTasks, compare, read_claims, verify, write_report
from .context import ScopedEvidence
from .assessment import assess_card, comparison_views
from .models import Claim, ClaimDraft, PaperCard, ResearchOptions, version_of
from .retrieval import EvidenceIndex
from .storage import ResearchStore, load_evidence
from .supplement import supplement_sources
from .visual import prepare_visual
from .provenance import document_version, source_manifest
from .scheduling import comparison_schedule
from .claims import merge_duplicate
from .reasoning import plan_arguments, research_questions, check_local_gap, synthesize, render_reasoning, review_core_question, render_core_review
from .incremental import IncrementalPlan, policy_version, dependency_manifest


VERDICTS = {"verified": "当前证据支持", "qualified": "需要限定条件", "insufficient_evidence": "证据不足",
            "contradicted": "与证据不符", "needs_human_review": "需要人工复核"}
KINDS = {"question": "研究问题", "method": "方法", "result": "结果", "premise": "前提", "limitation": "局限"}
RELATIONS = {"equivalent": "同质", "extends": "扩展", "supports": "支持", "conflicts": "冲突", "different_scope": "条件不同", "incomparable": "无法比较"}


def render_card(card: PaperCard, spans: ScopedEvidence, compact: bool = False) -> str:
    lines = [f"# {card.title}", f"论文 ID：{card.paper_id}", "核查只针对所列来源，不代表外部复现或领域新颖性验证。"]
    lines.extend(f"> {warning}" for warning in card.warnings)
    for claim in card.claims:
        result = next((v for v in reversed(card.verifications) if v.target_id == claim.id and v.target_version == claim.version), None)
        lines.extend([f"## {KINDS[claim.kind]}", claim.statement,
            f"前提与适用范围：{claim.conditions}", f"实验条件：{claim.experiment}",
            "判断来源：" + ("作者报告" if claim.attribution == "author_report" else "系统推断"),
            f"核查：{VERDICTS[result.verdict] if result else '未核查'}"])
        details = {key: value for key, value in claim.study.model_dump().items() if value != "unknown"}
        if details:
            lines.append("结构化研究条件：" + "；".join(f"{key}={value}" for key, value in details.items()))
        if result:
            lines.append(result.reason)
            for label, check in (("忠实转述", result.fidelity), ("实验支持充分性", result.experimental_support), ("内部反证", result.counterevidence)):
                lines.append(f"{label}：{check.status}；{check.reason}")
            for observation in result.visual_observations:
                lines.append(f"视觉观察：第 {observation.page} 页，{observation.locator}；{observation.observation}；值={observation.value}，单位={observation.unit}，条件={observation.conditions}；{observation.status}")
            for challenge in result.challenges:
                lines.append(f"待验证挑战（并非已证实反证）：前提={challenge.premise}；替代解释={challenge.alternative_explanation}；需要检查={challenge.required_test}")
            if result.repair_request:
                lines.append("待修订：" + result.repair_request)
            if result.unchecked:
                lines.append("未核查项：" + "；".join(result.unchecked))
        cited_ids = list(claim.evidence_ids)
        if result:
            cited_ids.extend(result.evidence_ids)
            for check in (result.fidelity, result.experimental_support, result.counterevidence):
                cited_ids.extend(check.evidence_ids)
            for dimension in result.dimensions:
                cited_ids.extend(dimension.evidence_ids)
        for span in spans.read(cited_ids):
            location = f"第 {span.page_start} 页" if span.page_start else span.section
            lines.append(f"来源 [{span.id}]（{'全文' if span.source_level == 'full_text' else '摘要'}，{location}）")
            if not compact:
                lines.append("\n".join("> " + line for line in span.text.splitlines()))
    if not card.claims:
        lines.append("未生成有来源支撑的判断，请检查论文可读性、模型配置及任务预算。")
    if card.omitted_evidence_ids:
        lines.append(f"尚未阅读 {len(card.omitted_evidence_ids)} 个证据片段；完整列表保存在研究产物中。")
    return "\n\n".join(lines)


def run_research_node():
    async def node(state):
        started = time.monotonic()
        core_reviews = []
        request = state["request"]
        options = ResearchOptions.from_constraints(request.constraints)
        runtime = state.get("runtime_context")
        port = getattr(runtime, "sync_port", None)
        reporter = port.for_node("research", "论文判断与证据核查") if port else None
        if reporter:
            reporter.started("正在逐篇形成判断并核查来源")
        llm = state["research_node_llm"] if "research_node_llm" in state else load_read_agent_llm()
        tasks = ModelTasks(llm, options, getattr(runtime, "cancellation", None),
            lambda role, usage: reporter.progress("模型任务已完成", role=role, **usage) if reporter else None)
        store = ResearchStore(state, reporter)
        items = list(state.get("read_results") or [])
        primary_ids = {str(i.get('paper', {}).get('paperId') or i.get('paper', {}).get('id')) for i in items if i.get('source_role') != 'external_candidate'}
        tasks.reading_call_reserve = min(3 * len(primary_ids), options.max_calls // 4) if options.reasoning_enabled else 0
        tasks.reading_time_reserve = min(120, options.timeout_seconds // 4) if options.reasoning_enabled and options.timeout_seconds > 60 else 0
        external_search = (state.get("read_resume_checkpoint") or {}).get("external_search") or {"attempted": False, "status": "disabled", "added_paper_ids": [], "warnings": []}
        revision = (state.get("read_resume_checkpoint") or {}).get("revision")
        parent_report = None
        if revision and store.repo and store.session:
            parent_path = await asyncio.to_thread(store.repo.read_artifact_path, store.session, revision["parent_artifact_id"])
            if parent_path:
                parent_report = json.loads(await asyncio.to_thread(parent_path.read_text, encoding="utf-8"))
        checkpoint = {
            "revision": revision,
            "phase": "research", "request": {"topic": request.topic, "constraints": request.constraints, "language": request.language},
            "search_results": [p.to_dict() for p in state.get("search_results", [])],
            "external_search": external_search,
            "read_results": items, "read_artifact_refs": state.get("read_artifact_refs", []),
        }
        state["read_resume_checkpoint"] = checkpoint
        await store.save("research_checkpoint", checkpoint)
        if reporter:
            reporter.progress("研究恢复现场已保存", checkpoint=checkpoint)
        retrieval = state.get("research_evidence_index")
        all_evidence = []
        inputs = []
        input_warnings = {}
        if parent_report and parent_report.get("visual_model") == getattr(llm, "signature", None):
            tasks.previous_visual_evidence = parent_report.get("evidence", [])
        tasks.question = request.topic
        tasks.remaining_visual_papers = sum(str(i.get("full_text", {}).get("source_path", "")).lower().endswith(".pdf") for i in items)
        visual_audits = []
        source_manifests = {}
        seen = set()
        async def prepare_input(item):
            paper = item.get("paper", {})
            paper_id = str(paper.get("paperId") or paper.get("id") or "")
            if not paper_id or paper_id in seen or item.get("relevance", {}).get("discarded"):
                return
            seen.add(paper_id)
            before_load = await asyncio.to_thread(document_version, item)
            try:
                evidence = await asyncio.to_thread(load_evidence, item)
            except (OSError, ValueError) as exc:
                evidence = []
                input_warnings[paper_id] = f"原文准备失败，未生成判断：{exc}"
            document_before = await asyncio.to_thread(document_version, item)
            evidence, visual_audit = await prepare_visual(tasks, item, evidence)
            if str(item.get("full_text", {}).get("source_path", "")).lower().endswith(".pdf"):
                tasks.remaining_visual_papers = max(0, tasks.remaining_visual_papers - 1)
            document_after = await asyncio.to_thread(document_version, item)
            if document_before != document_after or before_load["source"] != document_after["source"]:
                raise ValueError("读取期间原文发生变化，请基于稳定来源重新执行")
            source_manifests[paper_id] = source_manifest(document_after, evidence, visual_audit)
            visual_audits.append(visual_audit)
            all_evidence.extend(evidence)
            inputs.append((paper_id, paper.get("title") or paper_id, evidence))
        for item in items:
            await prepare_input(item)
        if revision:
            old = Claim.model_validate(revision["target"])
            seed = PaperCard.model_validate(revision["seed_card"])
            current_sources = {e.id: e.version for e in all_evidence if e.paper_id == old.paper_id}
            expected_document = revision.get("source_document")
            if (expected_document is not None and source_manifests[old.paper_id]["document"] != expected_document) or current_sources != revision["evidence_versions"]:
                raise ValueError("原文版本已变化，修订未执行；请基于新来源重新分析")
            if seed.paper_id != old.paper_id or not any(c.id == old.id and c.version == old.version for c in seed.claims):
                raise ValueError("修订目标版本与原始判断不一致")
        policy = policy_version(tasks, request.topic, retrieval)
        plan = IncrementalPlan(parent_report, all_evidence, policy, revision, source_manifests)
        semaphore = asyncio.Semaphore(3)

        async def process(paper_id, title, evidence):
            async with semaphore:
                reused_card = plan.card(paper_id)
                scope = ScopedEvidence(evidence, {paper_id}, retrieval)
                if retrieval and evidence:
                    try:
                        remaining = tasks.deadline - time.monotonic()
                        if tasks.cancellation:
                            tasks.cancellation.raise_if_requested()
                        async with asyncio.timeout(max(0, min(30, remaining))):
                            await retrieval.prepare(evidence)
                        if reporter:
                            reporter.progress(f"《{title}》的证据向量索引已就绪", paper_id=paper_id, stage="evidence_index_ready")
                    except Exception as exc:
                        retrieval.warnings.append(f"向量入库失败，继续使用原文：{type(exc).__name__}")
                if reused_card is not None:
                    return reused_card
                revised_claims = None
                paper_revision = revision if revision and revision["target"]["paper_id"] == paper_id else None
                if paper_revision:
                    seed = PaperCard.model_validate(paper_revision["seed_card"])
                    old = Claim.model_validate(paper_revision["target"])
                    if not any(c.id == old.id and c.version == old.version for c in seed.claims):
                        raise ValueError("修订目标版本与原始判断不一致")
                    draft = ClaimDraft.model_validate(paper_revision["replacement"])
                    cited = scope.read(draft.evidence_ids)
                    new_version = version_of({"claim": draft.model_dump(), "sources": [(e.id, e.version) for e in cited], "human_revision": old.version})
                    replacement = Claim(**draft.model_dump(), id=f"{paper_id}:claim:{new_version}", paper_id=paper_id, version=new_version)
                    revised_claims = [replacement if c.id == old.id else c for c in seed.claims]
                cache_key = version_of({"schema": "paper-card-v6", "source_manifest": source_manifests[paper_id], "policy": policy, "revision": paper_revision, "retrieval": retrieval.collection if retrieval else None, "retrieval_ready": bool(retrieval and retrieval.key(evidence) in retrieval.ready), "question": request.topic,
                    "paper": paper_id, "title": title, "evidence": [e.model_dump() for e in evidence],
                    "model": getattr(llm, "signature", "unknown"),
                    "context_chars": options.context_chars, "repair_rounds": options.max_repair_rounds})
                cached = await store.load(cache_key)
                if cached:
                    try:
                        card = PaperCard.model_validate(cached["card"])
                        if card.paper_id == paper_id and card.version == cache_key and not card.omitted_evidence_ids and card.claims and all(
                            v.verdict == "verified" for v in card.verifications
                        ) and {(v.target_id, v.target_version) for v in card.verifications} == {(c.id, c.version) for c in card.claims}:
                            for claim in card.claims:
                                scope.read(claim.evidence_ids)
                            return card
                    except (ValueError, KeyError):
                        pass
                card = PaperCard(paper_id=paper_id, title=title, version=cache_key)
                for audit in visual_audits:
                    if audit["paper_id"] == paper_id:
                        card.warnings.extend(audit["warnings"])
                        unread = list(audit["omitted_pages"]) + [page["page"] for page in audit["pages"] if page.get("error")]
                        card.omitted_evidence_ids.extend(f"{paper_id}:unread-visual-page:{page}" for page in unread)
                if paper_id in input_warnings:
                    card.warnings.append(input_warnings[paper_id])
                if not evidence:
                    card.warnings.append("没有可读取的全文或摘要，未进行语义核查")
                elif any(e.source_level == "abstract" for e in evidence):
                    card.warnings.append("仅取得摘要；无法据此确认全文实验细节")
                history = []
                priority = await scope.search(request.topic, tasks)
                largest_span = max((len(e.text) for e in evidence), default=0)
                read_budget = min(options.context_chars, max(options.context_chars * 3 // 4, largest_span))
                focus_pending = options.reasoning_enabled and paper_id in primary_ids and revised_claims is None
                priority_ids = [s.id for s in priority]
                if focus_pending:
                    # Include an experimental result alongside the opening and retrieved passages.
                    experimental = next((s.id for s in evidence if any(term in s.section.casefold()
                        for term in ("result", "experiment", "evaluation", "实验", "结果"))), None)
                    priority_ids = list(dict.fromkeys([*priority_ids[:1], *(s.id for s in evidence[:1]),
                        *([experimental] if experimental else []), *priority_ids[1:]]))
                batches = [None] if revised_claims is not None else scope.batches(read_budget, priority_ids)
                for batch in batches:
                    ids = [x.id for x in batch] if batch is not None else list(scope.spans)
                    questions = {}
                    try:
                        claims = revised_claims if revised_claims is not None else await read_claims(tasks, request.topic, paper_id, scope, ids, core_questions=questions if focus_pending else None)
                    except (ValueError, RuntimeError, TimeoutError) as exc:
                        card.warnings.append(str(exc))
                        card.omitted_evidence_ids.extend(ids)
                        continue
                    if questions:
                        focus_pending = False
                        by_id = {c.id: c for c in claims}
                        claims = [*(by_id[cid] for cid in questions), *(c for c in claims if c.id not in questions)]
                    for claim in claims:
                        core_question = questions.get(claim.id)
                        for previous in list(card.claims):
                            merged = merge_duplicate(previous, claim, scope)
                            if merged is not None:
                                claim = merged
                                card.claims.remove(previous)
                                card.verifications = [v for v in card.verifications if v.target_id != previous.id]
                                break
                        previous_check = plan.check(claim) if revised_claims is not None else None
                        if previous_check is not None:
                            card.claims.append(claim)
                            card.verifications.append(previous_check)
                            continue
                        result = await verify(tasks, claim, scope)
                        history.append({"target": claim.model_dump(), "verification": result.model_dump()})
                        if core_question and not tasks.terminal_error:
                            snapshot, issues = await review_core_question(tasks, core_question, claim, result,
                                card, evidence, policy, source_manifests)
                            card.warnings.extend(issues)
                            if snapshot:
                                snapshot['elapsed_seconds'] = round(time.monotonic() - started, 3)
                                await store.save('core_argument', render_core_review(snapshot),
                                                 name=f"core_{version_of(claim.id)}.md",
                                                 metadata={"question": core_question, "paper_id": paper_id})
                                core_reviews.append(snapshot)
                                if reporter:
                                    reporter.progress('核心论证已完成独立审查：' + core_question,
                                        stage='core_argument_reviewed', paper_id=paper_id,
                                        elapsed_seconds=snapshot['elapsed_seconds'])
                        # Targeted reread preserves the failed version in the audit trail.
                        current = [(claim, result)]
                        for _ in range(options.max_repair_rounds):
                            if len(current) != 1 or current[0][1].verdict not in {"qualified", "contradicted"}:
                                break
                            old, verdict = current[0]
                            try:
                                raw, _ = await scope.expand(old.evidence_ids, verdict.repair_request, options.context_chars, tasks)
                                repaired = await read_claims(tasks, request.topic, paper_id, scope,
                                    [x.id for x in raw], feedback=f"原判断：{old.statement}\n核查反馈：{verdict.reason}\n修订要求：{verdict.repair_request}")
                                if len(repaired) != 1:
                                    card.warnings.append("定向修订未返回单一判断，保留原判断及核查状态")
                                    break
                                current = []
                                for new in repaired:
                                    checked = await verify(tasks, new, scope)
                                    history.append({"target": new.model_dump(), "verification": checked.model_dump()})
                                    current.append((new, checked))
                            except (ValueError, RuntimeError, TimeoutError) as exc:
                                card.warnings.append(str(exc))
                                break
                        for target, checked in current:
                            if target.id not in {x.id for x in card.claims}:
                                card.claims.append(target)
                                card.verifications.append(checked)
                card.omitted_evidence_ids = list(dict.fromkeys([*card.omitted_evidence_ids, *sorted(tasks.reading_gaps.get(paper_id, set()))]))
                await store.save("research_paper", {"card": card.model_dump(), "evidence": [e.model_dump() for e in evidence], "history": history}, cache_key=cache_key)
                if reporter:
                    reporter.progress(f"已完成《{title}》的逐篇阅读核查", paper_id=paper_id)
                return card

        pending = [asyncio.create_task(process(*value)) for value in inputs if value[0] in primary_ids or options.cross_paper_enabled]
        try:
            cards = await asyncio.gather(*pending)
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        # Search only after concrete paper claims and weaknesses are available.
        challenges = []
        for card in cards:
            by_id = {c.id: c for c in card.claims}
            for check in card.verifications:
                claim = by_id.get(check.target_id)
                if not claim or check.target_version != claim.version:
                    continue
                for challenge in check.challenges:
                    challenges.append({"claim_id": claim.id, "claim_version": claim.version,
                        "statement": claim.statement, **challenge.model_dump()})
                if not check.challenges and (claim.conditions == 'unknown' or check.experimental_support.status == 'unknown'):
                    challenges.append({"claim_id": claim.id, "claim_version": claim.version,
                        "statement": claim.statement, "premise": claim.conditions,
                        "required_test": "查找相同任务和实验条件下的独立比较、消融或反例", "alternative_explanation": "实验支持尚未确认", "evidence_ids": claim.evidence_ids})
        argument_plans = []
        if options.reasoning_enabled:
            for card in cards:
                if card.paper_id in primary_ids:
                    key = version_of({'card': card.model_dump(), 'source': source_manifests.get(card.paper_id), 'policy': policy})
                    previous_plan = next((p for p in (parent_report or {}).get('argument_plans', []) if p.get('dependency_version') == key and p.get('status') == 'planned'), None)
                    current_ids = {c.id for c in card.claims}
                    early_arguments = [s['argument'] for s in core_reviews
                        if s['paper_id'] == card.paper_id and s['claim']['id'] in current_ids]
                    if previous_plan:
                        argument_plan = dict(previous_plan)
                    elif early_arguments and {a['claim_id'] for a in early_arguments} == current_ids:
                        argument_plan = {'paper_id': card.paper_id, 'status': 'planned',
                                         'arguments': early_arguments, 'warnings': []}
                    else:
                        argument_plan = await plan_arguments(tasks, request.topic, card,
                            [e for e in all_evidence if e.paper_id == card.paper_id])
                    argument_plan['dependency_version'] = key
                    argument_plans.append(argument_plan)
        questions = research_questions(argument_plans, cards) if options.reasoning_enabled else challenges
        search_questions = questions
        if options.reasoning_enabled:
            previous_reviews = {q['gap_id']: q['local_review'] for p in argument_plans
                                for q in p.get('evidence_questions', []) if q.get('local_review')}
            for index, question in enumerate(questions):
                if not options.allow_external_search:
                    local = {'decision': 'disabled', 'reason': '未开启外部查阅', 'evidence_ids': []}
                elif index >= 3:
                    local = {'decision': 'deferred', 'reason': '已达到本轮缺口核查上限', 'evidence_ids': []}
                elif previous_reviews.get(question['gap_id'], {}).get('decision') in {'external_needed', 'answered_here'}:
                    local = previous_reviews[question['gap_id']]
                else:
                    scope = ScopedEvidence(all_evidence, {question['paper_id']}, retrieval)
                    local = await check_local_gap(tasks, question, scope)
                question['local_review'] = local
            search_questions = [q for q in questions if q['local_review']['decision'] == 'external_needed']
        acquisition_state = {**state, "research_challenges": search_questions,
            "unresolved_local_gaps": any(q.get('local_review', {}).get('decision') in {'unresolved','deferred'} for q in questions),
            "research_planning_failed": options.reasoning_enabled and any(p['status'] != 'planned' for p in argument_plans)}
        previous_count = len(inputs)
        previous_items = len(items)
        items, external_search = await supplement_sources(acquisition_state, items, tasks)
        tasks.remaining_visual_papers = sum(str(i.get('full_text', {}).get('source_path', '')).lower().endswith('.pdf') for i in items[previous_items:])
        for item in items[previous_items:]:
            await prepare_input(item)
        if options.cross_paper_enabled:
            for value in inputs[previous_count:]:
                cards.append(await process(*value))
        checkpoint.update(read_results=items, external_search=external_search)
        state['read_resume_checkpoint'] = checkpoint
        await store.save('research_checkpoint', checkpoint)
        plan.bind_cards(cards)
        usable = [card for card in cards if card.claims]
        cross_enabled = options.cross_paper_enabled and len(usable) >= 2
        warnings = list(external_search["warnings"])
        warnings.extend(w for audit in visual_audits for w in audit["warnings"])
        if options.output_mode == "comparison" and not cross_enabled:
            warnings.append("未启用跨篇分析或可用论文不足两篇，已输出逐篇独立报告。")
        if options.output_mode == "review" and not cross_enabled:
            warnings.append("跨篇综合未启用或资料不足，本次输出逐篇报告汇编。")
        relations, checks, relation_history = [], [], []
        pair_outcomes = []
        compared_pairs, omitted_pairs = 0, 0
        if cross_enabled:
            if reporter:
                reporter.progress("正在对齐主张并核查跨论文关系")
            writing_reserve = min(10, options.max_calls // 5) if options.output_mode == "review" else 0
            for left, right, first, second in comparison_schedule(usable):
                scope = ScopedEvidence(all_evidence, {left.paper_id, right.paper_id}, retrieval)
                reused_pair = plan.pair(first, second)
                if reused_pair is not None:
                    old_relations, old_checks = reused_pair
                    relations.extend(old_relations)
                    checks.extend(old_checks)
                    pair_outcomes.append({"claim_ids": [first.id, second.id], "status": "complete",
                        "relations": [r.model_dump() for r in old_relations], "verifications": [v.model_dump() for v in old_checks]})
                    compared_pairs += 1
                    continue
                if tasks.calls + 2 + writing_reserve > options.max_calls:
                    omitted_pairs += 1
                    continue
                compared_pairs += 1
                pair_start = len(relations)
                try:
                    proposed = await compare(tasks, request.topic, [first, second], scope)
                    for relation in proposed:
                        check = await verify(tasks, relation, scope, [first, second])
                        relation_history.append({"target": relation.model_dump(), "verification": check.model_dump()})
                        if check.verdict in {"qualified", "contradicted"} and options.max_repair_rounds:
                            repaired = await compare(tasks, request.topic, [first, second], scope, check.reason + "\n" + check.repair_request)
                            if repaired:
                                relation = repaired[0]
                                check = await verify(tasks, relation, scope, [first, second])
                                relation_history.append({"target": relation.model_dump(), "verification": check.model_dump()})
                        supporting_checks = [v for card in (left, right) for v in card.verifications
                            if (v.target_id, v.target_version) in {(first.id, first.version), (second.id, second.version)}]
                        if check.verdict == "verified" and (len(supporting_checks) != 2 or any(v.verdict != "verified" for v in supporting_checks)):
                            check = check.model_copy(update={"verdict": "insufficient_evidence", "reason": "关系涉及的单篇主张尚未全部通过当前版本核查。"})
                        relations.append(relation)
                        checks.append(check)
                    pair_outcomes.append({"claim_ids": [first.id, second.id], "status": "complete" if all("本次语义核查未完成" not in v.unchecked for v in checks[pair_start:]) else "incomplete",
                        "relations": [r.model_dump() for r in relations[pair_start:]], "verifications": [v.model_dump() for v in checks[pair_start:]]})
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    warnings.append(str(exc))
        reasoning = []
        if options.reasoning_enabled:
            external_ids = {str(i['paper'].get('paperId') or i['paper'].get('id')) for i in items if i.get('source_role') == 'external_candidate'}
            for argument_plan in argument_plans:
                card = next(c for c in cards if c.paper_id == argument_plan['paper_id'])
                paper_questions = [q for q in questions if q['paper_id'] == card.paper_id]
                linked = set()
                for q in paper_questions:
                    q['candidate_ids'] = sorted({str(i['paper'].get('paperId') or i['paper'].get('id'))
                        for i in items if i.get('source_role') == 'external_candidate'
                        and q['local_review']['decision'] == 'external_needed'
                        and q['gap_id'] in i.get('acquisition_gap_ids', [])})
                    linked.update(q['candidate_ids'])
                    q['acquisition_status'] = external_search['status'] if q['local_review']['decision'] == 'external_needed' else 'not_requested'
                argument_plan['evidence_questions'] = paper_questions
                reasoning.append(await synthesize(tasks, request.topic, card, argument_plan, all_evidence, external_ids & linked, policy,
                    previous=next((r for r in (parent_report or {}).get('reasoning', []) if r['paper_id'] == card.paper_id), None),
                    source_manifests=source_manifests))
            await store.save('research_reasoning', reasoning)
        assessments = [assess_card(card, options) for card in cards]
        matrix, relation_graph = comparison_views(cards, relations, checks)
        sections = []
        for card, assessment in zip(cards, assessments):
            text = render_card(card, ScopedEvidence(all_evidence, {card.paper_id}), options.output_mode == "paper_cards")
            text += "\n\n## 分维度评价（论文内部已审证据）\n\n| 维度 | 分值（0–4） | 理由 | 证据 |\n| --- | --- | --- | --- |\n"
            for dimension in assessment["dimensions"]:
                score = dimension["score"] if dimension["score"] is not None else "unknown"
                reason = dimension["reason"].replace("|", "\\|").replace("\n", " ")
                text += f"| {dimension['label']} | {score} | {reason} | {'、'.join(dimension['evidence_ids'])} |\n"
            text += "\n" + assessment["score_reason"] + "\n" + assessment["novelty_scope"]
            if assessment["weighted_score"] is not None:
                text += f"\n评分目标：{assessment['goal']}；加权分：{assessment['weighted_score']}"
            sections.append(text)
            per_paper_reasoning = [r for r in reasoning if r["paper_id"] == card.paper_id]
            current_ids = {c.id for c in card.claims}
            final_ids = {a['claim_id'] for r in per_paper_reasoning if r['status'] == 'reviewed'
                         for a in r['arguments']}
            partial = [render_core_review(s) for s in core_reviews if s['paper_id'] == card.paper_id
                       and s['claim']['id'] in current_ids - final_ids]
            if partial:
                text = "\n\n".join(["# 已独立审查的核心问题", *partial, text])
                sections[-1] = text
            await store.save("paper_report", (render_reasoning(per_paper_reasoning) + "\n\n" if per_paper_reasoning else "") + text, name=f"paper_{version_of(card.paper_id)}.md")
        if relations:
            lines = ["# 跨论文关系分析"]
            for relation, check in zip(relations, checks):
                lines.extend([f"## {RELATIONS[relation.relation]}（{VERDICTS[check.verdict]}）", relation.rationale,
                    "比较条件：" + relation.conditions, "主张：" + "、".join(relation.claim_ids),
                    "证据：" + "、".join(relation.evidence_ids), "核查理由：" + check.reason])
                for axis in relation.axes:
                    lines.append(f"比较轴 {axis.axis}：{axis.left} / {axis.right}（{axis.status}）")
                lines.extend("未解决：" + question for question in relation.unresolved_questions)
                if check.repair_request:
                    lines.append("待修订：" + check.repair_request)
            sections.insert(0, "\n\n".join(lines))
        paragraphs, writing_history = [], []
        if options.output_mode == "review" and cross_enabled:
            eligible = [claim for card in cards for claim in card.claims if any(
                v.target_id == claim.id and v.target_version == claim.version and v.verdict == "verified"
                for v in card.verifications)]
            # Split the writer input on whole claims, preserving their conditions and versions.
            batches, batch, size = [], [], 0
            retained = plan.paragraphs(eligible, relations, checks)
            covered = {cid for entry in retained for cid in entry["paragraph"]["claim_ids"]}
            # A lost paragraph may also cite unchanged claims: regenerate its complete input scope.
            invalidated_inputs = {cid for entry in (parent_report or {}).get("paragraphs", []) if entry not in retained
                for cid in entry["paragraph"]["claim_ids"]}
            for claim in eligible:
                if claim.id in covered and claim.id not in invalidated_inputs:
                    continue
                length = len(json.dumps(claim.model_dump(), ensure_ascii=False))
                if length > options.context_chars:
                    warnings.append(f"主张 {claim.id} 超出写作上下文预算，保留在逐篇报告中")
                    continue
                if batch and size + length > options.context_chars // 2:
                    batches.append(batch)
                    batch, size = [], 0
                batch.append(claim)
                size += length
            if batch:
                batches.append(batch)
            from .writing import approved_relations
            write_ids = {c.id for group in batches for c in group}
            eligible_lookup = {c.id: c for c in eligible}
            for relation in approved_relations(relations, checks):
                endpoints = set(relation.claim_ids)
                if not endpoints.intersection(write_ids) or not endpoints <= eligible_lookup.keys():
                    continue
                if not any(endpoints <= {c.id for c in group} for group in batches):
                    pair = [eligible_lookup[cid] for cid in relation.claim_ids]
                    if len(json.dumps([c.model_dump() for c in pair], ensure_ascii=False)) < options.context_chars:
                        batches.append(pair)
                    else:
                        warnings.append(f"关系 {relation.id} 的双方主张超过写作预算，保留在比较矩阵中")
            rejected_prose = ["# 未通过核查的写作草稿", "以下内容仅供审查，不作为综述结论。"]
            accepted_count = 0
            prose = ["# 综述正文", "各段根据列出的主张生成，核查范围及未解决问题随段落保留。"]
            for entry in retained:
                paragraphs.append(entry)
                checked = entry["verification"]
                paragraph = entry["paragraph"]
                destination = prose if checked["verdict"] == "verified" else rejected_prose
                accepted_count += int(checked["verdict"] == "verified")
                destination.extend([paragraph["statement"], "主张来源：" + "、".join(paragraph["claim_ids"]), "关系来源：" + "、".join(paragraph.get("relation_ids", [])),
                    f"核查：{VERDICTS[checked['verdict']]}；{checked['reason']}"])
            lookup = {claim.id: claim for claim in eligible}
            for batch in batches:
                try:
                    drafts = await write_report(tasks, request.topic, batch, relations=relations, relation_checks=checks)
                    for paragraph in drafts:
                        sources = [lookup[cid] for cid in paragraph.claim_ids]
                        scope = ScopedEvidence(all_evidence, {x.paper_id for x in sources}, retrieval)
                        checked = await verify(tasks, paragraph, scope, sources, relations=[r for r in relations if r.id in paragraph.relation_ids])
                        writing_history.append({"paragraph": paragraph.model_dump(), "verification": checked.model_dump()})
                        for _ in range(options.max_repair_rounds):
                            if checked.verdict not in {"qualified", "contradicted"}:
                                break
                            revised = await write_report(tasks, request.topic, sources,
                                f"原段落：{paragraph.statement}\n{checked.reason}\n{checked.repair_request}", relations=relations, relation_checks=checks)
                            if len(revised) != 1:
                                warnings.append("正文定向修订未返回单一段落，保留原段落及未通过状态")
                                break
                            paragraph = revised[0]
                            sources = [lookup[cid] for cid in paragraph.claim_ids]
                            scope = ScopedEvidence(all_evidence, {x.paper_id for x in sources}, retrieval)
                            checked = await verify(tasks, paragraph, scope, sources, relations=[r for r in relations if r.id in paragraph.relation_ids])
                            writing_history.append({"paragraph": paragraph.model_dump(), "verification": checked.model_dump()})
                        paragraphs.append({"paragraph": paragraph.model_dump(), "verification": checked.model_dump()})
                        destination = prose if checked.verdict == "verified" else rejected_prose
                        accepted_count += int(checked.verdict == "verified")
                        destination.extend([paragraph.statement, "主张来源：" + "、".join(paragraph.claim_ids), "关系来源：" + "、".join(paragraph.relation_ids),
                            f"核查：{VERDICTS[checked.verdict]}；{checked.reason}"])
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    warnings.append(str(exc))
            discussed = {rid for entry in paragraphs if entry['verification']['verdict'] == 'verified' for rid in entry['paragraph'].get('relation_ids', [])}
            missing_relations = [r.id for r in approved_relations(relations, checks) if r.id not in discussed]
            if missing_relations:
                warnings.append(f"{len(missing_relations)} 条已核查关系尚未在通过核查的正文中显式论述，完整关系保留在比较矩阵中")
            if len(rejected_prose) > 2:
                sections.append("\n\n".join(rejected_prose))
            if accepted_count:
                sections.insert(0, "\n\n".join(prose))
            else:
                warnings.append("没有通过核查的综述正文，保留逐篇报告、关系分析及待审草稿。")
        if cross_enabled and not relations:
            warnings.append("未得到可交付的跨篇关系，不能据此推断论文之间没有冲突或差异。")
        if omitted_pairs:
            warnings.append(f"预算不足，{omitted_pairs} 个候选主张对尚未比较。")
        retrieval_status = retrieval.diagnostics() if retrieval else {"status": "not_configured", "warnings": [state.get("research_index_warning") or "未配置向量服务，使用原文分块阅读与引用回查。"]}
        warnings.extend(retrieval_status["warnings"])
        if reasoning:
            sections.insert(0, render_reasoning(reasoning))
            warnings.extend(w for result in reasoning for w in result['warnings'])
        title = "研究分析报告" if cross_enabled or reasoning else "逐篇阅读报告"
        markdown = f"# {title}\n\n" + "\n\n".join([*("> " + w for w in dict.fromkeys(warnings)), *sections])
        references = [i["paper"] for i in items if str(i.get("paper", {}).get("paperId") or i.get("paper", {}).get("id")) in ({c.paper_id for c in cards} | set(external_search["added_paper_ids"]))]
        if references:
            entries = []
            for paper in references:
                paper_id = str(paper.get("paperId") or paper.get("id"))
                location = paper.get("doi") or paper.get("url") or paper.get("pdf_url") or ""
                entries.append(f"- [{paper_id}] {paper.get('title', paper_id)} ({paper.get('year') or '年份未知'}) {location}")
            markdown += "\n\n## 论文来源\n\n" + "\n".join(entries)
        if not cards:
            markdown += "\n\n没有可阅读的论文，请调整检索条件或提供论文资料。"
        revision_history = []
        if revision:
            record = {k: v for k, v in revision.items() if k not in {"history", "seed_card", "evidence_versions"}}
            record["result_claims"] = [{"id": c.id, "version": c.version} for card in cards if card.paper_id == revision["target"]["paper_id"] for c in card.claims]
            revision_history = [*revision.get("history", []), record]
        report = {"schema_version": 5, "core_reviews": core_reviews,
            "first_core_review_seconds": min((s["elapsed_seconds"] for s in core_reviews), default=None), "primary_paper_ids": sorted(primary_ids), "research_questions": questions, "argument_plans": argument_plans, "reasoning": reasoning, "source_manifests": source_manifests, "execution_policy": policy, "incremental": plan.audit(),
            "dependencies": dependency_manifest(cards, relations, paragraphs, all_evidence, source_manifests),
            "challenges": challenges, "pair_outcomes": pair_outcomes, "visual": visual_audits, "visual_model": getattr(llm, "signature", None), "output_mode": options.output_mode, "cross_paper_executed": cross_enabled,
            "assessments": assessments, "comparison_matrix": matrix, "relation_graph": relation_graph,
            "cards": [x.model_dump() for x in cards], "relations": [x.model_dump() for x in relations],
            "verifications": [x.model_dump() for x in checks], "relation_history": relation_history, "paragraphs": paragraphs, "writing_history": writing_history,
            "evidence": [x.model_dump() for x in all_evidence], "references": references, "warnings": list(dict.fromkeys(warnings)),
            "coverage": {"candidate_policy": "condition_priority_paper_pair_round_robin", "compared_pairs": compared_pairs, "omitted_pairs": omitted_pairs},
            "calls": tasks.calls, "usage": tasks.usage, "retrieval": retrieval_status, "external_search": external_search,
            "revision_history": revision_history}
        await store.save("research_state", report)
        if reporter:
            reporter.completed("研究报告已生成，核查状态与证据已保留", **tasks.usage)
        return {"research_report": report, "research_markdown": markdown, "research_artifact_refs": store.artifacts}
    async def guarded(state):
        owned = "research_node_llm" not in state
        llm = load_read_agent_llm() if owned else state["research_node_llm"]
        index = state.get("research_evidence_index")
        owned_index = "research_evidence_index" not in state
        warning = ""
        execution_state = dict(state)
        try:
            if owned_index and (state["request"].constraints.get("allow_external_search") or any(i.get("full_text", {}).get("markdown_path") for i in state.get("read_results", []))):
                try:
                    index = EvidenceIndex.from_config()
                except Exception as exc:
                    warning = f"向量服务配置不可用，继续原文阅读：{type(exc).__name__}"
            elif owned_index:
                warning = "仅有摘要资料，本轮未建立全文向量索引。"
            execution_state.update(research_node_llm=llm, research_evidence_index=index, research_index_warning=warning)
            return await node(execution_state)
        except (Exception, asyncio.CancelledError) as exc:
            runtime = state.get("runtime_context")
            port = getattr(runtime, "sync_port", None)
            if port:
                request = state["request"]
                checkpoint = {
                    "phase": "research", "request": {"topic": request.topic, "constraints": request.constraints, "language": request.language},
                    "search_results": [p.to_dict() for p in state.get("search_results", [])],
                    "read_results": state.get("read_results", []), "read_artifact_refs": state.get("read_artifact_refs", []),
                }
                checkpoint = execution_state.get("read_resume_checkpoint") or checkpoint
                port.for_node("research", "论文判断与证据核查").failed(
                    str(exc) or "研究任务已停止", checkpoint=checkpoint, recovery_status="research_interrupted")
            raise
        finally:
            try:
                if owned_index and index is not None:
                    await index.aclose()
            finally:
                if owned and llm is not None:
                    await llm.aclose()
    return guarded
