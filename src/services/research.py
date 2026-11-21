"""Session-owned research reports and immutable human revision requests."""
import copy
import json

from src.research.models import Claim, ClaimDraft, PaperCard, version_of


def latest_report(repo, session_key):
    session = repo.get(session_key)
    for artifact in reversed(session.artifacts):
        if artifact.get("artifact_type") == "research_state":
            path = repo.read_artifact_path(session_key, artifact["id"])
            if path:
                return {"artifact_id": artifact["id"], "report": json.loads(path.read_text(encoding="utf-8"))}
    return {"artifact_id": None, "report": None}


def resume_review_request(repo, session_key, body):
    latest = latest_report(repo, session_key)
    report = latest['report']
    if not report or latest['artifact_id'] != body.get('artifact_id'):
        raise ValueError('报告版本已变化，请刷新后再复审')
    if not any(r.get('pending_review') for r in report.get('reasoning', [])):
        raise ValueError('该报告没有待恢复的核查')
    checkpoint_id = report.get('checkpoint_id')
    path = repo.read_artifact_path(session_key, checkpoint_id) if checkpoint_id else None
    if not path:
        raise ValueError('缺少该报告的复审现场')
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    checkpoint['resume_final_review'] = True
    return {'content': '继续核查已保存的研究草稿', 'read_resume_checkpoint': checkpoint}


def revision_request(repo, session_key, body):
    latest = latest_report(repo, session_key)
    if not latest["report"] or body.get("artifact_id") != latest["artifact_id"]:
        raise ValueError("报告版本已变化，请刷新后再修订")
    report = latest["report"]
    cards = [PaperCard.model_validate(card) for card in report["cards"]]
    target = next((c for card in cards for c in card.claims if c.id == body.get("claim_id")), None)
    if target is None or target.version != body.get("target_version"):
        raise ValueError("判断不存在或版本已过期")
    reason = body.get("reason", "")
    changes = body.get("changes")
    allowed = {"statement", "conditions", "experiment", "study", "attribution", "evidence_ids"}
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
        raise ValueError("请提供有效的修订理由（不超过 4000 字符）")
    if not isinstance(changes, dict) or not changes or not set(changes) <= allowed:
        raise ValueError("只能修订判断文字、前提、实验、归属及证据引用")
    payload = {k: v for k, v in target.model_dump().items() if k in ClaimDraft.model_fields}
    draft = ClaimDraft.model_validate({**payload, **changes})
    sources = {e["id"]: e for e in report["evidence"] if e["paper_id"] == target.paper_id}
    if not set(draft.evidence_ids) <= sources.keys():
        raise ValueError("修订不能引用当前论文范围外的证据")
    if len(json.dumps(draft.model_dump(), ensure_ascii=False)) > 24000:
        raise ValueError("修订内容过长，请拆成更小的判断")
    session = repo.get(session_key)
    artifact = next(a for a in session.artifacts if a["id"] == latest["artifact_id"])
    turn = artifact.get("metadata", {}).get("turn_id")
    checkpoint = None
    for record in reversed(session.artifacts):
        if record.get("artifact_type") == "research_checkpoint" and record.get("metadata", {}).get("turn_id") == turn:
            path = repo.read_artifact_path(session_key, record["id"])
            if path:
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                break
    if checkpoint is None:
        raise ValueError("缺少原文恢复现场，不能可靠重核查，请重新执行研究任务")
    has_original_file = any(str(i.get('paper', {}).get('paperId') or i.get('paper', {}).get('id')) == target.paper_id
        and i.get('full_text', {}).get('source_path') for i in checkpoint.get('read_results', []))
    if has_original_file and target.paper_id not in report.get('source_manifests', {}):
        raise ValueError("旧报告缺少原文件版本指纹，请重新分析后再修订")
    checkpoint = copy.deepcopy(checkpoint)
    # Revision stays in the same acquisition scope, without another external search.
    checkpoint["external_search"] = report.get("external_search", {"attempted": True, "status": "not_repeated", "warnings": [], "added_paper_ids": []})
    checkpoint["external_search"]["attempted"] = True
    checkpoint["revision"] = {"parent_artifact_id": latest["artifact_id"], "target": target.model_dump(),
        "replacement": draft.model_dump(), "reason": reason.strip(), "seed_card": next(card.model_dump() for card in cards if card.paper_id == target.paper_id),
        "evidence_versions": {eid: e["version"] for eid, e in sources.items()},
        "source_document": report.get("source_manifests", {}).get(target.paper_id, {}).get("document"),
        "history": report.get("revision_history", []),
        "invalidated_relation_ids": [r["id"] for r in report.get("relations", []) if target.id in r["claim_ids"]],
        "invalidated_paragraph_ids": [p["paragraph"]["id"] for p in report.get("paragraphs", []) if target.id in p["paragraph"]["claim_ids"]]}
    return {"content": f"修订判断并重新核查：{reason.strip()}", "read_resume_checkpoint": checkpoint}
