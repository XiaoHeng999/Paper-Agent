"""Evidence-linked presentation and scoring; no extra standing agent."""
from .models import DIMENSIONS


APPLICABLE_KINDS = {
    "evidence_sufficiency": {"result"},
    "comparison_fairness": {"result"},
    "method_increment": {"method"},
    "reproducibility": {"method", "result"},
    "scope_clarity": {"method", "result", "premise", "limitation"},
}


def assess_card(card, options):
    latest = {(v.target_id, v.target_version): v for v in card.verifications}
    dimensions = []
    for key, label in DIMENSIONS.items():
        applicable = [c for c in card.claims if c.kind in APPLICABLE_KINDS[key]]
        checks = [latest.get((c.id, c.version)) for c in applicable]
        observations = [d for v in checks if v and v.fidelity.status == "supported" and v.verdict in {"verified", "qualified"}
                        for d in v.dimensions if d.dimension == key and d.scope == "paper_internal"]
        known = [d for d in observations if d.score is not None]
        complete = bool(checks) and len(known) == len(checks) and not card.omitted_evidence_ids
        # Do not turn absent judgments into low scores, or hide disagreement in an average.
        dimensions.append({"dimension": key, "label": label,
            "score": min(d.score for d in known) if complete else None,
            "status": "assessed" if complete else "unknown", "scope": "paper_internal",
            "reason": "；".join(dict.fromkeys(d.reason for d in observations)) or "已审证据不足，未完成该维度判断",
            "evidence_ids": list(dict.fromkeys(eid for d in observations for eid in d.evidence_ids)),
            "assessed_claims": len(known), "total_claims": len(checks),
            "applicable_claim_ids": [c.id for c in applicable], "excluded_claim_ids": [c.id for c in card.claims if c not in applicable],
            "observations": [d.model_dump() for d in observations]})
    total = None
    reason = "未指定评分目标和权重，不计算总分"
    if options.score_weights:
        selected = {d["dimension"]: d["score"] for d in dimensions}
        if all(selected[key] is not None for key in options.score_weights):
            total = round(sum(selected[k] * w for k, w in options.score_weights.items()) / sum(options.score_weights.values()), 3)
            reason = "用户指定维度的加权分；仅代表当前论文内部已审证据，范围 0–4"
        else:
            reason = "权重包含 unknown 维度，未归零或重新分配权重，不计算总分"
    return {"paper_id": card.paper_id, "target_version": card.version, "dimensions": dimensions,
            "goal": options.scoring_goal, "weights": options.score_weights, "weighted_score": total,
            "score_reason": reason, "novelty_scope": "未据此验证相对整个领域的新颖性"}


def comparison_views(cards, relations, checks):
    by_relation = {(v.target_id, v.target_version): v for v in checks}
    lookup = {c.id: c for card in cards for c in card.claims}
    matrix, edges = [], []
    for relation in relations:
        left, right = [lookup[cid] for cid in relation.claim_ids]
        verification = by_relation.get((relation.id, relation.version))
        row = {"relation_id": relation.id, "left_paper_id": left.paper_id, "right_paper_id": right.paper_id,
               "left_claim_id": left.id, "right_claim_id": right.id, "relation": relation.relation,
               "verdict": verification.verdict if verification else "needs_human_review",
               "axes": [a.model_dump() for a in relation.axes], "rationale": relation.rationale,
               "evidence_ids": relation.evidence_ids, "unresolved_questions": relation.unresolved_questions}
        matrix.append(row)
        edges.append({**row, "source": left.id, "target": right.id})
    nodes = [{"id": c.id, "paper_id": card.paper_id, "title": card.title, "statement": c.statement,
              "evidence_ids": c.evidence_ids} for card in cards for c in card.claims]
    return matrix, {"nodes": nodes, "edges": edges}


def comparison_candidate(left, right):
    return left.kind == right.kind or ({left.kind, right.kind} <= {"result", "premise", "limitation", "method"})
