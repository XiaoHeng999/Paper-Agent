"""Version-bound reuse of immutable research objects; no agent conversation handoff."""
from collections import Counter
from pathlib import Path
import hashlib
from .models import PaperCard, AnalysisRelation, VerificationResult, version_of


def policy_version(tasks, question, retrieval):
    return version_of({"engine": "research-v9-multimodal-objects",
        "implementation": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("agents.py", "citation_checks.py", "retrieval.py", "context.py", "visual.py", "image_retrieval.py", "multimodal.py", "table_layout.py", "label_ocr.py", "claims.py", "scheduling.py", "assessment.py", "writing.py", "provenance.py", "workflow.py", "models.py", "planning.py", "selection.py", "incremental.py", "argumentation.py", "storage.py", "supplement.py")}, "model": getattr(tasks.llm, "signature", None),
        "question": question, "context_chars": tasks.options.context_chars,
        "repair_rounds": tasks.options.max_repair_rounds, "reasoning": tasks.options.reasoning_enabled, "external": tasks.options.allow_external_search, "visual": tasks.options.visual_verification,
        "evidence_reranking": tasks.options.evidence_reranking_enabled,
        "image_embedding": tasks.image_index.signature() if getattr(tasks, "image_index", None) else None,
        "visual_limit": tasks.options.visual_page_limit, "retrieval": getattr(retrieval, "collection", None)})


def source_versions(evidence, paper_id):
    return {e.id: e.version for e in evidence if e.paper_id == paper_id}


class IncrementalPlan:
    def __init__(self, parent, evidence, policy, revision=None, source_manifests=None):
        self.parent = parent or {}
        self.policy = policy
        self.counts = Counter()
        self.changed_papers = set()
        self.old_claim = revision["target"]["id"] if revision else None
        self.compatible = bool(parent and parent.get("execution_policy") == policy)
        self.reason = "版本匹配，按依赖复用" if self.compatible else "首次运行或执行策略变化，重新计算"
        self.cards = {c["paper_id"]: PaperCard.model_validate(c) for c in self.parent.get("cards", [])}
        old_evidence = self.parent.get("evidence", [])
        for pid in self.cards:
            old = {e["id"]: e["version"] for e in old_evidence if e["paper_id"] == pid}
            previous_manifest = self.parent.get("source_manifests", {}).get(pid)
            manifest_changed = source_manifests is not None and previous_manifest != source_manifests.get(pid)
            if manifest_changed or old != source_versions(evidence, pid):
                self.changed_papers.add(pid)
        self.claims = {c.id: c for card in self.cards.values() for c in card.claims}
        self.previous_checks = {(v.target_id, v.target_version): v for card in self.cards.values() for v in card.verifications}
        self.current_checks = self.previous_checks
        self.pairs = {tuple(p["claim_ids"]): p for p in self.parent.get("pair_outcomes", [])}

    def card(self, paper_id):
        card = self.cards.get(paper_id)
        if not self.compatible or paper_id in self.changed_papers or not card or not card.claims or card.omitted_evidence_ids or any("本次语义核查未完成" in v.unchecked for v in card.verifications) or any(c.id == self.old_claim for c in card.claims):
            return None
        self.counts["cards"] += 1
        return card.model_copy(deep=True)

    def check(self, claim):
        if not self.compatible or claim.paper_id in self.changed_papers or claim.id == self.old_claim:
            return None
        card = self.cards.get(claim.paper_id)
        old = self.claims.get(claim.id)
        if not card or old != claim:
            return None
        result = next((v for v in card.verifications if v.target_id == claim.id and v.target_version == claim.version), None)
        if result and "本次语义核查未完成" not in result.unchecked:
            self.counts["claim_checks"] += 1
            return result.model_copy(deep=True)

    def bind_cards(self, cards):
        self.current_checks = {(v.target_id, v.target_version): v for card in cards for v in card.verifications}

    def dependencies_match(self, claims):
        return self.compatible and all(c.id != self.old_claim and c.paper_id not in self.changed_papers
            and self.claims.get(c.id) == c and self.current_checks.get((c.id, c.version)) == self.previous_checks.get((c.id, c.version)) for c in claims)

    def pair(self, first, second):
        previous = self.pairs.get((first.id, second.id))
        if not previous or not self.dependencies_match([first, second]) or previous.get("status") != "complete":
            return None
        relations = [AnalysisRelation.model_validate(r) for r in previous["relations"]]
        checks = [VerificationResult.model_validate(v) for v in previous["verifications"]]
        if len(relations) != len(checks) or any(v.target_id != r.id or v.target_version != r.version for r, v in zip(relations, checks)):
            return None
        self.counts["pairs"] += 1
        return relations, checks

    def paragraphs(self, eligible, relations=(), checks=()):
        from .writing import relation_context
        lookup = {c.id: c for c in eligible}
        retained = []
        for entry in self.parent.get("paragraphs", []):
            p, check = entry["paragraph"], entry["verification"]
            ids = p["claim_ids"]
            context_matches = p.get("relation_context", {}) == relation_context(ids, relations, checks)
            if context_matches and "本次语义核查未完成" not in check.get("unchecked", []) and ids and all(cid in lookup for cid in ids) and self.dependencies_match([lookup[cid] for cid in ids]) and check["target_id"] == p["id"] and check["target_version"] == p["version"]:
                retained.append(entry)
        self.counts["paragraphs"] += len(retained)
        return retained

    def audit(self):
        return {"enabled": bool(self.parent), "compatible": self.compatible, "reason": self.reason,
            "changed_source_papers": sorted(self.changed_papers), "reused": dict(self.counts)}


def dependency_manifest(cards, relations, paragraphs, evidence, source_manifests=None):
    claims = {c.id: c for card in cards for c in card.claims}
    manifest = {}
    checks = {v.target_id: version_of(v.model_dump()) for card in cards for v in card.verifications}
    for card in cards:
        for c in card.claims:
            # Independent counterevidence search depends on the entire permitted paper corpus.
            manifest[c.id] = {"version": c.version, "sources": source_versions(evidence, c.paper_id), "verification_version": checks.get(c.id),
                "document_dependency": (source_manifests or {}).get(c.paper_id, {}).get("version")}
    for r in relations:
        manifest[r.id] = {"version": r.version, "claims": {cid: claims[cid].version for cid in r.claim_ids}, "claim_checks": {cid: checks.get(cid) for cid in r.claim_ids}}
    for entry in paragraphs:
        p = entry["paragraph"]
        manifest[p["id"]] = {"version": p["version"], "claims": {cid: claims[cid].version for cid in p["claim_ids"]}, "claim_checks": {cid: checks.get(cid) for cid in p["claim_ids"]}, "relations": p.get("relation_context", {})}
    return manifest
