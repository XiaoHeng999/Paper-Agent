"""Explicit relation context for synthesis and paragraph invalidation."""
from .models import version_of


def approved_relations(relations, checks):
    accepted = {(v.target_id, v.target_version): v for v in checks if v.verdict == 'verified'}
    return [r for r in relations if (r.id, r.version) in accepted]


def relation_context(claim_ids, relations, checks):
    ids = set(claim_ids)
    by_id = {(v.target_id, v.target_version): v for v in checks}
    return {r.id: version_of({'relation': r.model_dump(), 'verification': by_id[(r.id, r.version)].model_dump()})
            for r in approved_relations(relations, checks) if ids.intersection(r.claim_ids)}
