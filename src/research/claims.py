"""Merge only exact semantic duplicates; never merge merely similar wording."""
from .models import Claim, version_of


def meaning(claim):
    return {k: v for k, v in claim.model_dump().items() if k not in {'id', 'version', 'evidence_ids'}}


def merge_duplicate(previous, incoming, scope):
    if meaning(previous) != meaning(incoming):
        return None
    ids = list(dict.fromkeys([*previous.evidence_ids, *incoming.evidence_ids]))
    selected = scope.read(ids)
    payload = {k: v for k, v in incoming.model_dump().items() if k not in {'id', 'version', 'paper_id'}}
    payload['evidence_ids'] = ids
    version = version_of({'claim': payload, 'sources': [(e.id, e.version) for e in selected]})
    return Claim(**payload, paper_id=incoming.paper_id, id=f'{incoming.paper_id}:claim:{version}', version=version)
