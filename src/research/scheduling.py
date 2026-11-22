"""Prioritize comparable claims and distribute the finite budget across paper pairs."""
from collections import deque
from itertools import combinations
from .assessment import comparison_candidate


def reading_schedule(claims, core_ids=()):
    """Keep core questions first, then cover each claim kind before its details."""
    core = set(core_ids)
    ordered = [claim for claim in claims if claim.id in core]
    groups = {}
    for claim in claims:
        if claim.id not in core:
            groups.setdefault(claim.kind, deque()).append(claim)
    while groups:
        for kind, queue in list(groups.items()):
            ordered.append(queue.popleft())
            if not queue:
                del groups[kind]
    return ordered


def comparison_priority(first, second):
    score = 0
    for field, weight in [('problem', 8), ('task', 6), ('dataset', 4), ('metric', 2), ('mechanism', 2)]:
        left, right = getattr(first.study, field).casefold(), getattr(second.study, field).casefold()
        if left != 'unknown' and right != 'unknown':
            score += weight if left == right else -weight
    if 'limitation' in (first.kind, second.kind):
        score += 1
    return score


def comparison_schedule(cards):
    groups = []
    for left, right in combinations(cards, 2):
        pairs = [(a, b) for a in left.claims for b in right.claims if comparison_candidate(a, b)]
        pairs.sort(key=lambda pair: (-comparison_priority(*pair), pair[0].id, pair[1].id))
        if pairs:
            groups.append((left, right, deque(pairs)))
    # One promising pair per paper pair before spending another call on the same papers.
    while groups:
        next_round = []
        for left, right, queue in groups:
            first, second = queue.popleft()
            yield left, right, first, second
            if queue:
                next_round.append((left, right, queue))
        groups = next_round
