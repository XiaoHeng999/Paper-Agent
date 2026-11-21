from __future__ import annotations

from .models import EvidenceSpan


class ScopedEvidence:
    """Scope enforcement happens before text enters any model prompt."""

    def __init__(self, evidence: list[EvidenceSpan], allowed_paper_ids: set[str], retrieval=None):
        self.retrieval = retrieval
        self.allowed = frozenset(allowed_paper_ids)
        self.spans = {}
        for span in evidence:
            if span.paper_id not in self.allowed:
                continue
            if span.id in self.spans and self.spans[span.id] != span:
                raise ValueError(f"证据标识冲突: {span.id}")
            self.spans[span.id] = span

    def read(self, ids: list[str]) -> list[EvidenceSpan]:
        result = []
        for evidence_id in dict.fromkeys(ids):
            span = self.spans.get(evidence_id)
            if span is None or span.paper_id not in self.allowed:
                raise ValueError(f"证据不存在或不在本任务范围内: {evidence_id}")
            result.append(span)
        return result

    def neighbors(self, ids: list[str], max_chars: int) -> tuple[list[EvidenceSpan], list[str]]:
        selected = self.read(ids)
        total = sum(len(item.text) for item in selected)
        if total > max_chars:
            raise ValueError("核心证据超过上下文预算，需要拆分判断")
        all_spans = list(self.spans.values())
        candidates = []
        for span in selected:
            index = all_spans.index(span)
            for nearby in all_spans[max(0, index - 1):index + 2]:
                if nearby.paper_id == span.paper_id and nearby not in selected and nearby not in candidates:
                    candidates.append(nearby)
        omitted = []
        for span in candidates:
            if total + len(span.text) <= max_chars:
                selected.append(span)
                total += len(span.text)
            else:
                omitted.append(span.id)
        return selected, omitted

    async def search(self, query: str, tasks, top_k: int = 5):
        if self.retrieval is None:
            return []
        import asyncio
        import time
        if tasks.cancellation:
            tasks.cancellation.raise_if_requested()
        try:
            remaining = tasks.deadline - time.monotonic()
            if remaining <= 0:
                return []
            async with asyncio.timeout(min(15, remaining)):
                return await self.retrieval.search(query, self, top_k)
        except Exception as exc:
            self.retrieval.warnings.append(f"向量召回失败，继续使用原文引用：{type(exc).__name__}")
            return []

    async def expand(self, ids, query, max_chars, tasks):
        selected, omitted = self.neighbors(ids, max_chars)
        total = sum(len(s.text) for s in selected)
        for span in await self.search(query, tasks):
            if span in selected:
                continue
            if total + len(span.text) <= max_chars:
                selected.append(span)
                total += len(span.text)
            else:
                omitted.append(span.id)
        return selected, [eid for eid in dict.fromkeys(omitted) if eid not in {s.id for s in selected}]

    def batches(self, max_chars: int, priority_ids=()):
        batch, size = [], 0
        ordered = self.read(list(priority_ids))
        ordered.extend(s for s in self.spans.values() if s not in ordered)
        for span in ordered:
            if len(span.text) > max_chars:
                raise ValueError("原文片段超过上下文预算")
            if batch and size + len(span.text) > max_chars:
                yield batch
                batch, size = [], 0
            batch.append(span)
            size += len(span.text)
        if batch:
            yield batch

    async def adversarial(self, ids, query, max_chars, tasks):
        """Search failure/ablation passages independently of the author's selected citations."""
        core = self.read(ids)
        if sum(len(s.text) for s in core) > max_chars:
            raise ValueError("核心证据超过上下文预算，需要拆分判断")
        terms = ("limitation", "failure", "ablation", "counterexample", "negative result", "局限", "失败", "消融", "反例", "不足")
        candidates = [s for s in self.spans.values() if any(term in s.text.lower() for term in terms)]
        # Semantic recall supplements the deterministic scan; it never broadens the scope.
        candidates += await self.search(query + " limitations failure cases ablation counterexamples 局限 反例 消融", tasks)
        nearby, _ = self.neighbors(ids, max_chars)
        candidates += nearby
        selected, total = list(core), sum(len(s.text) for s in core)
        for span in candidates:
            if span in selected:
                continue
            if total + len(span.text) <= max_chars:
                selected.append(span)
                total += len(span.text)
        inspected = {s.id for s in selected}
        omitted = [s.id for s in self.spans.values() if s.id not in inspected]
        return selected, omitted
