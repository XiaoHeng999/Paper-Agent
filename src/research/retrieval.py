"""Vector retrieval is an evidence tool, independent of agent orchestration."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from src.llm import ModelConfig, SystemConfig, make_provider
from src.llm.base import normalize_token_usage
from src.repositories.chroma.vector_store import VectorUpsertItem, make_chroma_store
from .models import version_of

_INDEX_POLICY = "evidence-views-v3-visual-objects"


class EvidenceIndex:
    def __init__(self, snapshot, path, collection, *, dimensions=None, batch_size=32, owned=False):
        self.snapshot = snapshot
        self.path = path
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        self.owned = owned
        # Model/dimension changes must not mix incompatible vectors in one collection.
        self.collection = collection + "_evidence_" + version_of([snapshot.signature, dimensions, _INDEX_POLICY])[:12]
        self.ready = set()
        self.warnings = []
        self.queries = 0
        self.indexed = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.semaphore = asyncio.Semaphore(1)

    @classmethod
    def from_config(cls):
        path = Path("config/model.json")
        if not path.is_file():
            return None
        config = ModelConfig.from_dict(json.loads(path.read_text(encoding="utf-8")), SystemConfig.load())
        if not config.embedding_profiles:
            return None
        profile = config.resolve_embedding_profile()
        snapshot = make_provider(config, embedding_profile_name=config.default_embedding_profile)
        return cls(snapshot, config.system.read.vector_store_path, config.system.read.vector_store_collection,
                   dimensions=profile.dimensions, batch_size=profile.batch_size, owned=True)

    def key(self, spans):
        return version_of(sorted((s.id, s.version) for s in spans))

    async def embed(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = await self.snapshot.provider.embed(batch, dimensions=self.dimensions)
            if not response.ok or len(response.embeddings) != len(batch):
                raise ValueError("向量服务未返回完整有效结果")
            for vector in response.embeddings:
                if not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                    raise ValueError("向量服务返回无效向量")
            usage = normalize_token_usage(response.usage)
            for key in self.usage:
                self.usage[key] += usage[key]
            vectors.extend(response.embeddings)
        return vectors

    def _store(self, action):
        store = make_chroma_store(self.path, self.collection)
        try:
            return action(store)
        finally:
            store.close()

    async def prepare(self, spans):
        if not spans:
            return
        corpus = self.key(spans)
        ids = [corpus + ":" + s.id for s in spans]
        async with self.semaphore:
            existing = await asyncio.to_thread(self._store, lambda store: store.get_by_ids(ids))
            present = {x.id for x in existing}
            missing = [(key, span) for key, span in zip(ids, spans) if key not in present]
            if missing:
                vectors = await self.embed([s.text for _, s in missing])
                records = [VectorUpsertItem(id=key, embedding=v, document=s.text,
                           metadata={"paper_id": s.paper_id, "evidence_id": s.id, "version": s.version,
                                     "corpus": corpus, "source_level": s.source_level,
                                     "node_id": (s.visual_object_ids[0] if s.visual_object_ids else s.parent_id or s.id),
                                     "view_type": s.view_type,
                                     "section_path": s.section or "", "object_type":
                                     ("visual" if s.modality == "visual_transcription" else "text"),
                                     "page_start": s.page_start or 0, "page_end": s.page_end or 0})
                           for (key, s), v in zip(missing, vectors, strict=True)]
                await asyncio.to_thread(self._store, lambda store: store.upsert(records))
                self.indexed += len(records)
            self.ready.add(corpus)

    async def search(self, query, scope, top_k=5):
        # The caller supplies a scoped view, never arbitrary model-provided filters.
        results = []
        if not scope.allowed or not any(
            self.key([s for s in scope.spans.values() if s.paper_id == paper_id]) in self.ready
            for paper_id in scope.allowed
        ):
            return results
        async with self.semaphore:
            vectors = await self.embed([query])
            self.queries += 1
            for paper_id in sorted(scope.allowed):
                spans = [s for s in scope.spans.values() if s.paper_id == paper_id]
                corpus = self.key(spans)
                if not spans or corpus not in self.ready:
                    continue
                where = {"$and": [{"paper_id": paper_id}, {"corpus": corpus}]}
                hits = await asyncio.to_thread(self._store, lambda store: store.query_by_embedding(
                    vectors[0], top_k=min(top_k, len(spans)), where=where))
                for hit in hits:
                    span = scope.spans.get(hit.metadata.get("evidence_id"))
                    # Rehydrate authoritative source; never trust returned vector-store text.
                    if (span is None or span.paper_id != paper_id or hit.metadata.get("paper_id") != paper_id or hit.metadata.get("corpus") != corpus
                            or hit.metadata.get("version") != span.version):
                        continue
                    results.append((hit.distance if hit.distance is not None else float("inf"), span))
        results.sort(key=lambda pair: pair[0])
        return [span for _, span in results[:top_k]]

    def diagnostics(self):
        return {"status": "ready" if self.ready else "unavailable", "indexed_spans": self.indexed,
                "queries": self.queries, "usage": self.usage, "warnings": list(dict.fromkeys(self.warnings))}

    async def aclose(self):
        if self.owned:
            await self.snapshot.aclose()
