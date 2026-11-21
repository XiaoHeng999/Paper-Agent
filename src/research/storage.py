from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from src.utils.read_utils.chunkers import load_chunks_file, build_chunks_file
from src.paper_retrieval.models import PaperDocument
from src.utils.read_utils.read_fulltext import convert_fulltext_to_markdown, markdown_source_is_current
from .models import EvidenceSpan, version_of


def load_evidence(item: dict) -> list[EvidenceSpan]:
    paper = item.get("paper", {})
    paper_id = str(paper.get("paperId") or paper.get("id") or "")
    fulltext = item.get("full_text", {})
    markdown_path = fulltext.get("markdown_path")
    evidence = []
    if markdown_path:
        path = Path(markdown_path)
        document = PaperDocument(id=str(paper.get("id") or paper_id), paperId=paper_id, title=str(paper.get("title") or ""))
        source_path = Path(fulltext["source_path"]) if fulltext.get("source_path") else None
        if source_path and source_path.is_file() and not markdown_source_is_current(document, source_path, path):
            converted = convert_fulltext_to_markdown(document, source_path=source_path, source_url=fulltext.get("source_url"))
            if not converted.markdown_path:
                raise ValueError("原始文件已变化且无法重新解析，不能沿用旧证据")
            path = converted.markdown_path
        # Rebuild legacy/stale chunks from the authoritative source before versioning evidence.
        if path.is_file():
            chunks = build_chunks_file(document, markdown_path=path).chunks
        else:
            chunks = load_chunks_file(path.parent / "chunk.json")
        for chunk in chunks:
            if chunk.paperId != paper_id:
                continue
            # Preserve all text. Large legacy chunks become explicitly identified segments.
            for offset in range(0, len(chunk.content), 4000):
                text = chunk.content[offset:offset + 4000]
                locator = chunk.chunk_id if len(chunk.content) <= 4000 else f"{chunk.chunk_id}:offset{offset}"
                revision = version_of({"text": text, "locator": locator, "pages": [chunk.page_start, chunk.page_end]})
                evidence.append(EvidenceSpan(id=f"{locator}@{revision}", paper_id=paper_id, version=revision,
                    text=text, source_level="full_text", page_start=chunk.page_start, page_end=chunk.page_end, section=chunk.section))
    if not evidence and paper.get("abstract"):
        text = str(paper["abstract"])
        for offset in range(0, len(text), 4000):
            segment = text[offset:offset + 4000]
            revision = version_of(segment)
            evidence.append(EvidenceSpan(id=f"{paper_id}:abstract:{offset}@{revision}", paper_id=paper_id,
                version=revision, text=segment, source_level="abstract", section="abstract"))
    return evidence


class ResearchStore:
    """Immutable versioned artifacts indexed by the existing session SQLite repository."""

    def __init__(self, state: dict, reporter=None):
        self.repo = state.get("session_repo")
        self.session = state.get("session_key")
        self.turn = str(state.get("turn_id") or "run")
        self.reporter = reporter
        self.artifacts = []

    async def save(self, kind: str, payload, *, cache_key: str = "", name: str = "", metadata: dict | None = None):
        if not self.repo or not self.session:
            return
        content = payload if isinstance(payload, (str, bytes)) else json.dumps(payload, ensure_ascii=False, indent=2)
        name = name or f"{kind}.json"
        identity = version_of({"kind": kind, "content": hashlib.sha256(content).hexdigest() if isinstance(content, bytes) else content, "turn": self.turn})
        write = asyncio.create_task(asyncio.to_thread(self.repo.write_artifact, self.session, kind, name, content,
            relative_path=f"artifacts/research/{identity}/{name}",
            metadata={**(metadata or {}), "turn_id": self.turn, "cache_key": cache_key, "schema_version": 1}))
        try:
            record = await asyncio.shield(write)
        except asyncio.CancelledError:
            # A worker thread cannot be cancelled; finish its immutable write before returning.
            await write
            raise
        self.artifacts.append(record)
        if self.reporter:
            self.reporter.artifact(record)
        return record

    async def load(self, cache_key: str):
        if not self.repo or not self.session:
            return None
        record = await asyncio.to_thread(self.repo.get, self.session)
        for artifact in reversed(record.artifacts):
            if artifact.get("artifact_type") != "research_paper" or artifact.get("metadata", {}).get("cache_key") != cache_key:
                continue
            path = await asyncio.to_thread(self.repo.read_artifact_path, self.session, artifact["id"])
            if path:
                try:
                    return json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
                except (ValueError, OSError):
                    continue
        return None
