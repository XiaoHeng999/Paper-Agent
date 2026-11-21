"""Document and coverage dependencies, including pages that have not been read."""
import hashlib
from pathlib import Path
from .models import version_of


def file_version(value):
    if not value:
        return {"status": "not_provided"}
    path = Path(value)
    try:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {"status": "available", "sha256": digest}
    except OSError:
        return {"status": "unavailable"}


def document_version(item):
    fulltext = item.get('full_text', {})
    return {"source": file_version(fulltext.get('source_path')),
            "markdown": file_version(fulltext.get('markdown_path')),
            "paper_metadata": version_of(item.get('paper', {}))}


def source_manifest(document, evidence, audit):
    coverage = {"status": audit['status'], "omitted_pages": audit['omitted_pages'],
        "pages": [{k: p.get(k) for k in ('page', 'image_version', 'source_version', 'scanned')} |
                  {"failed": bool(p.get('error'))} for p in audit['pages']]}
    value = {"document": document, "coverage": coverage, "evidence": {e.id: e.version for e in evidence}}
    return {**value, "version": version_of(value)}
