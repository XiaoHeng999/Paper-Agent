"""Canonical Figure/Table/Equation objects assembled from existing PDF evidence views."""
from __future__ import annotations

import re

from .models import EvidenceSpan, VisualObject, version_of


_REFERENCE = re.compile(
    r"(?i)(?:(fig(?:ure)?\.?)|(table)|(eq(?:uation)?\.?)|([图表式]))\s*([A-Z]?\d+(?:[.\-]\d+)?[a-z]?)"
)
_CAPTION = re.compile(
    r"(?i)^\s*(?:(fig(?:ure)?\.?)|(table)|(eq(?:uation)?\.?)|([图表式]))\s*"
    r"([A-Z]?\d+(?:[.\-]\d+)?[a-z]?)\s*(?:[|.:：—\-]\s*)?(.*)$"
)
_DISCUSSION_OPENING = re.compile(
    r"(?i)^(?:(?:shows?|illustrates?|depicts?|presents?|reports?|demonstrates?|indicates?)\b|"
    r"展示|显示|表明|说明|给出)"
)


def build_multimodal_catalog(evidence: list[EvidenceSpan], catalog: list[dict] | None = None,
        visual_pages: dict | None = None) -> tuple[list[EvidenceSpan], list[VisualObject]]:
    """Bind caption, OCR and discussion spans to stable paper-scoped visual objects."""

    if not evidence:
        return evidence, []
    paper_id = evidence[0].paper_id
    records = {}
    caption_labels_by_evidence = {}
    for span in evidence:
        caption_labels = set()
        for line in span.text.splitlines():
            match = _CAPTION.match(line)
            if not match:
                continue
            if _DISCUSSION_OPENING.match(match.group(6).strip()):
                continue
            label, object_type = _canonical(match.groups()[:5])
            caption_labels.add(label)
            record = records.setdefault(label, _record(object_type))
            record["page"] = record["page"] or span.page_start
            caption = " ".join(line.split()).strip()
            if len(caption) > len(record["caption"]):
                record["caption"] = caption[:2000]
            record["caption_ids"].add(span.id)
        caption_labels_by_evidence[span.id] = caption_labels
        for label, object_type in find_object_labels(span.text):
            records.setdefault(label, _record(object_type))

    for page in catalog or []:
        for label in page.get("captions", []):
            object_type = label.split(":", 1)[0]
            if object_type not in {"figure", "table", "equation"}:
                continue
            record = records.setdefault(label, _record(object_type))
            record["page"] = record["page"] or page.get("page")

    object_ids = {label: f"{paper_id}:visual-object:{label}" for label in records}
    labels_by_page = {}
    for label, record in records.items():
        if record["page"]:
            labels_by_page.setdefault(record["page"], set()).add(label)

    updated = []
    for span in evidence:
        referenced = {label for label, _ in find_object_labels(span.text)}
        caption_labels = caption_labels_by_evidence.get(span.id, set())
        linked = referenced | caption_labels
        if span.modality == "visual_transcription" and span.page_start:
            page_labels = labels_by_page.get(span.page_start, set())
            if not linked and len(page_labels) == 1:
                linked |= page_labels
        view_type = "ocr" if span.modality == "visual_transcription" else (
            "caption" if caption_labels else
            "nearby_discussion" if linked else span.view_type)
        updated.append(span.model_copy(update={
            "visual_object_ids": [object_ids[label] for label in sorted(linked) if label in object_ids],
            "view_type": view_type,
        }))

    visual_pages = visual_pages or {}
    objects = []
    for label, record in sorted(records.items()):
        object_id = object_ids[label]
        views = [span for span in updated if object_id in span.visual_object_ids]
        caption_ids = [span.id for span in views if span.id in record["caption_ids"]]
        ocr_ids = [span.id for span in views if span.view_type == "ocr"]
        discussion_ids = [span.id for span in views if span.view_type == "nearby_discussion"]
        prepared_page = visual_pages.get((paper_id, record["page"])) if record["page"] else None
        image_version = str((prepared_page or {}).get("image_version") or "")
        payload = {"paper_id": paper_id, "object_type": record["object_type"], "label": label,
            "page": record["page"], "caption": record["caption"], "bbox": None,
            "image_version": image_version, "caption_evidence_ids": caption_ids,
            "ocr_evidence_ids": ocr_ids, "discussion_evidence_ids": discussion_ids,
            "view_evidence_ids": [span.id for span in views]}
        version = version_of({**payload, "views": [(span.id, span.version) for span in views]})
        objects.append(VisualObject(id=object_id, version=version, **payload))
    return updated, objects


def find_object_labels(text: str) -> list[tuple[str, str]]:
    result = []
    for match in _REFERENCE.finditer(text):
        value = _canonical(match.groups())
        if value not in result:
            result.append(value)
    return result


def _canonical(groups):
    figure, table, equation, chinese, number = groups[:5]
    object_type = "table" if table or chinese == "表" else "equation" if equation or chinese == "式" else "figure"
    return f"{object_type}:{number.casefold()}", object_type


def _record(object_type):
    return {"object_type": object_type, "page": None, "caption": "", "caption_ids": set()}
