"""Canonical Figure/Table/Equation objects assembled from existing PDF evidence views."""
from __future__ import annotations

import re

from .table_layout import html_table
from .models import EvidenceSpan, TableCell, VisualObject, version_of


_REFERENCE = re.compile(
    r"(?i)(?<![a-z0-9_])(?:(fig(?:ure)?\.?)|(table)|(eq(?:uation)?\.?)|([图表式]))\s*([A-Z]?\d+(?:[.\-]\d+)?(?:[a-z]|\([a-z]\))?)"
)
_CAPTION = re.compile(
    r"(?i)^\s*(?:续(?=表))?(?:(fig(?:ure)?\.?)|(table)|(eq(?:uation)?\.?)|([图表式]))\s*"
    r"([A-Z]?\d+(?:[.\-]\d+)?[a-z]?)\s*(?:[|.:：—\-]\s*)?(.*)$"
)
_DISCUSSION_OPENING = re.compile(
    r"(?i)^(?:(?:shows?|illustrates?|depicts?|presents?|reports?|demonstrates?|indicates?)\b|"
    r"展示|显示|表明|说明|给出)"
)


_CONTINUED = re.compile(r"^(?:\s*[(:：.—-]\s*)?(?:continued\b|cont[’']?d\b|续表|续)", re.I)


def _pipe_table(span, previous):
    lines = span.text.splitlines()
    rows = [(i, [v.strip() for v in line.strip()[1:-1].split("|")])
            for i, line in enumerate(lines) if line.strip().startswith("|") and line.strip().endswith("|")]
    if not rows or rows[-1][0] - rows[0][0] + 1 != len(rows):
        return None
    values = [row for _, row in rows]
    has_headers = len(values) >= 2 and all(re.fullmatch(r":?-{3,}:?", v) for v in values[1])
    if has_headers:
        headers = values[0]
        if len(values[1]) != len(headers):
            return None
        data = values[2:]
        header_id = span.id
    elif previous:
        headers, header_id, _ = previous
        data = values
    else:
        return None
    return headers, data, header_id


def _table_cells(span, previous=None):
    """Return cell views and the schema available to a numbered continuation."""
    if re.search(r"<table(?:\s|>)", span.text, re.I):
        parsed = html_table(span.text)
        if parsed is None:
            return [], None
        headers, data = parsed
        header_id = span.id
    else:
        parsed = _pipe_table(span, previous)
        if parsed is None:
            return [], None
        headers, data, header_id = parsed
    if (len(headers) < 2 or not all(headers[1:]) or len(set(headers[1:])) != len(headers)-1
            or not data or any(len(row) != len(headers) or not row[0] for row in data)
            or any(all(re.fullmatch(r":?-{3,}:?", v) for v in row) for row in data)
            or len(data)*(len(headers)-1) > 128):
        return [], None
    note_marker = r"(?:\*{1,3}|[†‡]|\[[a-z0-9]+\])"
    notes = {}
    for line in span.text.splitlines():
        match = re.fullmatch(rf"\s*({note_marker})(?:\s+|[:：]\s*)(\S.+)", line)
        if match:
            notes[match[1]] = line.strip() if match[1] not in notes else None
    offset = previous[2] if previous and previous[0] == headers else 0
    result = []
    for r, row in enumerate(data, 1 + offset):
        for c, value in enumerate(row[1:], 2):
            if not value:
                continue
            cell = TableCell(row=r, column=c, row_header=row[0], column_header=headers[c-1], value=value)
            marked = [cell.row_header, value]
            if header_id == span.id:
                marked.append(cell.column_header)
            applicable = list(dict.fromkeys(notes[m[0]] for field in marked
                if (m := re.search(note_marker + "$", field)) and notes.get(m[0])))
            version = version_of([span.version, header_id, cell.model_dump(), applicable])
            text = "\n".join([f"{cell.row_header} | {cell.column_header} | {cell.value}", *applicable])
            result.append(span.model_copy(update={"id": f"{span.id}:cell:{r}:{c}@{version}",
                "version": version, "header_evidence_id": header_id, "parent_id": span.id, "previous_id": "", "next_id": "",
                "overlap_characters": 0, "bbox": None, "reading_order": None, "table_cell": cell, "view_type": "table_cell",
                "text": text, "table_notes": applicable}))
    return result, (headers, header_id, offset + len(data))


def build_multimodal_catalog(evidence: list[EvidenceSpan], catalog: list[dict] | None = None,
        visual_pages: dict | None = None) -> tuple[list[EvidenceSpan], list[VisualObject]]:
    """Bind caption, OCR and discussion spans to stable paper-scoped visual objects."""

    evidence = [span for span in evidence if span.table_cell is None]
    if not evidence:
        return evidence, []
    paper_id = evidence[0].paper_id
    existing = {(s.page_start, s.text) for s in evidence}
    existing_ids = {span.id for span in evidence}
    for page in catalog or []:
        for label, text in page.get("caption_texts", {}).items():
            if (page["page"], text) in existing:
                continue
            version = version_of([paper_id, page["page"], label, page["source_version"], text])
            evidence.append(EvidenceSpan(id=f"{paper_id}:caption:{label}@{version}", version=version,
                paper_id=paper_id, text=text, source_level="full_text", source_version=page["source_version"],
                page_start=page["page"], page_end=page["page"], view_type="caption",
                reading_order=page.get("object_reading_order", {}).get(label)))
        for label, text in page.get("object_texts", {}).items():
            box = page["object_bboxes"][label]
            oid = f"{paper_id}:visual-object:{label}"
            for offset in range(0, len(text), 4000):
                content = text[offset:offset+4000]
                version = version_of([page["source_version"], page["page"], label, box, offset, content])
                eid = f"{oid}:native@{version}"
                if eid not in existing_ids:
                    existing_ids.add(eid)
                    evidence.append(EvidenceSpan(id=eid, paper_id=paper_id, version=version, text=content,
                        source_level="full_text", source_version=page["source_version"], bbox=box,
                        page_start=page["page"], page_end=page["page"], view_type="object_text",
                        visual_object_ids=[oid], reading_order=page["object_reading_order"].get(label)))
    records = {}
    caption_labels_by_evidence = {}
    for span in evidence:
        if span.crop_version or span.view_type == "object_text":
            continue
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

    pages_by_number = {page["page"]: page for page in catalog or []}
    updated = []
    table_parts = []
    for span in evidence:
        referenced = {label for label, _ in find_object_labels(span.text)}
        caption_labels = caption_labels_by_evidence.get(span.id, set())
        linked = referenced | caption_labels
        if span.crop_version or span.view_type == "object_text":
            linked = {label for label, oid in object_ids.items() if oid in span.visual_object_ids}
        elif span.modality == "visual_transcription" and span.page_start:
            page_labels = labels_by_page.get(span.page_start, set())
            if not linked and len(page_labels) == 1:
                linked |= page_labels
        view_type = "ocr" if span.modality == "visual_transcription" else (
            "object_text" if span.view_type == "object_text" else "caption" if caption_labels else
            "nearby_discussion" if linked else span.view_type)
        bound = span.model_copy(update={
            "visual_object_ids": [object_ids[label] for label in sorted(linked) if label in object_ids],
            "view_type": view_type,
        })
        updated.append(bound)
        if (bound.source_level == "full_text" and len(linked) == 1 and len(bound.visual_object_ids) == 1
                and next(iter(linked)).startswith("table:") and (bound.crop_version or caption_labels)):
            table_parts.append((bound, next(iter(linked))))

    schemas = {}
    for part, label in sorted(table_parts, key=lambda item: item[0].page_start or 0):
        source = part.source_version or part.crop_source_version
        page = part.page_start or 0
        lines = part.text.splitlines()
        if part.crop_version:
            lines.append(pages_by_number.get(page, {}).get("caption_texts", {}).get(label, ""))
        continued = any(line.lstrip().startswith("续表") or _CONTINUED.match(match[6].strip())
            or re.search(r"[(（]\s*(?:continued|续表?)\s*[)）]\s*$", match[6], re.I)
            for line in lines if (match := _CAPTION.match(line)) and _canonical(match.groups()[:5])[0] == label)
        key = (part.paper_id, source, label, page)
        previous = schemas.get((*key[:3], page-1)) if source and continued else None
        cells, schema = _table_cells(part, previous)
        updated.extend(cells)
        if schema and part.page_end in (None, page):
            schemas[key] = schema if key not in schemas else None

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
        page_meta = pages_by_number.get(record["page"], {})
        payload = {"source_version": page_meta.get("source_version", ""),
            "crop_version": next((span.crop_version for span in views if span.crop_version), ""),
            "paper_id": paper_id, "object_type": record["object_type"], "label": label,
            "panel_bboxes": page_meta.get("panel_bboxes", {}).get(label, {}),
            "reading_order": page_meta.get("object_reading_order", {}).get(label),
            "page": record["page"], "caption": record["caption"], "bbox": page_meta.get("object_bboxes", {}).get(label),
            "image_version": image_version, "caption_evidence_ids": caption_ids,
            "cell_evidence_ids": [s.id for s in views if s.table_cell],
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
    number = number.casefold().replace("(", "").replace(")", "")
    return f"{object_type}:{number}", object_type


def _record(object_type):
    return {"object_type": object_type, "page": None, "caption": "", "caption_ids": set()}
