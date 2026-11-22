"""Bounded, paper-scoped PDF rendering and original-page visual verification.

Page previews used by reading passages can be saved as session PNG artifacts.
PDFium is serialized because the native library is not thread safe.
"""
import asyncio
import base64
import hashlib
import io
import math
import re
import threading
from contextlib import closing
from pathlib import Path

from pydantic import Field
from .context import ScopedEvidence
from .planning import build_reading_plan
from .selection import build_evidence_packets
from .multimodal import find_object_labels
from .models import EvidenceSpan, Record, version_of

_LOCK = threading.RLock()
VISUAL = re.compile(r"\b(?:fig(?:ure)?\.?|table|chart|plot)\s*\d*|图\s*\d|表\s*\d", re.I)


class Transcription(Record):
    text: str = Field(max_length=24000)
    readable: bool
    uncertainties: list[str] = Field(default_factory=list)


def labels(text):
    return {label for label, _ in find_object_labels(text)}


def _raster_layout(page, textpage, text, images):
    """Bind captions to nearby raster objects only for an unambiguous one-to-one layout."""
    if not images or len(images) > 12 or page.get_rotation():
        return {}
    pattern = r"^\s*((?:fig(?:ure)?\.?|table|图|表)\s*[A-Z]?\d+(?:[.-]\d+)?[a-z]?)\s*[:.|：—-]\s*\S"
    captions = [line.strip() for line in text.splitlines() if re.match(pattern, line, re.I)]
    if len(captions) != len(images):
        return {}
    x0, y0, x1, y1 = page.get_bbox()
    layout = {}
    for caption in captions:
        label = next(iter(labels(re.match(pattern, caption, re.I)[1])))
        with closing(textpage.search(caption)) as search:
            match = search.get_next()
            if not match or search.get_next():
                return {}
            x, y, _, upper = textpage.get_charbox(match[0])
        candidates = []
        for left, bottom, right, top in images:
            if not (x0 <= left < right <= x1 and y0 <= bottom < top <= y1
                    and right-left >= 48 and top-bottom >= 24):
                continue
            gap = bottom-upper if label.startswith("figure:") or upper <= bottom else y-top
            if left-12 <= x <= right and -1 <= gap <= 32:
                candidates.append([left, bottom, right, top])
        if len(candidates) != 1 or candidates[0] in layout.values() or label in layout:
            return {}
        layout[label] = candidates[0]
    return layout


def _crop_png(page, bbox):
    """Map unrotated PDF canvas coordinates to the selected page bitmap."""
    from PIL import Image
    x0, y0, x1, y1 = page["page_bbox"]
    left, bottom, right, top = bbox
    data = base64.b64decode(page["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(data)) as image:
        sx, sy = image.width / (x1-x0), image.height / (y1-y0)
        box = (math.floor((left-x0)*sx), math.floor((y1-top)*sy),
               math.ceil((right-x0)*sx), math.ceil((y1-bottom)*sy))
        output = io.BytesIO()
        image.crop(box).save(output, format="PNG")
        return output.getvalue()


async def _visual_gaps(plan, evidence):
    """Use the native-text packet before spending any visual-model budget."""
    questions = [q for q in plan.subqueries if q.source == "user" and not labels(q.question)
        and (any(r.kind == "precision" for r in q.evidence_requirements)
             or re.search(r"hardware|gpu|cpu|a100|硬件|显卡", q.question, re.I))]
    if not questions:
        return []
    scope = ScopedEvidence(evidence, {plan.paper_id})
    packets = await build_evidence_packets(scope, plan.model_copy(update={"subqueries": questions}), None)
    gaps = []
    for packet in packets:
        relevant = {"precision", "baseline_precision"}
        if re.search(r"hardware|gpu|cpu|a100|硬件|显卡", packet.question, re.I):
            relevant.add("hardware")
        missing = sorted(set(packet.missing_requirements) & relevant)
        if missing:
            gaps.append({"subquery_id": packet.subquery_id, "question": packet.question,
                "requirements": missing, "references": sorted({label for span in scope.read(packet.evidence_ids)
                    for label in labels(span.text)})})
    return gaps


def _gap_score(label, caption, gap):
    score = 3 if label in gap["references"] else 0
    topics = (r"latency|speedup|speed|throughput|faster|加速|速度|吞吐|延迟",
              r"precision|quantiz|fp16|bf16|int8|精度|量化",
              r"hardware|gpu|cpu|显卡|硬件", r"baseline|comparison|基线|对比")
    for pattern in topics:
        if re.search(pattern, gap["question"], re.I) and re.search(pattern, caption, re.I):
            score += 2
    return score


def inspect_pdf(path, limit, query="", gaps=()):
    import pypdfium2 as pdfium
    from pypdfium2 import raw
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("视觉 PDF 超过 50 MB 限制")
    source = path.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    candidates, selected = [], []
    wanted = labels(query)
    with _LOCK, pdfium.PdfDocument(source) as document:
        for index in range(min(len(document), 500)):
            with closing(document[index]) as page:
                direct_images = []
                nested_images = False
                width, height = page.get_size()
                image_area = 0
                for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE]):
                    left, bottom, right, top = obj.get_bounds()
                    if obj.level:
                        nested_images = True
                    else:
                        direct_images.append((left, bottom, right, top))
                    image_area += max(0, right-left) * max(0, top-bottom)
                with closing(page.get_textpage()) as textpage:
                    text = textpage.get_text_range()
                    layout = {}
                    if not nested_images and image_area < width * height * .6 and len(text.strip()) >= 40:
                        try:
                            layout = _raster_layout(page, textpage, text, direct_images)
                        except pdfium.PdfiumError:
                            layout = {}
                # Full-page raster plus a searchable header is still a scan.
                scanned = len(text.strip()) < 40 or image_area >= width * height * .6
                gap_candidates = {}
                captions = set(layout)
                for line in text.splitlines():
                    if re.match(r"(?i)^\s*(fig(?:ure)?\.?|table|图|表)\s*\d+", line):
                        captions.update(labels(line))
                    if not gaps or not layout or not line.strip().casefold().startswith(("fig", "table", "图", "表")):
                        continue
                    refs = find_object_labels(line)
                    if refs and refs[0][0] in layout:
                        label = refs[0][0]
                        for gap in gaps:
                            score = _gap_score(label, line, gap)
                            if score > gap_candidates.get(label, {}).get("score", 0):
                                gap_candidates[label] = {"score": score, "subquery_id": gap["subquery_id"],
                                    "requirements": gap["requirements"]}
                if not scanned and not image_area and not VISUAL.search(text):
                    continue
                candidates.append({"page": index+1, "scanned": scanned, "captions": sorted(captions), "visual_content": bool(scanned or image_area or captions),
                    "source_version": source_hash, "page_bbox": list(page.get_bbox()),
                    "object_bboxes": layout, "gap_candidates": gap_candidates,
                    "priority": (int(bool(captions & wanted)), max((v["score"] for v in gap_candidates.values()), default=0), int(bool(re.search(r"(?i)ablation|benchmark|speedup|latency|实验|消融|加速", text)) and bool(captions)), int(scanned), int(bool(captions)), int(image_area > 0))})
        ordered = sorted(candidates, key=lambda p: (tuple(-x for x in p['priority']), p['page']))
        for metadata in ordered[:limit]:
            with closing(document[metadata['page']-1]) as page:
                width, height = page.get_size()
                with closing(page.render(scale=min(2, 1800 / max(width, height)))) as bitmap:
                    output = io.BytesIO()
                    bitmap.to_pil().save(output, format="PNG")
                    data = output.getvalue()
                selected.append({**metadata, "image_version": hashlib.sha256(data).hexdigest(),
                    "url": "data:image/png;base64," + base64.b64encode(data).decode()})
        selected_pages = {p['page'] for p in selected}
        omitted = [p['page'] for p in candidates if p['page'] not in selected_pages] + list(range(501, len(document)+1))
    return selected, omitted, candidates


def render_candidates(path, limit):
    pages, omitted, _ = inspect_pdf(path, limit)
    return pages, omitted


async def prepare_visual(tasks, item, evidence, reading_plan=None):
    paper = item.get("paper", {})
    paper_id = str(paper.get("paperId") or paper.get("id") or "")
    path = item.get("full_text", {}).get("source_path")
    audit = {"paper_id": paper_id, "status": "not_applicable", "pages": [], "omitted_pages": [], "warnings": []}
    if not path or Path(path).suffix.lower() != ".pdf":
        return evidence, audit
    if not tasks.options.visual_verification:
        audit["status"] = "disabled"
        audit["warnings"].append("PDF 视觉核查已关闭；扫描页及图表不能视为已核查")
        return evidence, audit
    try:
        remaining_pages = max(0, 24 - len(tasks.visual_pages))
        fair_limit = max(1, remaining_pages // max(1, getattr(tasks, "remaining_visual_papers", 1))) if remaining_pages else 0
        question = getattr(tasks, "question", "")
        plan = reading_plan or build_reading_plan(question, paper_id, paper.get("title") or paper_id)
        gaps = await _visual_gaps(plan, evidence)
        audit["visual_gaps"] = gaps
        pages, omitted, catalog = await asyncio.to_thread(inspect_pdf, path, min(tasks.options.visual_page_limit, fair_limit), question, gaps)
        tasks.visual_catalog[paper_id] = catalog
        audit.update(status="prepared", omitted_pages=omitted)
        if omitted:
            audit["warnings"].append(f"{paper_id} 有 {len(omitted)} 个候选页未做视觉读取（每篇或本轮页数预算）")
        # Bind text spans to their page image too: unchanged OCR text cannot hide changed charts.
        image_versions = {p["page"]: p["image_version"] for p in pages}
        updated = []
        for span in evidence:
            page_numbers = range(span.page_start, (span.page_end or span.page_start) + 1) if span.page_start else []
            hashes = [image_versions[n] for n in page_numbers if n in image_versions]
            if hashes:
                version = version_of([span.version, hashes])
                span = span.model_copy(update={"id": span.id + ":visual@" + version, "version": version})
            updated.append(span)
        remapped = {old.id: new.id for old, new in zip(evidence, updated)}
        evidence = [span.model_copy(update={
            "previous_id": remapped.get(span.previous_id, span.previous_id),
            "next_id": remapped.get(span.next_id, span.next_id),
        }) for span in updated]
        additions = []
        wanted = labels(question)
        ranked = sorted((label not in wanted, -page["gap_candidates"].get(label, {}).get("score", 0),
            page["page"], label) for page in pages for label in page["object_bboxes"]
            if label in wanted or label in page["gap_candidates"])
        chosen = {(number, label) for _, _, number, label in ranked[:2]}
        for page in pages:
            metadata = {k: v for k, v in page.items() if k != "url"}
            audit["pages"].append(metadata)
            tasks.visual_pages[(paper_id, page["page"])] = page
            if not page["scanned"]:
                for number, label in sorted(chosen):
                    if number != page["page"]:
                        continue
                    try:
                        spans, reused = await _transcribe_crop(tasks, paper_id, page, label)
                        additions.extend(spans)
                        metadata.setdefault("crop_transcriptions", []).append({
                            "label": label, "trigger": "explicit_reference" if label in wanted else "requirement_gap",
                            "crop_version": spans[0].crop_version, "reused": reused})
                    except Exception as exc:
                        audit["warnings"].append(f"{paper_id} 第 {page['page']} 页 {label} 裁剪转录未完成：{exc}")
                continue
            previous = [EvidenceSpan.model_validate(e) for e in getattr(tasks, "previous_visual_evidence", [])
                if e.get("paper_id") == paper_id and e.get("modality") == "visual_transcription"
                and not e.get("crop_version") and e.get("image_version") == page["image_version"] and e.get("page_start") == page["page"]]
            if previous:
                additions.extend(previous)
                metadata["transcription_reused"] = True
                continue
            try:
                result = await tasks.ask("visual_transcription", "逐字转录这一论文扫描页。保留表格行列、数值单位、图注和公式；不解读趋势、不补齐看不清的内容。不清楚的部分记录 uncertainties。没有可辨文字时 readable=false。", {
                    "paper_id": paper_id, "page": page["page"], "image_version": page["image_version"],
                }, Transcription, images=[page["url"]])
                metadata["uncertainties"] = result.uncertainties
                if not result.readable or not result.text.strip():
                    raise ValueError("扫描页无法可靠转录")
                for offset in range(0, len(result.text), 4000):
                    text = result.text[offset:offset + 4000]
                    version = version_of([page["source_version"], page["image_version"], text, offset, result.uncertainties])
                    additions.append(EvidenceSpan(id=f"{paper_id}:visual:{page['page']}:{offset}@{version}",
                        paper_id=paper_id, version=version, text=text, source_level="full_text",
                        page_start=page["page"], page_end=page["page"], section="扫描页视觉转录，须回看原图",
                        modality="visual_transcription", image_version=page["image_version"], transcription_uncertainties=result.uncertainties))
            except Exception as exc:
                metadata["error"] = str(exc)
                audit["warnings"].append(f"{paper_id} 第 {page['page']} 页转录未完成：{exc}")
        if additions:
            evidence = [e for e in evidence if e.source_level != "abstract"] + additions
    except Exception as exc:
        audit.update(status="unavailable")
        audit["warnings"].append(f"{paper_id} 视觉读取不可用：{exc}")
    tasks.visual_audit.append(audit)
    return evidence, audit


async def _transcribe_crop(tasks, paper_id, page, label):
    bbox = page["object_bboxes"][label]
    data = await asyncio.to_thread(_crop_png, page, bbox)
    crop_version = hashlib.sha256(data).hexdigest()
    object_id = f"{paper_id}:visual-object:{label}"
    previous = []
    for value in getattr(tasks, "previous_visual_evidence", []):
        if (value.get("crop_version") == crop_version and value.get("image_version") == page["image_version"]
                and value.get("crop_source_version") == page["source_version"]
                and value.get("page_start") == page["page"] and value.get("visual_object_ids") == [object_id]):
            previous.append(EvidenceSpan.model_validate(value))
    if previous:
        return previous, True
    result = await tasks.ask("visual_transcription", "逐字转录此图表裁剪中的图例、坐标轴、数值、单位和表格行列。"
        "不可辨内容记入 uncertainties，无法读取时 readable=false。", {
            "paper_id": paper_id, "page": page["page"], "object_label": label, "bbox": bbox,
            "crop_version": crop_version}, Transcription,
        images=["data:image/png;base64," + base64.b64encode(data).decode()])
    if not result.readable or not result.text.strip():
        raise ValueError("裁剪内没有可靠文字")
    version = version_of([page["source_version"], page["page"], bbox, crop_version, result.model_dump()])
    spans = []
    for offset in range(0, len(result.text), 4000):
        spans.append(EvidenceSpan(id=f"{object_id}:ocr:{offset}@{version}", paper_id=paper_id, version=version,
            text=result.text[offset:offset+4000], source_level="full_text", page_start=page["page"], page_end=page["page"],
            section=f"{label} 图表转录", modality="visual_transcription", view_type="ocr",
            visual_object_ids=[object_id], crop_version=crop_version, crop_source_version=page["source_version"],
            image_version=page["image_version"], transcription_uncertainties=result.uncertainties))
    return spans, False


def verification_images(tasks, raw, target):
    images, context, missing, requested = [], [], [], []
    target_text = getattr(target, "statement", getattr(target, "rationale", ""))
    core_ids = set(target.evidence_ids)
    core = [s for s in raw if s.id in core_ids]
    for span in core:
        catalog = tasks.visual_catalog.get(span.paper_id, [])
        if span.crop_version:
            requested.append((span, span.page_start))
            continue
        references = labels(span.text + " " + target_text)
        resolved = {p['page'] for p in catalog if references.intersection(p['captions'])}
        local = range(span.page_start, (span.page_end or span.page_start) + 1) if span.page_start else []
        if resolved:
            requested.extend((span, n) for n in sorted(resolved))
        if span.modality == 'visual_transcription' or any(p['page'] in local and p.get('visual_content', bool(p.get('captions'))) for p in catalog) or (not resolved and VISUAL.search(span.text + " " + target_text)):
            requested.extend((span, n) for n in (list(local) or [None]))
    seen = set()
    for span, number in requested:
        key = (span.paper_id, number)
        page = tasks.visual_pages.get(key)
        if not page or (span.crop_source_version and span.crop_source_version != page.get("source_version")) or (span.image_version and number == span.page_start and page['image_version'] != span.image_version):
            missing.append(f"{span.paper_id} 第 {number or '未知'} 页未提供匹配原图")
            continue
        if key in seen:
            continue
        seen.add(key)
        if len(images) >= 4:
            missing.append(f"{span.paper_id} 第 {number} 页超出本次原图预算")
            continue
        images.append(page['url'])
        context.append({"paper_id": span.paper_id, "page": number, "image_version": page['image_version'], "source_version": page['source_version']})
    return images, context, missing


def original_page_png(path, number, source_version):
    """Render a display copy of the version catalogued during reading."""
    import pypdfium2 as pdfium

    with Path(path).open('rb') as stream:
        data = stream.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != source_version:
        raise ValueError('原 PDF 已变化，无法展示匹配的原页')
    with _LOCK, pdfium.PdfDocument(data) as document, closing(document[number - 1]) as page:
        with closing(page.render(scale=min(2, 1800 / max(page.get_size())))) as bitmap:
            output = io.BytesIO()
            bitmap.to_pil().save(output, format='PNG')
            return output.getvalue()


async def attach_reading_figures(tasks, result, evidence, source_path, store):
    """Resolve proposed figures against caption-bearing primary spans and PDF catalog."""
    from urllib.parse import quote

    pid = result['paper_id']
    catalog = {p['page']: p for p in tasks.visual_catalog.get(pid, [])}
    spans = {e.id: e for e in evidence if e.paper_id == pid and e.source_level == 'full_text'}
    assets = {}
    figures = []
    approved = {c['index'] for c in result.get('checks', []) if c['target_type'] == 'reading_passage'
                and c['verdict'] in {'supported', 'qualified'}}
    for index, passage in enumerate(result.get('reading_guide', [])):
        n = passage.get('figure_number')
        if not n or index not in approved:
            continue
        pattern = re.compile(rf'^\s*(?:fig(?:ure)?\.?|图)\s*{int(n)}(?!\d)\s*[|.:：—\-]\s*(.*)$', re.I | re.M)
        for eid in passage['evidence_ids']:
            span = spans.get(eid)
            if not span or not span.page_start or span.page_end not in (None, span.page_start):
                continue
            match = pattern.search(span.text)
            if not match:
                continue
            page = span.page_start
            figure = dict(passage_index=index, number=n, page=page, caption_excerpt=match.group(0).strip()[:220],
                          evidence_id=eid)
            meta = catalog.get(page)
            if page not in assets and len(assets) < 3 and meta and store.session and store.repo:
                prepared = tasks.visual_pages.get((pid, page))
                try:
                    if prepared:
                        data = base64.b64decode(prepared['url'].split(',', 1)[1])
                    elif source_path:
                        data = await asyncio.to_thread(original_page_png, source_path, page, meta['source_version'])
                    else:
                        data = None
                    if data:
                        asset = await store.save('research_figure_page', data, name=f'page-{page}.png')
                        assets[page] = f"/api/sessions/{quote(store.session, safe='')}/artifacts/{quote(asset['id'], safe='')}"
                except (OSError, ValueError, RuntimeError) as exc:
                    result['warnings'].append(f'图 {n} 原页未能展示：{exc}')
            if page in assets:
                figure['url'] = assets[page]
            figures.append(figure)
            break
    result['reading_figures'] = figures
