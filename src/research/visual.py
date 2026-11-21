"""Bounded, paper-scoped PDF rendering and original-page visual verification.

Page previews used by reading passages can be saved as session PNG artifacts.
PDFium is serialized because the native library is not thread safe.
"""
import asyncio
import base64
import hashlib
import io
import re
import threading
from contextlib import closing
from pathlib import Path

from pydantic import Field
from .models import EvidenceSpan, Record, version_of

_LOCK = threading.RLock()
VISUAL = re.compile(r"\b(?:fig(?:ure)?\.?|table|chart|plot)\s*\d*|图\s*\d|表\s*\d", re.I)


class Transcription(Record):
    text: str = Field(max_length=24000)
    readable: bool
    uncertainties: list[str] = Field(default_factory=list)


def labels(text):
    matches = re.findall(r"(?i)\b(fig(?:ure)?\.?|table)\s*(\d+)|([图表])\s*(\d+)", text)
    return {("table" if (a.lower() == "table" or c == "表") else "figure") + ":" + (b or d) for a, b, c, d in matches}


def inspect_pdf(path, limit, query=""):
    import pypdfium2 as pdfium
    from pypdfium2 import raw
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("视觉 PDF 超过 50 MB 限制")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates, selected = [], []
    wanted = labels(query)
    with _LOCK, pdfium.PdfDocument(path) as document:
        for index in range(min(len(document), 500)):
            with closing(document[index]) as page:
                with closing(page.get_textpage()) as textpage:
                    text = textpage.get_text_range()
                width, height = page.get_size()
                image_area = 0
                for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE]):
                    left, bottom, right, top = obj.get_bounds()
                    image_area += max(0, right-left) * max(0, top-bottom)
                # Full-page raster plus a searchable header is still a scan.
                scanned = len(text.strip()) < 40 or image_area >= width * height * .6
                captions = set()
                for line in text.splitlines():
                    if re.match(r"(?i)^\s*(fig(?:ure)?\.?|table|图|表)\s*\d+", line):
                        captions.update(labels(line))
                if not scanned and not image_area and not VISUAL.search(text):
                    continue
                candidates.append({"page": index+1, "scanned": scanned, "captions": sorted(captions), "visual_content": bool(scanned or image_area or captions),
                    "source_version": source_hash,
                    "priority": (int(bool(captions & wanted)), int(bool(re.search(r"(?i)ablation|benchmark|speedup|latency|实验|消融|加速", text)) and bool(captions)), int(scanned), int(bool(captions)), int(image_area > 0))})
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


async def prepare_visual(tasks, item, evidence):
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
        pages, omitted, catalog = await asyncio.to_thread(inspect_pdf, path, min(tasks.options.visual_page_limit, fair_limit), getattr(tasks, "question", ""))
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
        evidence = updated
        additions = []
        for page in pages:
            metadata = {k: v for k, v in page.items() if k != "url"}
            audit["pages"].append(metadata)
            tasks.visual_pages[(paper_id, page["page"])] = page
            if not page["scanned"]:
                continue
            previous = [EvidenceSpan.model_validate(e) for e in getattr(tasks, "previous_visual_evidence", [])
                if e.get("paper_id") == paper_id and e.get("modality") == "visual_transcription"
                and e.get("image_version") == page["image_version"] and e.get("page_start") == page["page"]]
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


def verification_images(tasks, raw, target):
    images, context, missing, requested = [], [], [], []
    target_text = getattr(target, "statement", getattr(target, "rationale", ""))
    core_ids = set(target.evidence_ids)
    core = [s for s in raw if s.id in core_ids]
    for span in core:
        catalog = tasks.visual_catalog.get(span.paper_id, [])
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
        if key in seen:
            continue
        seen.add(key)
        page = tasks.visual_pages.get(key)
        if not page or len(images) >= 4 or (span.image_version and number == span.page_start and page['image_version'] != span.image_version):
            missing.append(f"{span.paper_id} 第 {number or '未知'} 页未提供匹配原图")
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
