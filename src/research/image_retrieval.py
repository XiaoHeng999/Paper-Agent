"""Optional task-local CLIP retrieval over versioned, prepared PDF crops."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

from PIL import Image, ImageOps

from .multimodal import find_object_labels

_REVISION = "Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523"
_FILES = {
    "text_model_quantized.onnx": "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
    "vision_model_quantized.onnx": "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
    "tokenizer.json": "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
    "preprocessor_config.json": "6f638fb9401a6d6296feff533ee7efe657b787c49f954f82f5906b36ef2a1b1f",
}
_APPEARANCE = re.compile(r"\b(?:curves?|bar charts?|scatter(?:plots?)?|heatmaps?|dashed|dotted|red|blue)\b|颜色|曲线|虚线|柱状|散点|热图", re.I)


def visual_gap(query, packet, scope):
    if query.source != "user" or find_object_labels(query.question):
        return False
    features = _APPEARANCE.findall(query.question.casefold())
    text = " ".join(s.text.casefold() for s in scope.read(packet.evidence_ids))
    return bool({"figure", "table"} & set(packet.missing_requirements)
                or any(feature not in text for feature in features))


class ClipIndex:
    def __init__(self):
        directory = os.environ.get("PAPER_AGENT_CLIP_DIR")
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        self.directory = Path(directory) if directory else cache / "paper-agent" / "clip-vit-base-patch32" / _REVISION.split("@")[1]
        self.lock = asyncio.Lock()
        self.vectors = {}
        self.sessions = None
        self.audit = []
        self.error = ""

    def signature(self):
        paths = [self.directory / name for name in _FILES]
        return [_REVISION, [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()]]

    def _load(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        if not self.directory.is_dir():
            raise ValueError("请配置 PAPER_AGENT_CLIP_DIR 指向本地 CLIP 模型目录")
        for name, digest in _FILES.items():
            with (self.directory / name).open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError(f"CLIP 模型文件版本不匹配 {name}")
        self.processor = json.loads((self.directory / "preprocessor_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=77)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.sessions = [ort.InferenceSession(str(self.directory / name), options,
            providers=["CPUExecutionProvider"]) for name in ("text_model_quantized.onnx", "vision_model_quantized.onnx")]

    def _rank(self, question, candidates):
        import numpy as np
        from .visual import _crop_png
        if not self.sessions:
            self._load()
        tokens = np.asarray([self.tokenizer.encode(question).ids], dtype=np.int64)
        text = self.sessions[0].run(["text_embeds"], {"input_ids": tokens})[0][0]
        text = text / np.linalg.norm(text)
        scores = []
        encoded = 0
        for oid, page, label in candidates:
            key = (oid, page["page"], page["source_version"], page["image_version"], tuple(page["object_bboxes"][label]))
            if key not in self.vectors:
                png = _crop_png(page, label)
                with Image.open(io.BytesIO(png)) as image:
                    crop = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
                    pixels = np.asarray(crop, dtype=np.float32) * self.processor["rescale_factor"]
                pixels = ((pixels - self.processor["image_mean"])/self.processor["image_std"]).astype(np.float32)
                vector = self.sessions[1].run(["image_embeds"], {"pixel_values": pixels.transpose(2,0,1)[None]})[0][0]
                self.vectors[key] = vector / np.linalg.norm(vector)
                encoded += 1
            score = float(text @ self.vectors[key])
            if np.isfinite(score):
                scores.append((oid, score))
        return sorted(scores, key=lambda v: (-v[1], v[0]))[:2], encoded

    async def search(self, query, scope, pages):
        views = {}
        for span in scope.spans.values():
            for oid in span.visual_object_ids:
                if oid not in views or span.view_type == "caption":
                    views[oid] = span
        candidates = []
        for (pid, _), page in pages.items():
            if pid not in scope.allowed:
                continue
            for label in page.get("pixel_bboxes", {}):
                oid = f"{pid}:visual-object:{label}"
                if oid in views:
                    candidates.append((oid, page, label))
        counts = {}
        for oid, _, _ in candidates:
            counts[oid] = counts.get(oid, 0) + 1
        candidates = [item for item in candidates if counts[item[0]] == 1]
        if not candidates:
            return []
        async with self.lock:
            if self.error:
                raise ValueError(self.error)
            started = time.monotonic()
            try:
                hits, encoded = await asyncio.to_thread(self._rank, query.question, candidates[:8])
            except Exception as exc:
                self.error = str(exc)
                raise ValueError(self.error) from exc
            self.audit.append({"subquery_id": query.id, "model": _REVISION, "encoded_crops": encoded,
                "query_embeddings": 1, "seconds": time.monotonic()-started, "hits": hits})
            return [views[oid] for oid, _ in hits]
