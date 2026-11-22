"""Task-local image retrieval over versioned, prepared PDF crops."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import re
import time
import threading
from functools import lru_cache
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
_APPEARANCE = re.compile(r"\b(?:curves?|bar charts?|scatter(?:plots?)?|heatmaps?|dashed|dotted|red|blue|green|yellow|gr[ae]y|orange|black|white|circles?|arrows?|snapshots?|photos?|photographs?|pictures?|rectangles?|arcs?|stacks?|stacked|columns?|colored|vertically|horizontally|grids?)\b|颜色|曲线|虚线|柱状|散点|热图|矩形|堆叠|网格|箭头", re.I)


def localization_type(question):
    match = re.search(r"\bwhich (figure|diagram|chart|plot|table)\b|哪(?:一)?(?:张|个|幅)?(图|表)", question, re.I)
    if match and not find_object_labels(question):
        return "table" if (match[1] or match[2]).casefold() in {"table", "表"} else "figure"
    return None


def visual_gap(query, packet, scope):
    if query.source != "user" or find_object_labels(query.question):
        return False
    features = _APPEARANCE.findall(query.question.casefold())
    spans = scope.read(packet.evidence_ids)
    kind = localization_type(query.question)
    if kind and features:
        return not any(len(s.visual_object_ids) == 1 and f":visual-object:{kind}:" in s.visual_object_ids[0]
            and all(f in s.text.casefold() for f in features) for s in spans)
    text = " ".join(s.text.casefold() for s in spans)
    return bool({"figure", "table"} & set(packet.missing_requirements)
                or any(feature not in text for feature in features))


class ImageIndex:
    def __init__(self, directory):
        self.directory = directory
        self.lock = asyncio.Lock()
        self.vectors = {}
        self.audit = []
        self.error = ""

    def signature(self):
        paths = [self.directory / name for name in self.files]
        return [self.revision, [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()]]

    async def search(self, query, scope, pages):
        kind = localization_type(query.question)
        views = {}
        for span in scope.spans.values():
            for oid in span.visual_object_ids:
                if kind and (len(span.visual_object_ids) != 1 or not oid.startswith(f"{span.paper_id}:visual-object:{kind}:")):
                    continue
                previous = views.get(oid)
                if previous is None or (span.view_type == "caption" and
                        (previous.view_type != "caption" or (len(span.visual_object_ids), len(span.text))
                         < (len(previous.visual_object_ids), len(previous.text)))):
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
            self.audit.append({"subquery_id": query.id, "model": self.revision, "encoded_crops": encoded,
                "query_embeddings": 1, "seconds": time.monotonic()-started, "hits": hits})
            return [views[oid] for oid, _ in hits]


class ClipIndex(ImageIndex):
    revision = _REVISION
    files = _FILES

    def __init__(self):
        directory = os.environ.get("PAPER_AGENT_CLIP_DIR")
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        super().__init__(Path(directory) if directory else cache / "paper-agent" / "clip-vit-base-patch32" / _REVISION.split("@")[1])
        self.sessions = None

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


_GME_FILES = {
    "added_tokens.json": "7fa54985b58718a8fdb4f4d97484c4bd908db114847675e4bf3afe3e1d5d7bd4",
    "chat_template.json": "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96",
    "config.json": "dd18f3c69154bc6ccd1f55a6e0d0831c7edde2e29fee19d7ee91ad0e2e9f24c4",
    "generation_config.json": "ff59fad29b00eab5bef61be17668b46b3eac6e12d452b85ec88ad8f8cdfd4e59",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00003.safetensors": "ce525de1d47307ebf63285ce4f2f80ca8ffffd94235b2151585e5714307e4e99",
    "model-00002-of-00003.safetensors": "7b42b76064f5d14ee9d0390f9a1f3571fc66eae4f40764f818cb5e9aafc5d6f5",
    "model-00003-of-00003.safetensors": "e3f1342fe1dd7df1582be00ebd45557316015050ae2edb841e61150825f16adc",
    "model.safetensors.index.json": "047653b243143a82b3460c187826bebca04ec93b48f0bb19109d5674a124cbca",
    "modeling_gme_qwen2vl.py": "b21b5071fcbba5ab71bfa2667c82f446c32d7d27b42148c7571c5d59f940e8c4",
    "preprocessor_config.json": "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "0b233397aeabcb441562b9186b921d6449dfc5f7bc567880400dc3662d264f02",
    "tokenizer_config.json": "c715bebf439296530bc63ac45461f3f70d6e45b561ebf02d3a21091e07b698a1",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
}
# Research tasks share the large read-only model, but keep their crop vectors separate.
_gme_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_gme(directory, fingerprint):
    import torch
    from transformers import AutoModel

    for name, digest in _GME_FILES.items():
        with (directory / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"GME 模型文件版本不匹配 {name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModel.from_pretrained(str(directory), local_files_only=True, trust_remote_code=True,
                                     torch_dtype=dtype).to(device).eval()


class GmeIndex(ImageIndex):
    revision = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct@9cfa6413f704a7c1cf5064d240748e10c876b286"
    files = _GME_FILES

    def __init__(self):
        super().__init__(Path(os.environ.get("PAPER_AGENT_GME_DIR",
            "/data/models/liveact/qwen-vl/gme-Qwen2-VL-2B-Instruct")))

    def _rank(self, question, candidates):
        from .visual import _crop_png

        if not self.directory.is_dir():
            raise ValueError("请配置 PAPER_AGENT_GME_DIR 指向本地 GME-Qwen2-VL-2B-Instruct 目录")
        with _gme_lock:
            model = _load_gme(self.directory, tuple(self.signature()[1]))
            query = model.embed([question], [None], instruction="Find an image that matches the given text.")[0].float().cpu()
            scores = []
            encoded = 0
            for oid, page, label in candidates:
                caption = page.get("caption_texts", {}).get(label, "")
                key = (oid, page["page"], page["source_version"], page["image_version"],
                       tuple(page["object_bboxes"][label]), caption)
                if key not in self.vectors:
                    with Image.open(io.BytesIO(_crop_png(page, label))) as image:
                        vector = model.embed([caption], [image.convert("RGB")], is_query=False)[0]
                    self.vectors[key] = vector.float().cpu()
                    encoded += 1
                score = float(query @ self.vectors[key])
                if not math.isfinite(score):
                    raise ValueError("GME 返回无效相似度")
                scores.append((oid, score))
            return sorted(scores, key=lambda item: (-item[1], item[0]))[:2], encoded
