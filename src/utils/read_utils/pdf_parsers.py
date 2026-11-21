from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ParsedPdfPage:
    """保存 PDF 单页解析后的文字。

    中文注释：这里只放最基础的页码和正文。后面如果接入 PyMuPDF、OCR 或公式解析，
    可以在 metadata 里补更多信息，不需要改阅读节点的主流程。
    """

    page_number: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PdfParseResult:
    """保存一次 PDF 解析的统一结果。

    中文注释：不管底层用 pypdf、PyMuPDF 还是更高级的解析器，最终都整理成
    pages + warnings。调用方只关心这个统一形状。
    """

    pages: list[ParsedPdfPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BasePdfParser(ABC):
    """PDF 解析器基类。

    中文注释：这个类只规定“输入一个 PDF 文件，输出一组页文字”。底层具体怎么读
    文件由子类决定，这样以后加 PyMuPDF 时不会搅乱现在的 pypdf 实现。
    """

    name = "base"

    @abstractmethod
    def parse(self, source_path: Path) -> PdfParseResult:
        """读取 PDF 并返回统一格式的结果。"""


class PypdfParser(BasePdfParser):
    """用 pypdf 读取 PDF 正文的基础解析器。"""

    name = "pypdf"

    def parse(self, source_path: Path) -> PdfParseResult:
        """用 pypdf 逐页提取文字。

        中文注释：pypdf 对扫描版 PDF、复杂公式和双栏排版不一定理想，但它安装简单、
        速度快，适合作为当前默认方案。保留所有页的摘要、参考文献、附录及重复文字。
        """

        try:
            from pypdf import PdfReader
        except ImportError:
            return PdfParseResult(warnings=["未安装 pypdf，暂时无法读取 PDF 正文"])
        try:
            reader = PdfReader(str(source_path))
            raw_pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            return PdfParseResult(warnings=[f"PDF 正文读取失败：{exc}"])
        if not any(page.strip() for page in raw_pages):
            return PdfParseResult(warnings=["PDF 中没有可读取的文字，可能是扫描文件"])
        pages = [
            ParsedPdfPage(page_number=index, text=_normalise_text(text), metadata={"parser": self.name})
            for index, text in enumerate(raw_pages, start=1)
        ]
        return PdfParseResult(pages=pages)


def get_pdf_parser(name: str = "pypdf") -> BasePdfParser:
    """按名字返回 PDF 解析器。

    中文注释：现在只实现 pypdf。这里留一个很小的选择入口，后续新增 PyMuPDF 时
    只需要在这里加一个分支。
    """

    if name == "pypdf":
        return PypdfParser()
    raise ValueError(f"未知 PDF 解析器：{name}")


def _normalise_text(value: str) -> str:
    """整理 PDF 抽取出来的文字。

    中文注释：删除空字符，压缩过多空行。这里不做激进改写，避免误伤公式、标题和
    表格说明。
    """

    text = value.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
