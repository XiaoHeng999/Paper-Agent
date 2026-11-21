from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


@dataclass(slots=True)
class ReadNote:
    """保存仅根据标题和摘要整理出的论文笔记。"""

    main_question: str = ""
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    contributions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    main_results: list[str] = field(default_factory=list)
    short_summary: str = ""
    # 中文说明：证据等级由程序根据是否提供摘要确定，不再要求模型额外判断。
    evidence_level: str = "metadata"

    def to_dict(self) -> JsonObject:
        """把笔记转换为普通字典，方便写入状态和 JSON 文件。"""

        return asdict(self)

# 中文说明：精读名额状态。discarded 表示被 AI 判定与主题无关而软丢弃，
# 不进下游分析/写作，但笔记照常落盘可审计。
READ_RELEVANCE_STATUSES = {
    "pending",
    "selected_for_deep_read",
    "not_selected_due_to_limit",
    "discarded",
}


@dataclass(slots=True)
class ReadRelevance:
    """保存模型对单篇论文的二元相关性判断与精读名额状态。

    中文注释：relevant 是模型"内部三维分析后给出的唯一结论"；reason 是判断依据，
    用于前端展示和事后审计。discarded 标记软丢弃（不物理删除）；uncertain 标记
    保底回捞回来的"相关性存疑"论文。
    """

    relevant: bool = True
    reason: str = ""
    discarded: bool = False
    uncertain: bool = False
    status: str = "pending"

    def to_dict(self) -> JsonObject:
        """把相关性判断转换为普通字典。"""

        return asdict(self)


@dataclass(slots=True)
class FullTextStatus:
    """保存全文下载、转换和索引的当前结果。"""

    status: str = "not_requested"
    reason: str = ""
    source_url: str | None = None
    source_path: str | None = None
    markdown_path: str | None = None
    page_count: int | None = None
    chunk_count: int = 0

    def to_dict(self) -> JsonObject:
        """把全文处理状态转换为普通字典。"""

        return asdict(self)


@dataclass(slots=True)
class PaperReadResult:
    """保存一篇论文从摘要阅读到全文入库的完整结果。"""

    paper: PaperDocument
    note: ReadNote = field(default_factory=ReadNote)
    relevance: ReadRelevance = field(default_factory=ReadRelevance)
    full_text: FullTextStatus = field(default_factory=FullTextStatus)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonObject:
        """把单篇阅读结果转成可序列化字典。"""

        return {
            "paper": self.paper.to_dict(),
            "note": self.note.to_dict(),
            "relevance": self.relevance.to_dict(),
            "full_text": self.full_text.to_dict(),
            "warnings": list(self.warnings),
        }
