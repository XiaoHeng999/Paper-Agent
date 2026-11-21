from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

JsonObject = dict[str, Any]
AgentRole = Literal["search", "read"]

@dataclass(slots=True)
class ResearchRequest:
    """论文研究请求：逐篇报告、可选比较或综述。"""
    topic: str
    constraints: JsonObject = field(default_factory=dict)
    language: str = "zh"

# 保留请求类型的旧导入名称；所有请求均执行同一新架构。
ReviewRequest = ResearchRequest
