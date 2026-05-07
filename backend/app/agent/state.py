"""LangGraph Agent 状态定义（全流程共享的“白板”）。

TypedDict 用在这是为了：
- 静态类型检查器（pyright/mypy）能提示你缺字段。
- 读起来像结构体，方便新人建立心智模型。

关于 `Annotated[List[str], operator.add]`：
- LangGraph 在每个节点返回 `{"messages": ["a"]}` 这类补丁时，需要知道如何把它 merge 进旧 state。
- operator.add 等价于 list 拼接：旧 messages + 新 messages。
- 不加 reducer 的字段默认是**覆盖**语义。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

from app.schemas import RetrievedChunk, Citation, IntentType, ConfidenceLevel


class AgentState(TypedDict, total=False):
    """Agent 全局状态。

    使用 total=False 是因为各节点只更新自己关心的字段。
    """

    # 输入
    question: str
    user_profile: Optional[Dict[str, Any]]
    old_document_id: Optional[str]
    new_document_id: Optional[str]

    # 中间产物
    intent: IntentType
    retrieved_chunks: List[RetrievedChunk]
    tool_name: Optional[str]
    tool_result: Optional[Dict[str, Any]]

    # 输出
    answer: str
    citations: List[Citation]
    confidence: ConfidenceLevel
    refused: bool

    # 调试
    messages: Annotated[List[str], operator.add]
