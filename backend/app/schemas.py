"""项目通用 Pydantic 数据结构（API 的“合同”）。

为什么用 Pydantic？
- FastAPI 会自动把请求体 / 响应体与这些模型对齐，做校验并在 `/docs` 里生成文档。
- `Literal` 类型让“只能是某几个字符串”的配置在运行时就被拦住，尽早暴露错误。

阅读建议：
- 先看 `Chunk` → `RetrievedChunk`：理解入库与检索阶段的“段落对象”有什么区别（后者多了各路分数）。
- 再看 `ChatRequest` / `ChatResponse`：理解前端和后端一问一答的 JSON 契约。
"""

from __future__ import annotations

from typing import Any, List, Optional, Dict, Literal
from datetime import datetime

from pydantic import BaseModel, Field


# =========================================================
# 文档相关
# =========================================================

class DocumentMeta(BaseModel):
    """单份已上传文件的元数据，持久化在 `backend/app/storage/metadata.json`。

    status 生命周期示意：pending → indexing → indexed（失败则 failed + error）。
    """

    document_id: str
    filename: str
    file_type: str
    upload_time: str
    source_path: str
    chunk_count: int = 0
    status: Literal["pending", "indexing", "indexed", "failed"] = "pending"
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """`POST /api/documents/upload` 成功后的响应体。"""
    document_id: str
    filename: str
    chunk_count: int
    status: str
    message: str = "ok"


class DocumentListResponse(BaseModel):
    """`GET /api/documents` 返回的列表包装。"""

    total: int
    documents: List[DocumentMeta]


# =========================================================
# Chunk
# =========================================================

class Chunk(BaseModel):
    """切分后的最小检索单元（写入 Qdrant 前的“半成品”）。

    - `chunk_id`：全局唯一，用于引用溯源与用户界面展示。
    - `chunk_index`：同一文档内从 0 递增，方便按顺序拼回全文（例如版本对比）。
    """

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    start_char: int = 0
    end_char: int = 0


class RetrievedChunk(BaseModel):
    """检索阶段返回的一条命中结果。

    `vector_score`：向量相似度打分（本项目用 Cosine，`normalize_embeddings=True` 时等价于“越接近越好”）。
    `bm25_score`：BM25 传统关键词排序的原始相关度（只做内部融合，阈值解释见 `citation_service`）。
    `final_score`：`retriever.py` 里把两路分数归一化后加权融合的结果——**这是对外的主要排序依据**。
    """

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0


# =========================================================
# 引用
# =========================================================

class Citation(BaseModel):
    """给前端展示的一条引用：`filename` / `chunk_id` / 截取后的原文 snippet / 置信相关分数。"""

    filename: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float


# =========================================================
# Chat
# =========================================================

ConfidenceLevel = Literal[
    "high",
    "medium",
    "low",
]  # 三档置信度，由检索 top1 的融合分与阈值共同决定（见 citation_service）

ChatMode = Literal[
    "rag",
    "agent",
    "retrieve",
]  # 当前实现里主要是为了前后端对齐字段；路由真正逻辑在各自的 endpoint
IntentType = Literal[
    "policy_qa",
    "eligibility_check",
    "checklist_generation",
    "version_compare",
    "unknown",
]


class ChatRequest(BaseModel):
    """统一聊天请求体：`/api/chat/query` / `/api/chat/agent` / `/api/chat/retrieve` 三家接口复用。

    注意：
    - `top_k` 为 `None` 时，服务端会回落到 `settings.top_k`。
    - Agent 版本对比需要 `old_document_id` + `new_document_id`（来自左栏文档列表里的 id）。
    """

    question: str = Field(..., description="用户问题")
    mode: ChatMode = "rag"
    top_k: Optional[int] = None
    user_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="用于 eligibility_check 等工具，例如 {年级:'研二', 成绩排名:'前20%'}",
    )
    old_document_id: Optional[str] = Field(
        default=None, description="version_compare 模式下的旧版文档 ID"
    )
    new_document_id: Optional[str] = Field(
        default=None, description="version_compare 模式下的新版文档 ID"
    )


class ChatResponse(BaseModel):
    """统一聊天响应：`answer` 给人看，`citations` 给机器/界面做溯源，`tool_result` 给 Agent 调试用。

    `refused=True` 表示系统主动拒答——要么没检索结果，要么 top1 分数太低（防幻觉）。
    """

    answer: str
    citations: List[Citation] = []
    confidence: ConfidenceLevel = "low"
    refused: bool = False
    intent: Optional[IntentType] = None
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class RetrieveResponse(BaseModel):
    """仅检索模式的响应：不进行 LLM 生成，直接把混合检索的中间结果透出，便于你调分数融合。"""

    query: str
    total: int
    chunks: List[RetrievedChunk]


class HealthResponse(BaseModel):
    """轻量心跳：运维脚本或 Streamlit 侧栏用它来确认 Qdrant / 后端是否活着。"""

    status: str = "ok"
    service: str = "school-policy-agent"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    qdrant_connected: bool = False
    document_count: int = 0
    chunk_count: int = 0
