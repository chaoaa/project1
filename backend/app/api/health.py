"""`/api/health`：Compose / Streamlit / 人肉脚本用来做“活着吗？”探测的路由。

设计原则：**轻量**。不要在此处做会向 LLM 发请求的重活。

返回：
- `qdrant_connected`: 向量库端口是否连通
- `document_count`: metadata.json 里登记了多少个文件
- `chunk_count`: Qdrant collection 里的 point 总数（向量段条数）
"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.vector_store import get_vector_store
from app.utils.file_utils import load_metadata


router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    """聚合两类信息源：本地 JSON 元数据 + 远程 Qdrant 统计。"""

    docs = load_metadata()
    vstore = get_vector_store()
    connected = vstore.is_connected()
    chunk_count = vstore.count() if connected else 0
    return HealthResponse(
        qdrant_connected=connected,
        document_count=len(docs),
        chunk_count=chunk_count,
    )
