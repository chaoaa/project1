"""文档上传与列表查询。

一次完整入库在做什么？（建议你边 debug 边在心里对照这条流水线）
上传字节流 → 落地磁盘 → 写入 metadata（indexing）→ 各格式解析纯文本 → `clean_text`
→ `split_text` → `EmbeddingService` 批量向量化 → 写入 Qdrant（payload 里带原文）
→ 全量刷新 BM25 → 更新 metadata（indexed）。

任何一步失败都应把 `meta.status` 置为 `failed` 并写明 `error`，让用户可追踪。

=============================================================================
URL 入库 vs 文件上传的异同
=============================================================================
两者最终都调用 ingest_file() 完成入库，区别只在于"文件怎么来的"：
- 文件上传：用户通过 Streamlit 的 file_uploader 组件选择本地文件
- URL 采集：从用户提供的网页链接抓取 HTML → 清洗 → 保存为本地 txt → 再入库

这就是为什么要把入库逻辑抽出到 ingestion_service.ingest_file()：
router 层只负责"接收请求 + 准备文件"，service 层负责"统一入库流水线"。
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import (
    DocumentListResponse,
    UrlIngestRequest,
    UrlIngestResponse,
    UploadResponse,
)
from app.services.ingestion_service import IngestionError, ingest_file
from app.services.url_ingestion import ingest_policy_from_url
from app.utils.file_utils import (
    is_supported_file,
    load_metadata,
    save_uploaded_file,
)
from app.utils.logger import logger


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """接收单个文件（multipart/form-data，字段名必须是 `file`，与 curl -F 对齐）。

    `async def` + `await file.read()`：
    虽然是 I/O 操作（写磁盘），但 FastAPI 建议上传用 async，
    这样在等待大文件读取时不会阻塞其他请求的事件循环。
    """

    # ---- 输入校验 ----
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename 缺失")
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="仅支持 .pdf .docx .txt .md .html 等政策文件格式",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # ---- 保存到磁盘 ----
    # save_uploaded_file 会处理同名文件冲突（加时间戳后缀）
    saved_path = save_uploaded_file(file.filename, content)

    # ---- 调用统一入库逻辑 ----
    try:
        result = ingest_file(file_path=saved_path, filename=file.filename)
    except IngestionError as e:
        # IngestionError 是入库链路中任一环失败的统一定义异常
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 未预期的异常（如磁盘满、权限不足）
        logger.error(f"upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"入库失败：{e}")

    return UploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        status=result["status"],
        message=result["message"],
    )


@router.post("/ingest-url", response_model=UrlIngestResponse)
def ingest_url(req: UrlIngestRequest) -> UrlIngestResponse:
    """从 URL 采集政策网页，清洗后切分、向量化并入库。

    与 /upload 复用同一套 ingest_file 入库逻辑。
    区别在于"文件的来源"：这里是从网页抓取后保存为 txt，而非用户上传。

    【使用示例】
    curl -X POST http://localhost:8000/api/documents/ingest-url \
      -H "Content-Type: application/json" \
      -d '{"url":"https://example.edu.cn/notice/scholarship.html"}'

    【限制】
    - 仅支持 http/https 协议
    - 网页正文必须 >= 100 字符
    - JS 动态渲染的 SPA 页面可能无法正确抓取（httpx 不执行 JS）
    """

    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="url 不能为空")

    try:
        # ingest_policy_from_url 内部：
        # httpx GET → BeautifulSoup 清洗 → 保存为 txt → 调用 ingest_file 入库
        result = ingest_policy_from_url(req.url.strip())
    except ValueError as e:
        # URL 非法（协议不对、没域名等）
        raise HTTPException(status_code=400, detail=str(e))
    except IngestionError as e:
        # 网页请求失败、内容过短、入库失败等
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"ingest-url failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL 入库失败：{e}")

    return UrlIngestResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        title=result.get("title"),
        source_url=result.get("source_url"),
        chunk_count=result["chunk_count"],
        status=result["status"],
        message=result["message"],
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
def list_documents() -> DocumentListResponse:
    """读出 JSON 形式的"图书馆目录"。

    不调 Qdrant（轻量），只读本地的 metadata.json。
    文件级 `document_id` 也是 Agent「版本对比」下拉框的数据源。

    【性能说明】
    当前实现每次请求都读一次 metadata.json。在校招 demo 规模下（< 1000 个文档）
    完全没问题。如果未来文档量很大，可以加内存缓存（functools.lru_cache + TTL）。
    """

    docs = load_metadata()
    return DocumentListResponse(total=len(docs), documents=docs)
