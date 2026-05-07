"""文档上传与列表查询。

一次完整入库在做什么？（建议你边 debug 边在心里对照这条流水线）
上传字节流 → 落地磁盘 → 写入 metadata（indexing）→ 各格式解析纯文本 → `clean_text`
→ `split_text` → `EmbeddingService` 批量向量化 → 写入 Qdrant（payload 里带原文）
→ 全量刷新 BM25 → 更新 metadata（indexed）。

任何一步失败都应把 `meta.status` 置为 `failed` 并写明 `error`，让用户可追踪。
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import DocumentListResponse, DocumentMeta, UploadResponse
from app.services.document_loader import DocumentLoadError, load_document
from app.services.retriever import rebuild_bm25_from_qdrant
from app.services.text_cleaner import clean_text
from app.services.text_splitter import split_text
from app.services.vector_store import get_vector_store
from app.utils.file_utils import (
    generate_document_id,
    get_file_type,
    is_supported_file,
    load_metadata,
    save_uploaded_file,
    upsert_metadata,
)
from app.utils.logger import logger


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """接收单个文件（multipart/form-data，字段名必须是 `file`，与 curl -F 对齐）。

    `async def` + `await file.read()`：FastAPI 在遇到大文件时更高效地释放事件循环。
    """
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

    saved_path = save_uploaded_file(file.filename, content)
    document_id = generate_document_id(file.filename)
    meta = DocumentMeta(
        document_id=document_id,
        filename=file.filename,
        file_type=get_file_type(file.filename),
        upload_time=datetime.utcnow().isoformat(),
        source_path=str(saved_path),
        chunk_count=0,
        status="indexing",
    )
    # 先落库元数据——就算后面解析失败，用户也能在列表里看到失败原因。
    upsert_metadata(meta)

    try:
        raw_text = load_document(saved_path)  # 按后缀_dispatch 到 pdf/docx/html/txt...
    except DocumentLoadError as e:
        meta.status = "failed"
        meta.error = str(e)
        upsert_metadata(meta)
        raise HTTPException(status_code=400, detail=f"文档解析失败：{e}")

    cleaned = clean_text(raw_text)  # 噪声治理：页码、多空行、残余 HTML；但守住条款骨架
    if not cleaned.strip():
        meta.status = "failed"
        meta.error = "解析结果为空，可能是扫描版 PDF 或加密文档"
        upsert_metadata(meta)
        raise HTTPException(
            status_code=400,
            detail="文档解析后为空。若为扫描版 PDF，请改用文本版或后续接入 OCR。",
        )

    chunks = split_text(
        cleaned, document_id=document_id, filename=file.filename
    )  # 语义切分：按章节编号 → 段落 → 句子兜底
    if not chunks:
        meta.status = "failed"
        meta.error = "切分得到 0 个 chunk"
        upsert_metadata(meta)
        raise HTTPException(status_code=400, detail="文档切分失败，未得到有效 chunk")

    try:
        vstore = get_vector_store()
        vstore.upsert_chunks(chunks)
    except Exception as e:
        logger.error(f"upsert chunks failed: {e}")
        meta.status = "failed"
        meta.error = f"向量入库失败：{e}"
        upsert_metadata(meta)
        raise HTTPException(status_code=500, detail=f"向量入库失败：{e}")

    # BM25：纯内存索引。每次有新文档入库，最简单可靠的做法就是“遍历 Qdrant 全量 rebuild”。
    try:
        rebuild_bm25_from_qdrant()
    except Exception as e:
        logger.warning(f"rebuild bm25 failed: {e}")

    meta.chunk_count = len(chunks)
    meta.status = "indexed"
    upsert_metadata(meta)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
        status="indexed",
        message="文档已成功入库",
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
def list_documents() -> DocumentListResponse:
    """读出 JSON 形式的“图书馆目录”，不调 Qdrant（轻量）。

    文件级 `document_id` 也是 Agent 「版本对比」下拉框的数据源。
    """

    docs = load_metadata()
    return DocumentListResponse(total=len(docs), documents=docs)
