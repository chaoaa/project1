"""文档入库公共服务：统一文件上传和 URL 采集的入库逻辑。

=============================================================================
为什么单独抽出来？（重构的动机）
=============================================================================
原先 documents.py 的 upload_document 函数里写了完整的"解析→清洗→切分→向量化"链路。
后来新增 URL 采集功能，发现 URL 入库也需要做完全一样的事。

如果不在这个节点抽取出 ingest_file()，会出现什么问题？
1. 代码重复：documents.py 和 url_ingestion.py 各写一遍同样的链路
2. 行为不一致：哪天修了 documents.py 的 bug，url_ingestion.py 可能忘记同步修复
3. 难测试：只能通过 HTTP 测试，不能直接 import 函数做单元测试

所以抽出这个 "公共服务层"——像是一个微缩的 "Clean Architecture Use Case"：
  控制器层（API Router）→ 用例层（ingest_file）→ 实体/基础设施层（loader/cleaner/splitter/vector_store）

=============================================================================
入库流程（按顺序的 7 步流水线）
=============================================================================
输入：文件路径 + 文件名
  1. 生成 document_id → 写 metadata（status="indexing"）
  2. load_document()     → 解析 PDF/DOCX/TXT 等格式，得到纯文本
  3. clean_text()        → 去页码、HTML 残留、多空行等噪声
  4. split_text()        → 按章节/段落/句子做层次化切分，生成 Chunk 列表
  5. vector_store.upsert_chunks() → 向量化后写入 Qdrant
  6. rebuild_bm25_from_qdrant()   → 全量重建进程内 BM25 索引
  7. 更新 metadata（status="indexed", chunk_count=xxx）
输出：dict {document_id, filename, chunk_count, status, message}

任何一步失败 → meta.status = "failed" + 明确的 error 消息。
这样用户在前端就能看到"为什么入库失败"，而不是一个模糊的 500 错误。
=============================================================================
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from app.schemas import DocumentMeta
from app.services.document_loader import DocumentLoadError, load_document
from app.services.retriever import rebuild_bm25_from_qdrant
from app.services.text_cleaner import clean_text
from app.services.text_splitter import split_text
from app.services.vector_store import get_vector_store
from app.utils.file_utils import (
    generate_document_id,
    get_file_type,
    upsert_metadata,
)
from app.utils.logger import logger


class IngestionError(Exception):
    """入库过程中任一环节失败时抛出的统一异常。

    为什么自定义异常？
    - 让调用方（documents.py 的路由）可以用 `except IngestionError` 精准捕获
    - 与 `DocumentLoadError`（解析失败）区分开，方便日志告警分级
    - FastAPI 的 HTTPException 不适合在服务层使用（那应该留在路由层）
    """


def ingest_file(
    file_path: Path,
    filename: str,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """统一的文档入库函数 —— 本项目所有入库操作的唯一入口。

    被以下场景复用：
    - POST /api/documents/upload      （用户从 Streamlit 上传文件）
    - POST /api/documents/ingest-url  （从 URL 采集网页政策）
    - Agent web_ingestion_node        （Agent 对话中请求采集网页）

    参数：
        file_path:  已保存在磁盘上的文件绝对路径（Path 对象）
        filename:   用于展示的文件名（如"奖学金政策.pdf"或"url_notice_20260507.txt"）
        source_url: 可选，网页来源 URL（仅 URL 采集时传入，用于追溯来源）
        title:      可选，文档标题（网页采集时从 <title> 或 <h1> 提取）

    返回：
        dict: {
            "document_id": str,   # 全局唯一文档 ID
            "filename": str,      # 文件名
            "title": str|None,    # 文档标题（如有）
            "source_url": str|None, # 来源 URL（如有）
            "chunk_count": int,   # 切分后的 chunk 数量
            "status": str,        # "indexed" 表示成功
            "message": str,       # 可读的状态消息
        }

    异常：
        IngestionError: 入库任一环节失败（解析/清洗/切分/向量化）
    """

    # ---- 第 1 步：生成 ID 并预写 metadata（status="indexing"） ----
    # 提前把 metadata 落盘的原因是：即使后续步骤失败，用户也能在列表里
    # 看到这条记录并知道失败原因（meta.error 字段）。
    document_id = generate_document_id(filename)
    meta = DocumentMeta(
        document_id=document_id,
        filename=filename,
        file_type=get_file_type(filename),
        upload_time=datetime.utcnow().isoformat(),
        source_path=str(file_path),
        chunk_count=0,
        status="indexing",  # 初始状态："正在处理中"
    )
    upsert_metadata(meta)

    # ---- 第 2 步：解析文档 → 纯文本 ----
    # load_document 内部根据文件后缀分发到不同的解析器：
    #   .pdf → pypdf, .docx → python-docx, .html → BeautifulSoup, .txt/.md → 直接读
    try:
        raw_text = load_document(file_path)
    except DocumentLoadError as e:
        meta.status = "failed"
        meta.error = str(e)
        upsert_metadata(meta)
        raise IngestionError(f"文档解析失败：{e}")

    # ---- 第 3 步：文本清洗 ----
    # 去除页码、HTML 残留、多空行等噪声，但保留条款编号和章节标题
    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        meta.status = "failed"
        meta.error = "解析结果为空，可能是扫描版 PDF 或加密文档"
        upsert_metadata(meta)
        raise IngestionError(
            "文档解析后为空。若为扫描版 PDF，请改用文本版或后续接入 OCR。"
        )

    # ---- 第 4 步：文本切分 → Chunk 列表 ----
    # 层次化切分：章节编号 → 段落（空行）→ 句子（句号），保证条款不会被腰斩
    chunks = split_text(cleaned, document_id=document_id, filename=filename)
    if not chunks:
        meta.status = "failed"
        meta.error = "切分得到 0 个 chunk"
        upsert_metadata(meta)
        raise IngestionError("文档切分失败，未得到有效 chunk")

    # ---- 第 5 步：向量化 + 写入 Qdrant ----
    # 每个 chunk 都会经过 sentence-transformers 编码为向量，再存入 Qdrant
    try:
        vstore = get_vector_store()
        vstore.upsert_chunks(chunks)
    except Exception as e:
        logger.error(f"upsert chunks failed: {e}")
        meta.status = "failed"
        meta.error = f"向量入库失败：{e}"
        upsert_metadata(meta)
        raise IngestionError(f"向量入库失败：{e}")

    # ---- 第 6 步：重建 BM25 索引 ----
    # BM25 是纯内存索引，每次有新文档入库都需要全量重建。
    # 校招 demo 规模下全量重建完全 OK（< 10000 chunks）。
    # 生产环境可改为增量索引，但复杂度会增加不少。
    try:
        rebuild_bm25_from_qdrant()
    except Exception as e:
        logger.warning(f"rebuild bm25 failed: {e}")

    # ---- 第 7 步：更新 metadata 为最终成功状态 ----
    meta.chunk_count = len(chunks)
    meta.status = "indexed"
    upsert_metadata(meta)

    logger.info(
        f"[ingestion] ingested {filename} -> {document_id} "
        f"({len(chunks)} chunks)"
        + (f" from {source_url}" if source_url else "")
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "title": title,
        "source_url": source_url,
        "chunk_count": len(chunks),
        "status": "indexed",
        "message": "文档已成功入库",
    }
