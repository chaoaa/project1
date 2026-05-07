"""与「磁盘 + JSON 元数据」打交道的小工具合集。

为什么不用数据库？
校招项目为了**减少部署依赖**，用 `metadata.json` 即可表达“文件级目录”。
真正的大段文本与向量则躺在 Qdrant 里——注意职责划分：
- 磁盘：原始文件 bytes
- metadata.json：文件名、状态、chunk_count …
- Qdrant：chunk 文本 + 向量
"""

from __future__ import annotations

import json
import uuid
import hashlib
from pathlib import Path
from typing import List
from datetime import datetime
from threading import Lock

from app.config import settings
from app.schemas import DocumentMeta
from app.utils.logger import logger


SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

_metadata_lock = Lock()  # 避免 Streamlit 高频轮询 + 上传并发时读写 JSON 打架


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTS


def get_file_type(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def generate_document_id(filename: str) -> str:
    """生成业务主键：`hash(文件名 + 时间戳 + 随机盐)` → 简短 `doc_xxx`。

    - 不可逆没关系，我们只要**几乎不碰撞**、`metadata.json` 里好写即可。
    - 前缀 `doc_`：日志里一眼就知这是文档级 id，而非 chunk id。
    """
    base = f"{filename}-{datetime.utcnow().isoformat()}-{uuid.uuid4().hex[:8]}"
    return "doc_" + hashlib.md5(base.encode("utf-8")).hexdigest()[:16]


def save_uploaded_file(filename: str, content: bytes) -> Path:
    """保存上传文件到 storage/uploaded_files。

    若同名文件存在，则增加时间戳避免覆盖。
    """
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / filename
    if target.exists():
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        target = upload_dir / f"{Path(filename).stem}_{ts}{Path(filename).suffix}"
    target.write_bytes(content)
    logger.info(f"saved uploaded file -> {target}")
    return target


def load_metadata() -> List[DocumentMeta]:
    """加载 metadata.json，返回 DocumentMeta 列表。"""
    path = settings.metadata_path
    if not path.exists():
        return []
    try:
        with _metadata_lock:
            raw = json.loads(path.read_text(encoding="utf-8"))
        return [DocumentMeta(**item) for item in raw]
    except Exception as e:
        logger.warning(f"load metadata failed: {e}")
        return []


def save_metadata(items: List[DocumentMeta]) -> None:
    path = settings.metadata_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _metadata_lock:
        payload = [item.model_dump() for item in items]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def upsert_metadata(meta: DocumentMeta) -> None:
    items = load_metadata()
    for i, m in enumerate(items):
        if m.document_id == meta.document_id:
            items[i] = meta
            break
    else:
        items.append(meta)
    save_metadata(items)


def get_metadata(document_id: str) -> DocumentMeta | None:
    for m in load_metadata():
        if m.document_id == document_id:
            return m
    return None
