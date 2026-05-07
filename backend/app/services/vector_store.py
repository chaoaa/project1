"""Qdrant 向量存储封装。"""

from __future__ import annotations

import uuid
from typing import List, Dict, Any, Optional
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.schemas import Chunk, RetrievedChunk
from app.services.embedding_service import get_embedding_service
from app.utils.logger import logger


class VectorStore:
    """Qdrant 的一切：建表、upsert、向量检索、scroll 拉全量。

    概念备忘（超简版）：
    - **Collection**：类比“一张表”，里面存很多 **Point**。
    - **Vector**：Point 上真正参与 ANN 近邻搜索(float 数组)。
    - **Payload**：随向量一起存 JSON 元数据（我们放原文、chunk_id……），搜索完原样取回。

    为什么 search 里 `id=str(uuid.uuid4())`？
    Qdrant 只要求 id 唯一即可；业务主键放到 payload 的 `chunk_id` 里，方便你阅读与对账。
    """

    _instance: "VectorStore | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=20.0,
        )
        logger.info(
            f"qdrant client created: {settings.qdrant_host}:{settings.qdrant_port}"
        )

    @classmethod
    def get_instance(cls) -> "VectorStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ---------------- collection ----------------

    def ensure_collection(self, dim: int) -> None:
        """如果 collection 不存在则创建。"""
        try:
            existed = self.client.collection_exists(self.collection)
        except Exception as e:
            logger.error(f"qdrant 连接失败: {e}")
            raise

        if existed:
            return

        logger.info(f"creating qdrant collection: {self.collection} (dim={dim})")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,  # 与 sentence-transformers normalize 后的点积语义一致
            ),
        )

    def is_connected(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def count(self) -> int:
        try:
            res = self.client.count(self.collection, exact=True)
            return int(res.count)
        except Exception:
            return 0

    # ---------------- write ----------------

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """对一组 chunk 计算 embedding 并写入 qdrant。"""
        if not chunks:
            return 0

        emb = get_embedding_service()
        self.ensure_collection(emb.dimension)

        vectors = emb.encode_texts([c.text for c in chunks])
        points: List[qmodels.PointStruct] = []
        for c, v in zip(chunks, vectors):
            payload: Dict[str, Any] = {
                "document_id": c.document_id,
                "filename": c.filename,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),  # qdrant 内部 id
                    vector=v,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        logger.info(f"upserted {len(points)} points -> {self.collection}")
        return len(points)

    def delete_by_document(self, document_id: str) -> int:
        """删除某个文档的全部向量。"""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info(f"deleted vectors of {document_id}")
            return 1
        except Exception as e:
            logger.warning(f"delete vectors failed for {document_id}: {e}")
            return 0

    # ---------------- read ----------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 6,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """向量检索，返回 RetrievedChunk 列表。"""
        if not self.is_connected():
            return []

        flt: Optional[qmodels.Filter] = None
        if document_ids:
            flt = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=document_ids),
                    )
                ]
            )

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=flt,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"qdrant search failed: {e}")
            return []

        out: List[RetrievedChunk] = []
        for r in results:
            payload = r.payload or {}
            out.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    document_id=payload.get("document_id", ""),
                    filename=payload.get("filename", ""),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=payload.get("text", ""),
                    vector_score=float(r.score or 0.0),
                )
            )
        return out

    def fetch_all_chunks(self) -> List[RetrievedChunk]:
        """`scroll`（游标翻页）读出整个 collection ——工程上比 `limit=超大` 更安全。

        BM25 必须在内存里拥有一份“全文倒排”，所以需要定期全量同步。
        """
        if not self.is_connected():
            return []
        try:
            self.ensure_collection(get_embedding_service().dimension)
        except Exception:
            pass

        all_chunks: List[RetrievedChunk] = []
        next_offset = None
        try:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_offset,
                )
                for p in points:
                    payload = p.payload or {}
                    all_chunks.append(
                        RetrievedChunk(
                            chunk_id=payload.get("chunk_id", ""),
                            document_id=payload.get("document_id", ""),
                            filename=payload.get("filename", ""),
                            chunk_index=int(payload.get("chunk_index", 0)),
                            text=payload.get("text", ""),
                        )
                    )
                if not next_offset:
                    break
        except Exception as e:
            logger.warning(f"fetch_all_chunks failed: {e}")
        return all_chunks

    def fetch_chunks_by_document(self, document_id: str) -> List[RetrievedChunk]:
        """获取某个文档的全部 chunk，用于版本对比等。"""
        if not self.is_connected():
            return []
        all_chunks: List[RetrievedChunk] = []
        next_offset = None
        try:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_offset,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    ),
                )
                for p in points:
                    payload = p.payload or {}
                    all_chunks.append(
                        RetrievedChunk(
                            chunk_id=payload.get("chunk_id", ""),
                            document_id=payload.get("document_id", ""),
                            filename=payload.get("filename", ""),
                            chunk_index=int(payload.get("chunk_index", 0)),
                            text=payload.get("text", ""),
                        )
                    )
                if not next_offset:
                    break
        except Exception as e:
            logger.warning(f"fetch_chunks_by_document failed: {e}")
        all_chunks.sort(key=lambda x: x.chunk_index)
        return all_chunks


def get_vector_store() -> VectorStore:
    return VectorStore.get_instance()
