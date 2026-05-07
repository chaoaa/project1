"""Hybrid Search：向量检索 + BM25 关键词检索 融合。

融合策略：
- 各路 score 做 min-max 标准化到 [0, 1]
- final_score = 0.65 * vector_score_norm + 0.35 * bm25_score_norm
- 单路命中也保留（缺失分数视为 0）
- 按 chunk_id 去重
"""

from __future__ import annotations

from typing import Dict, List

from app.config import settings
from app.schemas import RetrievedChunk
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.bm25_store import get_bm25_store
from app.utils.logger import logger


VECTOR_WEIGHT = 0.65
BM25_WEIGHT = 0.35


def _min_max_norm(values: List[float]) -> List[float]:
    """把一串分数压到 [0, 1] 区间（Min-Max 归一）。

    为什么需要归一？BM25 raw score 的量级与向量 cosine score 完全不同，直接加权会“偏科”。
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        # 全相等：要么全 0，要么统一给 1
        return [0.0 if hi == 0 else 1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    """混合检索主入口：语义 + 关键词两路召回 → 归一化融合 → 全局排序截断。

    经验：先各取 `top_k * 2` 再融合，比只取 `top_k` 更能减少“漏召回”。
    """

    top_k = top_k or settings.top_k
    if not query or not query.strip():
        return []

    # ---- 向量检索 ----
    vstore = get_vector_store()
    try:
        qvec = get_embedding_service().encode_query(query)
        vector_hits = vstore.search(qvec, top_k=top_k * 2)
    except Exception as e:
        logger.error(f"vector search failed: {e}")
        vector_hits = []

    # ---- BM25 检索 ----
    bm25 = get_bm25_store()
    bm25_hits = bm25.keyword_search(query, top_k=top_k * 2) if bm25.is_ready() else []

    # ---- 标准化 ----
    v_scores = [h.vector_score for h in vector_hits]
    b_scores = [h.bm25_score for h in bm25_hits]
    v_norm = _min_max_norm(v_scores)
    b_norm = _min_max_norm(b_scores)

    merged: Dict[str, RetrievedChunk] = {}

    for hit, vn in zip(vector_hits, v_norm):
        key = hit.chunk_id or f"{hit.document_id}#{hit.chunk_index}"
        merged[key] = RetrievedChunk(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            text=hit.text,
            vector_score=hit.vector_score,
            bm25_score=0.0,
            final_score=VECTOR_WEIGHT * vn,
        )

    for hit, bn in zip(bm25_hits, b_norm):
        key = hit.chunk_id or f"{hit.document_id}#{hit.chunk_index}"
        if key in merged:
            existing = merged[key]
            existing.bm25_score = hit.bm25_score
            existing.final_score = (
                VECTOR_WEIGHT * _safe_norm(existing.vector_score, v_scores)
                + BM25_WEIGHT * bn
            )
        else:
            merged[key] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                text=hit.text,
                vector_score=0.0,
                bm25_score=hit.bm25_score,
                final_score=BM25_WEIGHT * bn,
            )

    ranked = sorted(merged.values(), key=lambda c: c.final_score, reverse=True)
    logger.info(
        f"hybrid_search query='{query[:30]}' "
        f"vec={len(vector_hits)} bm25={len(bm25_hits)} "
        f"merged={len(merged)} -> top {top_k}"
    )
    return ranked[:top_k]


def _safe_norm(value: float, all_values: List[float]) -> float:
    """单个值相对 `all_values` 这组查询结果再做一遍 min-max（与 zip 那套保持一致）。"""
    if not all_values:
        return 0.0
    lo, hi = min(all_values), max(all_values)
    if hi - lo < 1e-9:
        return 0.0 if hi == 0 else 1.0
    return (value - lo) / (hi - lo)


def rebuild_bm25_from_qdrant() -> int:
    """从 Qdrant 拉全量 chunk 重建 BM25 索引。"""
    vstore = get_vector_store()
    chunks = vstore.fetch_all_chunks()
    get_bm25_store().rebuild_from_chunks(chunks)
    return len(chunks)
