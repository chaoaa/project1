"""BM25Okapi + jieba：`rank-bm25` 负责统计，`jieba` 把连续汉字切成词。

直觉理解：
- 向量检索更像“意思差不多就行”（语义泛化）。
- BM25 更像“必须出现关键 policy 词”（字面匹配）。

工程折中：Hybrid = 两者互补，尤其适合政策里大量专有名词、编号的场景。

实现层面的取舍：
- 全局单例索引，进程内热更新。
- 每次文档入库（或你希望同步时），调用 rebuild_from_chunks 全量重建（实现简单）。
- 索引仅驻内存——演示项目够用；上生产可改用磁盘 KV 存 token 列表。
"""
from __future__ import annotations

from typing import List, Dict
from threading import Lock

import jieba
from rank_bm25 import BM25Okapi

from app.schemas import RetrievedChunk
from app.utils.logger import logger


class BM25Store:
    """极简倒排索引：`rebuild_from_chunks` O(n) 重新统计，`keyword_search` O(n) 打分。

    复杂度提醒：当 chunk 达到百万级时你需要持久化 + 增量索引；校招 demo 规模下全量重建完全 OK。
    """

    _instance: "BM25Store | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: List[RetrievedChunk] = []
        self._tokenized_corpus: List[List[str]] = []

    @classmethod
    def get_instance(cls) -> "BM25Store":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        # 去掉非常用空白
        return [t for t in jieba.lcut(text) if t.strip()]

    def rebuild_from_chunks(self, chunks: List[RetrievedChunk]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._bm25 = None
            logger.info(f"BM25 index rebuilt with {len(self._chunks)} chunks")

    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    def keyword_search(self, query: str, top_k: int = 6) -> List[RetrievedChunk]:
        if not self.is_ready() or not query.strip():
            return []
        assert self._bm25 is not None
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # 取 top_k 索引
        idx_score = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        out: List[RetrievedChunk] = []
        for idx, score in idx_score:
            if score <= 0:
                continue
            base = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    filename=base.filename,
                    chunk_index=base.chunk_index,
                    text=base.text,
                    bm25_score=float(score),
                )
            )
        return out

    def stats(self) -> Dict[str, int]:
        return {"chunk_count": len(self._chunks)}


def get_bm25_store() -> BM25Store:
    return BM25Store.get_instance()
