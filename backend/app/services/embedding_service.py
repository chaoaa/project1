"""Embedding 服务封装。

抽象出统一接口，方便后续替换成在线 API（如 OpenAI / DashScope embedding）。
默认本地 sentence-transformers 加载 bge-m3。
"""

from __future__ import annotations

from typing import List
from threading import Lock

from app.config import settings
from app.utils.logger import logger


class EmbeddingService:
    """Embedding = 把文字映射到固定维度向量空间里的一个点。

    为什么写单例 + 延迟加载？
    1. 模型体积大、耗显存/内存——进程里只保留一份最省资源。
    2. 若只是健康检查 / 上传接口，也许暂时用不到模型，延迟到第一次 `encode` 再加载能加快冷启动。

    `normalize_embeddings=True`：把向量单位化，配合 Qdrant 的 Cosine 距离更稳定。
    """

    _instance: "EmbeddingService | None" = None
    _lock = Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dim: int | None = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info(f"loading embedding model: {self.model_name}")
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(
                f"加载 embedding 模型失败: {e}. "
                f"如本地无法下载，请在 .env 中切换 EMBEDDING_MODEL 为可用模型"
                f"（例如 BAAI/bge-small-zh-v1.5）。"
            )
            raise

        # 探测维度
        try:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            test_vec = self._model.encode(["dim_probe"], normalize_embeddings=True)
            self._dim = int(len(test_vec[0]))
        logger.info(
            f"embedding model ready: {self.model_name} (dim={self._dim})"
        )

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码：一次 forward 多条句子，GPU/CPU 利用率更高。

        返回 Python 原生 `list` 而不是 `numpy.ndarray`，是为了与 `qdrant-client` 直接对接。
        """
        if not texts:
            return []
        self._ensure_loaded()
        assert self._model is not None
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,  # 归一化，便于用余弦/内积
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> List[float]:
        return self.encode_texts([query])[0]


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService.get_instance()
