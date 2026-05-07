"""引用（Citation）与置信度评估。

这里实现的是一套**很轻量但好用**的工程策略：
- **citation**：把检索结果转成前端可折叠展示的 snippet + score。
- **confidence**：只用 top1 fusion score + 阈值做三档离散化（不依赖 LLM 自评，省成本、可复现）。
- **refused**：第一道安全闸，早于 LLM，降低“无证据乱答”的概率。
"""

from __future__ import annotations

from typing import List, Tuple

from app.config import settings
from app.schemas import Citation, ConfidenceLevel, RetrievedChunk


def chunks_to_citations(
    chunks: List[RetrievedChunk],
    max_text_len: int = 240,
) -> List[Citation]:
    """把检索结果转成对外暴露的 Citation。"""
    citations: List[Citation] = []
    for c in chunks:
        snippet = c.text or ""
        if len(snippet) > max_text_len:
            snippet = snippet[:max_text_len].rstrip() + "..."
        citations.append(
            Citation(
                filename=c.filename,
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                text=snippet,
                # fusion 不存在时退回单科成绩，避免出现全 0 的前端观感
                score=round(float(c.final_score or c.vector_score or c.bm25_score), 4),
            )
        )
    return citations


def assess_confidence(
    chunks: List[RetrievedChunk],
) -> Tuple[ConfidenceLevel, bool]:
    """根据 top1 score 判断置信度，并给出是否拒答。

    返回:
        (confidence, refused)
    """
    if not chunks:
        return "low", True

    top_score = max(
        (c.final_score or c.vector_score or c.bm25_score) for c in chunks
    )

    if top_score >= settings.score_threshold_high:
        return "high", False
    if top_score >= settings.score_threshold_medium:
        return "medium", False
    return "low", True


def build_context_block(chunks: List[RetrievedChunk], max_chunks: int = 6) -> str:
    """拼接给 LLM 的上下文块，每段带编号便于引用。"""
    if not chunks:
        return "(无检索结果)"
    parts = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        parts.append(
            f"[片段 {i}] 来源文件：{c.filename} | chunk_id：{c.chunk_id}\n{c.text}"
        )
    return "\n\n".join(parts)


REFUSAL_MESSAGE = (
    "当前知识库中没有找到足够可靠的政策依据，无法直接判断。"
    "建议补充相关政策文件，或咨询学院研究生办公室。"
)
