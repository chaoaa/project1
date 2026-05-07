"""中文政策文档 Chunk 切分器。

策略：
1. 优先按结构标记切（一、二、三 / 第X条 / 数字编号）。
2. 同一节内若过长，则按段落（空行）二次切。
3. 仍过长再按句号 / 标点切。
4. 输出按 chunk_size 控制 + chunk_overlap 重叠。
"""

from __future__ import annotations

import re
import uuid
from typing import List

from app.config import settings
from app.schemas import Chunk
from app.utils.logger import logger


# 说明：这里用的是“政策里常见的篇-章-节句式”而不是 AI 分句，优点是零依赖、确定性高。
SECTION_RE = re.compile(
    r"(?m)^(?:"
    r"[一二三四五六七八九十百零〇]+[、．\.]"  # 一、
    r"|第[一二三四五六七八九十百零〇\d]+[条章节款项]"  # 第X条/章
    r"|\d+[、．\.](?!\d)"  # 1. （但不是 1.5 之类）
    r")"
)

SENTENCE_END_RE = re.compile(r"([。！？!?；;])")


def _split_by_section(text: str) -> List[str]:
    """按一级结构切。"""
    if not text:
        return []

    indices = [m.start() for m in SECTION_RE.finditer(text)]
    if not indices:
        return [text]

    # 第一个 section 之前的内容也要保留（可能是文件标题/前言）
    sections: List[str] = []
    if indices[0] > 0:
        head = text[: indices[0]].strip()
        if head:
            sections.append(head)

    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(text)
        seg = text[start:end].strip()
        if seg:
            sections.append(seg)
    return sections


def _split_by_paragraph(text: str) -> List[str]:
    """按空行二次切。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras or [text]


def _split_by_sentence(text: str, max_len: int) -> List[str]:
    """按句号/分号在 max_len 限制下切。"""
    parts = SENTENCE_END_RE.split(text)
    sentences: List[str] = []
    buf = ""
    for piece in parts:
        if not piece:
            continue
        if SENTENCE_END_RE.fullmatch(piece):
            buf += piece
            sentences.append(buf)
            buf = ""
        else:
            buf += piece
    if buf:
        sentences.append(buf)

    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            if len(s) <= max_len:
                cur = s
            else:
                # 单句仍超长（极少见），强切
                for i in range(0, len(s), max_len):
                    chunks.append(s[i : i + max_len])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def _merge_with_overlap(
    pieces: List[str], chunk_size: int, overlap: int
) -> List[str]:
    """把小段拼装成贴近 `chunk_size` 上限的大段，同时在边界处制造 overlap。

    overlap 的工程意义：避免“条款刚好被一分为二”后，半截上下文在检索阶段彼此不见面，
    导致答案断章取义。代价是文本会重复存储一丁点内容（可在向量库里被接受）。
    """

    if not pieces:
        return []
    chunks: List[str] = []
    cur = ""
    for piece in pieces:
        if not piece:
            continue
        if not cur:
            cur = piece
            continue
        if len(cur) + len(piece) + 1 <= chunk_size:
            cur = f"{cur}\n{piece}"
        else:
            chunks.append(cur)
            # overlap：从上一个 chunk 末尾取 overlap 长度作为下一个开头
            if overlap > 0 and len(cur) > overlap:
                cur = cur[-overlap:] + "\n" + piece
            else:
                cur = piece
    if cur:
        chunks.append(cur)
    return chunks


def split_text(
    text: str,
    document_id: str,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """主入口：将清洗后的文本切成 Chunk 列表。"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    # 1) 一级切
    sections = _split_by_section(text)

    # 2) 对每个 section：若过长则段落 -> 句子细切
    fine: List[str] = []
    for sec in sections:
        if len(sec) <= chunk_size:
            fine.append(sec)
            continue
        for para in _split_by_paragraph(sec):
            if len(para) <= chunk_size:
                fine.append(para)
            else:
                fine.extend(_split_by_sentence(para, chunk_size))

    # 3) 合并 + overlap
    merged = _merge_with_overlap(fine, chunk_size, chunk_overlap)

    # 4) 计算 start_char / end_char (尽力定位)
    chunks: List[Chunk] = []
    cursor = 0
    for idx, body in enumerate(merged):
        # 在原文中尝试找子串（去掉 overlap 重复）
        body_clean = body.strip()
        start = text.find(body_clean[:80], cursor) if body_clean else -1
        if start == -1:
            start = cursor
        end = start + len(body_clean)
        cursor = max(cursor, end - chunk_overlap)

        chunk = Chunk(
            chunk_id=f"{document_id}_chunk_{idx:04d}_{uuid.uuid4().hex[:6]}",
            document_id=document_id,
            filename=filename,
            chunk_index=idx,
            text=body_clean,
            start_char=start,
            end_char=end,
        )
        chunks.append(chunk)

    logger.info(
        f"split [{filename}] -> {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
