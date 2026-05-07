"""中文政策文档文本清洗。

设计原则：
- 清理 HTML 残留、重复空白、明显的页眉页脚噪声、无意义短行
- **保留**条款编号（一、二、三 / 第几条）、文件标题、日期、部门名称等政策信息
"""

from __future__ import annotations

import re

# 下面这些正则来自“人肉观察大量公文 PDF 转换结果”总结的启发式规则，并非银弹——

NOISE_PATTERNS = [
    r"^\s*第\s*\d+\s*页\s*共\s*\d+\s*页\s*$",  # 第 1 页 共 10 页
    r"^\s*[-—]\s*\d+\s*[-—]\s*$",                # - 1 -
    r"^\s*\d+\s*/\s*\d+\s*$",                      # 1 / 10
    r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$",        # Page 1 of 10
]

# “短行一刀切”很危险：会把真正的“章节编号（一）”砍掉，所以必须用 KEEP_* 放行。
KEEP_PATTERNS = [
    r"^[一二三四五六七八九十百零〇]+[、．\.]",  # 一、二、
    r"^第[一二三四五六七八九十百零〇\d]+[条章节款项]",  # 第X条
    r"^\d+[、．\.]",                                  # 1. 1、
    r"^\(\d+\)|（\d+）",                                # (1) （1）
    r"^[【〔（\(].*[】〕）\)]$",                          # 【...】
]

KEEP_RE = re.compile("|".join(KEEP_PATTERNS))


def _is_noise_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    for p in NOISE_PATTERNS:
        if re.match(p, line):
            return True
    return False


def _is_meaningful_short_line(line: str) -> bool:
    """判断一个短行是否需要保留（如条款编号、标题）。"""
    return bool(KEEP_RE.match(line.strip()))


def clean_text(text: str) -> str:
    """对原始文本做“温和清洗”（更看重**政策文本信息保留**，而不是极致压缩）。

    处理顺序很关键：先规范化字符与换行 → 再去标签 → 再做行级判定 → 最后压缩空行。
    """
    if not text:
        return ""

    # 1) 全角空格、零宽字符
    text = text.replace("\u3000", " ").replace("\u200b", "").replace("\ufeff", "")

    # 2) Windows 换行统一
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3) 去除残留的孤立 HTML 标签（保险起见）
    text = re.sub(r"<[^>]+>", "", text)

    # 4) 行处理
    cleaned_lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        # 行内多空格 -> 单空格
        line = re.sub(r"[ \t]+", " ", line)

        if _is_noise_line(line):
            continue
        # 短行：长度 <= 2 且不是结构标记 -> 丢弃
        if len(line) <= 2 and not _is_meaningful_short_line(line):
            continue
        cleaned_lines.append(line)

    # 5) 多空行压缩为单个空行
    result: list[str] = []
    blank = 0
    for line in cleaned_lines:
        if not line:
            blank += 1
            if blank <= 1:
                result.append(line)
        else:
            blank = 0
            result.append(line)

    return "\n".join(result).strip()
