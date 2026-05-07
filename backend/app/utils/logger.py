"""统一日志出口（loguru）。

为什么封装一层？
1. 全局 `import logger` ——少传参、少配置。
2. 以后若你要把日志打到 Kafka / 文件，只需改这个文件而不是全项目搜索 `print`。

loguru 特性：比 stdlib logging 更人性化的默认格式 + 颜色。
"""

from __future__ import annotations

import sys
from loguru import logger

# Loguru 自带一个默认 handler；先 remove 再加，保证格式完全可控。
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)


__all__ = ["logger"]
