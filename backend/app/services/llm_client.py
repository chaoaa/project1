"""LLM HTTP 封装层：对上统一成 `chat` / `structured_chat`，对下用 OpenAI 官方 SDK。

为什么要强调「OpenAI-compatible」这个关键词？
因为 DeepSeek / 通义千问 / Moonshot 等众多国内服务都提供**同一套 REST 契约**：
你只需要改 `base_url` + `model` + `api_key`，上层 Prompt 逻辑可以原封不动复用。

这层代码的价值：把供应商细节挡在门面之后，demo/生产切换成本低。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import settings
from app.utils.logger import logger


class LLMClientError(Exception):
    """自定义异常：让上层（FastAPI Router / LangGraph 节点）能用 `except LLMClientError` 精准捕获。"""


class LLMClient:
    """对 OpenAI SDK 做薄封装：`chat`（自然语言） vs `structured_chat`（尽量 JSON）。"""

    def __init__(self) -> None:
        if not settings.llm_api_key:
            logger.warning(
                "LLM_API_KEY 未配置！请编辑 .env 文件设置 LLM_API_KEY 后再使用问答功能。"
            )
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "missing",
        )
        self.model = settings.llm_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """最基础的补全接口。

        `messages` 采用 chat 模板：[{role, content}, ...]，
        兼容 system / user / assistant 三段式提示工程。
        """
        if not settings.llm_api_key:
            raise LLMClientError(
                "LLM_API_KEY 未配置，无法调用大模型。请先在 .env 中填写 LLM_API_KEY。"
            )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise LLMClientError(f"调用大模型失败: {e}") from e

    def structured_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """要求模型返回 JSON。

        优先使用 response_format=json_object，失败则手动解析。
        """
        if not settings.llm_api_key:
            raise LLMClientError(
                "LLM_API_KEY 未配置，无法调用大模型。请先在 .env 中填写 LLM_API_KEY。"
            )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
        except Exception as e:
            # 部分服务商不支持 `response_format=json_object`；降级为普通文本输出，再靠 `_safe_parse_json` 抢救。
            logger.warning(
                f"structured_chat 走普通 chat 模式（response_format 不被支持）: {e}"
            )
            content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)

        return self._safe_parse_json(content)

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, Any]:
        """尽力解析 JSON，宽容处理代码块、前后缀文字。"""
        if not text:
            return {}
        text = text.strip()

        # 优先去除 ```json ... ``` 包裹
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            pass

        # 尝试抓取第一个完整 JSON 对象
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.warning(f"json parse failed: {e}; raw={text[:200]}")
        return {"_raw": text}


_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton
