"""Agent 侧的工具函数集合（本项目里的 Tool Calling 实现载体）。

为什么是“Python 函数”而不是网关 JSON Schema？
——更利于学习和单测：`search_policy`、`check_eligibility` 都能在 REPL / notebook 里直接调用，
无需先注册到某个模型厂商控制台。

共性约定：
1. **search_policy**：纯检索，读 Qdrant + BM25。
2. 其它三类：把检索上下文喂给 Prompt → `structured_chat` 拉回 JSON → `_normalize_*` 统一字段，
   LangGraph 节点即可拿 dict 走人。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent.prompts import (
    CHECKLIST_PROMPT,
    ELIGIBILITY_PROMPT,
    VERSION_COMPARE_PROMPT,
)
from app.schemas import RetrievedChunk
from app.services.citation_service import build_context_block
from app.services.llm_client import get_llm_client
from app.services.retriever import hybrid_search
from app.services.vector_store import get_vector_store
from app.utils.logger import logger


# =========================================================
# Tool 1: search_policy
# =========================================================

def search_policy(query: str, top_k: int | None = None) -> List[RetrievedChunk]:
    """检索政策原文。"""
    logger.info(f"[tool] search_policy: {query}")
    return hybrid_search(query, top_k=top_k)


# =========================================================
# Tool 2: check_eligibility
# =========================================================

def check_eligibility(
    question: str,
    user_profile: Optional[Dict[str, Any]],
    chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """根据用户条件和政策上下文，逐项判断是否满足条件。"""
    logger.info("[tool] check_eligibility")
    profile_text = _format_profile(user_profile)
    context = build_context_block(chunks)

    prompt = ELIGIBILITY_PROMPT.format(
        context=context,
        question=question,
        user_profile=profile_text,
    )
    try:
        result = get_llm_client().structured_chat(
            messages=[
                {"role": "system", "content": "你严格按 JSON 模式输出。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception as e:
        return _error_payload("check_eligibility", e)

    return _normalize_eligibility(result)


# =========================================================
# Tool 3: generate_checklist
# =========================================================

def generate_checklist(
    question: str,
    chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """根据政策上下文生成材料清单。"""
    logger.info("[tool] generate_checklist")
    context = build_context_block(chunks)
    prompt = CHECKLIST_PROMPT.format(context=context, question=question)
    try:
        result = get_llm_client().structured_chat(
            messages=[
                {"role": "system", "content": "你严格按 JSON 模式输出。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception as e:
        return _error_payload("generate_checklist", e)

    return _normalize_checklist(result)


# =========================================================
# Tool 4: compare_policy_versions
# =========================================================

def compare_policy_versions(
    question: str,
    old_text: str,
    new_text: str,
) -> Dict[str, Any]:
    """对比新旧版本政策文本，输出差异。"""
    logger.info("[tool] compare_policy_versions")
    if not old_text or not new_text:
        return {
            "added": [],
            "removed": [],
            "changed": [],
            "summary": "缺少旧版或新版政策内容，无法对比。",
            "error": "missing_input",
        }

    prompt = VERSION_COMPARE_PROMPT.format(
        old_text=_truncate(old_text, 6000),
        new_text=_truncate(new_text, 6000),
        question=question or "请对比新旧版本差异",
    )
    try:
        result = get_llm_client().structured_chat(
            messages=[
                {"role": "system", "content": "你严格按 JSON 模式输出。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
    except Exception as e:
        return _error_payload("compare_policy_versions", e)

    return _normalize_compare(result)


def fetch_document_text(document_id: str) -> str:
    """从 Qdrant 拉取某个文档全部 chunk 的文本。"""
    if not document_id:
        return ""
    chunks = get_vector_store().fetch_chunks_by_document(document_id)
    return "\n".join(c.text for c in chunks)


# =========================================================
# helpers
# =========================================================

def _format_profile(profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return "(用户未提供个人信息)"
    lines = [f"- {k}: {v}" for k, v in profile.items()]
    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len // 2] + "\n...（内容过长已截断）...\n" + text[-max_len // 2 :]


def _normalize_eligibility(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "result": data.get("result", "need_more_info"),
        "satisfied_conditions": list(data.get("satisfied_conditions", []) or []),
        "unsatisfied_conditions": list(data.get("unsatisfied_conditions", []) or []),
        "missing_information": list(data.get("missing_information", []) or []),
        "evidence": list(data.get("evidence", []) or []),
        "explanation": data.get("explanation", ""),
    }


def _normalize_checklist(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_name": data.get("task_name", ""),
        "materials": list(data.get("materials", []) or []),
        "steps": list(data.get("steps", []) or []),
        "notes": list(data.get("notes", []) or []),
        "evidence": list(data.get("evidence", []) or []),
    }


def _normalize_compare(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "added": list(data.get("added", []) or []),
        "removed": list(data.get("removed", []) or []),
        "changed": list(data.get("changed", []) or []),
        "summary": data.get("summary", ""),
    }


def _error_payload(tool: str, e: Exception) -> Dict[str, Any]:
    logger.error(f"tool {tool} failed: {e}")
    return {"error": str(e), "tool": tool}
