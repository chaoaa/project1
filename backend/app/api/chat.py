"""问答接口：`/api/chat/*`。

把本文件想成三道门——
1. `/query`：经典 RAG = 检索 + 拼装 Prompt + 单次 LLM 生成（不走 LangGraph）。适合对比学习。
2. `/agent`：LangGraph 状态机在后台跑一整条链路（意图 → 检索 → 按意图选工具节点 → 汇总）。这是简历上的主打能力。
3. `/retrieve`：只拿混合检索结果，不问大模型——用来调 `top_k`、看 BM25 / 向量谁贡献更大。

三者底层共享同一套检索 (`hybrid_search`)，保证行为一致。
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrievedChunk,
    RetrieveResponse,
)
from app.services.citation_service import (
    REFUSAL_MESSAGE,
    assess_confidence,
    build_context_block,
    chunks_to_citations,
)
from app.services.llm_client import LLMClientError, get_llm_client
from app.services.retriever import hybrid_search
from app.agent.prompts import RAG_ANSWER_PROMPT, SYSTEM_POLICY_QA
from app.utils.logger import logger


router = APIRouter(prefix="/api/chat", tags=["chat"])


# =========================================================
# /api/chat/query  - 基础 RAG 问答（不走 Agent）
# =========================================================

@router.post("/query", response_model=ChatResponse)
def query(req: ChatRequest) -> ChatResponse:
    """最直接的一条龙：Hybrid Search → 置信度判定 → （可选）LLM 作答。

    学习提示：它和 `policy_qa_node`（agent/graph.py）逻辑几乎等价，区别在于这里**没有意图分类**
    ，永远按“政策问答”口吻回答。
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    # Hybrid Search：同时走向量语义 + BM25 关键词两路召回，再在 retriever 里加权融合。
    chunks: List[RetrievedChunk] = hybrid_search(req.question, top_k=req.top_k)
    confidence, refused = assess_confidence(chunks)
    citations: List[Citation] = chunks_to_citations(chunks)

    # 第一道“安全闸”：没证据或最高分太低 → 直接把拒答模版返回给用户，堵住模型胡编。
    if refused or not chunks:
        return ChatResponse(
            answer=REFUSAL_MESSAGE,
            citations=citations,
            confidence=confidence,
            refused=True,
            intent="policy_qa",
            tool_name="search_policy",
            tool_result={"hits": len(chunks)},
        )

    context = build_context_block(chunks)  # 给 LLM 的“只允许阅读的材料区”，编号 [片段 X] 方便引用对齐
    prompt = RAG_ANSWER_PROMPT.format(context=context, question=req.question)

    try:
        answer = get_llm_client().chat(
            messages=[
                {"role": "system", "content": SYSTEM_POLICY_QA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except LLMClientError as e:
        return ChatResponse(
            answer=f"调用大模型失败：{e}",
            citations=citations,
            confidence=confidence,
            refused=True,
            intent="policy_qa",
            tool_name="search_policy",
            tool_result={"hits": len(chunks), "error": str(e)},
        )

    return ChatResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        refused=False,
        intent="policy_qa",
        tool_name="search_policy",
        tool_result={"hits": len(chunks)},
    )


# =========================================================
# /api/chat/agent - Agent 自动识别意图并调用工具
# =========================================================

@router.post("/agent", response_model=ChatResponse)
def agent_chat(req: ChatRequest) -> ChatResponse:
    """把整个业务交给 LangGraph。

    `graph.invoke(init_state)` 做的事可以类比“流程编排引擎”——你只需描述状态结构（AgentState），
    边的跳转逻辑全部写在 graph.py。

    invoke 返回值是一个**累计后的字典**（融合了每个节点返回的补丁字段）。
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    # TypedDict + total=False ⇒ 只用填起点字段，其余在执行过程中由各节点按需填充。
    init_state: AgentState = {
        "question": req.question,
        "user_profile": req.user_profile,
        "old_document_id": req.old_document_id,
        "new_document_id": req.new_document_id,
        "messages": [],
    }

    graph = get_graph()
    try:
        final_state: Dict[str, Any] = graph.invoke(init_state)
    except Exception as e:
        logger.error(f"agent invoke failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{e}")

    return ChatResponse(
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []) or [],
        confidence=final_state.get("confidence", "low"),
        refused=bool(final_state.get("refused", False)),
        intent=final_state.get("intent"),
        tool_name=final_state.get("tool_name"),
        tool_result=final_state.get("tool_result"),
    )


# =========================================================
# /api/chat/retrieve - 仅检索调试
# =========================================================

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_only(req: ChatRequest) -> RetrieveResponse:
    """开发者 / 助教模式：只看检索结果（不走 LLM）。"""
    if not req.question or not req.question.strip():        raise HTTPException(status_code=400, detail="question 不能为空")
    chunks = hybrid_search(req.question, top_k=req.top_k)
    return RetrieveResponse(query=req.question, total=len(chunks), chunks=chunks)
