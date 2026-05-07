"""LangGraph Agent 编排（本项目里最有“Agents 味儿”的一层）。

推荐给初学者的阅读顺序：
1）先看文末 `build_graph()`：有哪些 **节点（node）**、**普通边（add_edge）**、**条件边（add_conditional_edges）**。
2）再看 `_route()`：它只是根据 `intent` 返回字符串，LangGraph 用返回值决定下一步跳入哪个节点。
3）最后再回到各个 `*_node`：把每个函数都理解成 「输入 state → 输出要合并回 state 的一片字典」。

和“用 if else 写一个假 Agent”相比，这里的优势是学习曲线清晰：
以后要加第五个工具节点，大多数情况下是「复制一个节点的模板 → 接上新的条件分支」。

流程示意图：
START
  → classify_intent_node       # LLM(JSON) → intent
  → retrieve_node               # Hybrid Search → retrieved_chunks（版本对比会跳过检索）
  → route_tool_node             # 纯占位日志；真正路由写在 _route conditional_edges
  → policy_qa_node | eligibility_check_node | checklist_generation_node | version_compare_node
  → final_answer_node           # 统一收尾日志
  → END
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from app.agent import tools as toolset
from app.agent.prompts import (
    INTENT_CLASSIFY_PROMPT,
    RAG_ANSWER_PROMPT,
    SYSTEM_POLICY_QA,
)
from app.agent.state import AgentState
from app.schemas import IntentType
from app.services.citation_service import (
    REFUSAL_MESSAGE,
    assess_confidence,
    build_context_block,
    chunks_to_citations,
)
from app.services.llm_client import LLMClientError, get_llm_client
from app.utils.logger import logger


VALID_INTENTS = {
    "policy_qa",
    "eligibility_check",
    "checklist_generation",
    "version_compare",
    "unknown",
}


# =========================================================
# Nodes
# =========================================================

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """先用 LLM 做一次「意图分类」（文本 → 离散标签）。

    Fallback：只要调用失败或返回非法标签，就退回 `policy_qa`，避免整条 Agent 直接报错。
    """
    question = state.get("question", "").strip()

    # 如果调用方明确指定了 version_compare 所需文档，就跳过 LLM
    if state.get("old_document_id") and state.get("new_document_id"):
        return {
            "intent": "version_compare",
            "messages": [f"[intent] forced=version_compare"],
        }

    prompt = INTENT_CLASSIFY_PROMPT.format(question=question)
    try:
        data = get_llm_client().structured_chat(
            messages=[
                {"role": "system", "content": "你只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        intent = data.get("intent", "policy_qa")
    except LLMClientError as e:
        logger.warning(f"intent classify fallback to policy_qa: {e}")
        intent = "policy_qa"
    except Exception as e:
        logger.warning(f"intent classify error, fallback: {e}")
        intent = "policy_qa"

    if intent not in VALID_INTENTS:
        intent = "policy_qa"

    logger.info(f"[node] classify_intent -> {intent}")
    return {"intent": intent, "messages": [f"[intent] {intent}"]}


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Hybrid Search：为问答/资格判断/清单生成准备上下文。

    `version_compare` 会直接拼两篇政策全文交由 LLM 对比，不再需要 Top‑K chunk，因此跳过本节检索。"""
    intent: IntentType = state.get("intent", "policy_qa")  # type: ignore[assignment]
    if intent == "version_compare":
        return {"retrieved_chunks": [], "messages": ["[retrieve] skipped for version_compare"]}

    question = state.get("question", "")
    chunks = toolset.search_policy(question)
    return {
        "retrieved_chunks": chunks,
        "messages": [f"[retrieve] hits={len(chunks)}"],
    }


def route_tool_node(state: AgentState) -> Dict[str, Any]:
    """占位节点：方便统一打点日志 / tracing。

    真正实现分支的是 `conditional_edges`；此处仅把当前意图写进调试用的 messages。"""
    return {"messages": ["[route] -> " + state.get("intent", "policy_qa")]}


# ---- 工具节点 ----

def policy_qa_node(state: AgentState) -> Dict[str, Any]:
    """标准 RAG：检索 →（可选拒答）→ `SYSTEM_POLICY_QA + RAG_ANSWER_PROMPT`。"""
    chunks = state.get("retrieved_chunks", []) or []
    confidence, refused = assess_confidence(chunks)
    citations = chunks_to_citations(chunks)

    if refused or not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": citations,
            "confidence": confidence,
            "refused": True,
            "tool_name": "search_policy",
            "tool_result": {"hits": len(chunks)},
            "messages": ["[policy_qa] refused"],
        }

    context = build_context_block(chunks)
    prompt = RAG_ANSWER_PROMPT.format(context=context, question=state.get("question", ""))
    try:
        answer = get_llm_client().chat(
            messages=[
                {"role": "system", "content": SYSTEM_POLICY_QA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except LLMClientError as e:
        return {
            "answer": f"调用大模型失败：{e}",
            "citations": citations,
            "confidence": confidence,
            "refused": True,
            "tool_name": "search_policy",
            "tool_result": {"hits": len(chunks), "error": str(e)},
            "messages": ["[policy_qa] llm_error"],
        }

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "refused": False,
        "tool_name": "search_policy",
        "tool_result": {"hits": len(chunks)},
        "messages": ["[policy_qa] ok"],
    }


def eligibility_check_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("retrieved_chunks", []) or []
    confidence, refused = assess_confidence(chunks)
    citations = chunks_to_citations(chunks)

    if refused or not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": citations,
            "confidence": confidence,
            "refused": True,
            "tool_name": "check_eligibility",
            "tool_result": {"hits": len(chunks)},
            "messages": ["[eligibility] refused"],
        }

    result = toolset.check_eligibility(
        question=state.get("question", ""),
        user_profile=state.get("user_profile"),
        chunks=chunks,
    )
    answer = _format_eligibility_answer(result)
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "refused": False,
        "tool_name": "check_eligibility",
        "tool_result": result,
        "messages": ["[eligibility] ok"],
    }


def checklist_generation_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("retrieved_chunks", []) or []
    confidence, refused = assess_confidence(chunks)
    citations = chunks_to_citations(chunks)

    if refused or not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": citations,
            "confidence": confidence,
            "refused": True,
            "tool_name": "generate_checklist",
            "tool_result": {"hits": len(chunks)},
            "messages": ["[checklist] refused"],
        }

    result = toolset.generate_checklist(
        question=state.get("question", ""),
        chunks=chunks,
    )
    answer = _format_checklist_answer(result)
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "refused": False,
        "tool_name": "generate_checklist",
        "tool_result": result,
        "messages": ["[checklist] ok"],
    }


def version_compare_node(state: AgentState) -> Dict[str, Any]:
    old_id = state.get("old_document_id")
    new_id = state.get("new_document_id")
    if not old_id or not new_id:
        return {
            "answer": "请指定要对比的旧版与新版政策文档 ID（old_document_id / new_document_id）。",
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "compare_policy_versions",
            "tool_result": {"error": "missing document ids"},
            "messages": ["[version_compare] missing ids"],
        }

    old_text = toolset.fetch_document_text(old_id)
    new_text = toolset.fetch_document_text(new_id)
    if not old_text or not new_text:
        return {
            "answer": "未在知识库中找到指定的旧版或新版政策文档，请先上传。",
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "compare_policy_versions",
            "tool_result": {"error": "document not found"},
            "messages": ["[version_compare] doc not found"],
        }

    result = toolset.compare_policy_versions(
        question=state.get("question", ""),
        old_text=old_text,
        new_text=new_text,
    )
    answer = _format_compare_answer(result)
    return {
        "answer": answer,
        "citations": [],
        "confidence": "high" if result.get("summary") else "medium",
        "refused": False,
        "tool_name": "compare_policy_versions",
        "tool_result": result,
        "messages": ["[version_compare] ok"],
    }


def final_answer_node(state: AgentState) -> Dict[str, Any]:
    """收尾节点：可在此处做最终格式化、日志。"""
    logger.info(
        f"[final] intent={state.get('intent')} refused={state.get('refused')} "
        f"confidence={state.get('confidence')}"
    )
    return {"messages": ["[final] done"]}


# =========================================================
# 路由
# =========================================================

def _route(state: AgentState) -> str:
    intent: IntentType = state.get("intent", "policy_qa")  # type: ignore[assignment]
    if intent == "eligibility_check":
        return "eligibility_check_node"
    if intent == "checklist_generation":
        return "checklist_generation_node"
    if intent == "version_compare":
        return "version_compare_node"
    return "policy_qa_node"


# =========================================================
# Graph 构建
# =========================================================

_compiled_graph = None


def build_graph():
    """构建 LangGraph 状态图。"""
    g = StateGraph(AgentState)

    g.add_node("classify_intent_node", classify_intent_node)
    g.add_node("retrieve_node", retrieve_node)
    g.add_node("route_tool_node", route_tool_node)
    g.add_node("policy_qa_node", policy_qa_node)
    g.add_node("eligibility_check_node", eligibility_check_node)
    g.add_node("checklist_generation_node", checklist_generation_node)
    g.add_node("version_compare_node", version_compare_node)
    g.add_node("final_answer_node", final_answer_node)

    g.add_edge(START, "classify_intent_node")
    g.add_edge("classify_intent_node", "retrieve_node")
    g.add_edge("retrieve_node", "route_tool_node")

    g.add_conditional_edges(
        "route_tool_node",
        _route,
        {
            "policy_qa_node": "policy_qa_node",
            "eligibility_check_node": "eligibility_check_node",
            "checklist_generation_node": "checklist_generation_node",
            "version_compare_node": "version_compare_node",
        },
    )

    g.add_edge("policy_qa_node", "final_answer_node")
    g.add_edge("eligibility_check_node", "final_answer_node")
    g.add_edge("checklist_generation_node", "final_answer_node")
    g.add_edge("version_compare_node", "final_answer_node")
    g.add_edge("final_answer_node", END)

    return g.compile()


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# =========================================================
# 工具：把结构化结果格式化成自然语言
# =========================================================

def _format_eligibility_answer(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"资格判断失败：{result['error']}"

    decision_map = {
        "eligible": "满足申请条件",
        "not_eligible": "不满足申请条件",
        "need_more_info": "信息不足，无法完整判断",
    }
    decision = decision_map.get(result.get("result", ""), "结论未知")
    parts = [f"【资格判断结论】{decision}"]
    if result.get("explanation"):
        parts.append(f"\n说明：{result['explanation']}")
    if result.get("satisfied_conditions"):
        parts.append("\n已满足条件：")
        parts.extend(f"  - {x}" for x in result["satisfied_conditions"])
    if result.get("unsatisfied_conditions"):
        parts.append("\n未满足条件：")
        parts.extend(f"  - {x}" for x in result["unsatisfied_conditions"])
    if result.get("missing_information"):
        parts.append("\n仍需补充信息：")
        parts.extend(f"  - {x}" for x in result["missing_information"])
    if result.get("evidence"):
        parts.append("\n依据：")
        parts.extend(f"  - {x}" for x in result["evidence"])
    return "\n".join(parts)


def _format_checklist_answer(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"清单生成失败：{result['error']}"
    parts = [f"【办事任务】{result.get('task_name') or '未指定'}"]
    if result.get("materials"):
        parts.append("\n所需材料：")
        for i, m in enumerate(result["materials"], 1):
            parts.append(f"  {i}. {m}")
    if result.get("steps"):
        parts.append("\n办理步骤：")
        for i, s in enumerate(result["steps"], 1):
            parts.append(f"  {i}. {s}")
    if result.get("notes"):
        parts.append("\n注意事项：")
        parts.extend(f"  - {n}" for n in result["notes"])
    if result.get("evidence"):
        parts.append("\n依据：")
        parts.extend(f"  - {e}" for e in result["evidence"])
    return "\n".join(parts)


def _format_compare_answer(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"版本对比失败：{result.get('error') or result.get('summary', '')}"
    parts = [f"【版本对比摘要】{result.get('summary', '')}"]
    if result.get("added"):
        parts.append("\n新增内容：")
        parts.extend(f"  + {x}" for x in result["added"])
    if result.get("removed"):
        parts.append("\n删除内容：")
        parts.extend(f"  - {x}" for x in result["removed"])
    if result.get("changed"):
        parts.append("\n修改内容：")
        for c in result["changed"]:
            if isinstance(c, dict):
                parts.append(
                    f"  ~ 旧：{c.get('from','')}\n    新：{c.get('to','')}"
                )
            else:
                parts.append(f"  ~ {c}")
    return "\n".join(parts)
