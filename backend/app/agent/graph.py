"""LangGraph Agent 编排（本项目里最有"Agents 味儿"的一层）。

=============================================================================
推荐给初学者的阅读顺序（由浅入深）
=============================================================================
第 1 步：看文末 `build_graph()` → 理解"图"是怎么搭起来的
  - add_node()：注册了哪些节点（每个节点就是一个 Python 函数）
  - add_edge()：固定边（A 完了必须去 B，无条件）
  - add_conditional_edges()：条件边（根据 state 中的某个字段决定下一步去哪）

第 2 步：看 `_route()` 函数 → 理解"分支"是怎么实现的
  - 输入：state（当前图状态）
  - 输出：字符串（下一个节点的名字）
  - LangGraph 用这个字符串去查找对应的节点函数并执行

第 3 步：逐个看 `*_node(state) → dict` 函数
  - 每个节点函数：输入 当前 state → 输出 要合并回 state 的补丁字典
  - state 是累积的：上一个节点返回的字段会合并到 state 中，下一个节点能看到
  - TypedDict 中用 operator.add 标注的字段（如 messages）是追加而非覆盖

=============================================================================
什么是"节点返回 dict，LangGraph 自动合并到 state"？
=============================================================================
假设 state 当前 = {"question": "我能申请吗？", "retrieved_chunks": []}

retrieve_node 返回 {"retrieved_chunks": [chunk1, chunk2], "messages": ["hits=2"]}

合并后 state = {
    "question": "我能申请吗？",        ← 没被覆盖，保留
    "retrieved_chunks": [chunk1, chunk2],  ← 被覆盖为返回值
    "messages": ["[intent] eligibility_check", "hits=2"],  ← operator.add 追加
}

=============================================================================
流程图（画成 Mermaid 就是）：
START → classify_intent → retrieve → route_tool → {
    policy_qa_node | eligibility_check_node | checklist_generation_node |
    version_compare_node | web_ingestion_node
} → final_answer → END
=============================================================================
"""

from __future__ import annotations

import re
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from app.agent import tools as toolset
from app.agent.business_tools import (
    extract_scholarship_profile_from_question,
    generate_checklist_docx,
    rule_based_scholarship_checker,
)
from app.agent.prompts import (
    INTENT_CLASSIFY_PROMPT,
    RAG_ANSWER_PROMPT,
    SYSTEM_POLICY_QA,
)
from app.agent.state import AgentState
# IntentType 用于类型标注；Literal 类型确保值只能是 LIST 中的某几个
from app.schemas import IntentType
from app.services.citation_service import (
    REFUSAL_MESSAGE,
    assess_confidence,
    build_context_block,
    chunks_to_citations,
)
from app.services.ingestion_service import IngestionError
from app.services.llm_client import LLMClientError, get_llm_client
from app.services.url_ingestion import ingest_policy_from_url
from app.utils.logger import logger


# ---- 合法的意图标签集合 ----
# 如果 LLM 返回的 intent 不在此集合中 → 降级为 policy_qa（安全兜底）
VALID_INTENTS = {
    "policy_qa",              # 普通政策问答
    "eligibility_check",      # 资格判断（规则引擎）
    "checklist_generation",   # 材料清单生成（含 Word 导出）
    "version_compare",        # 新旧版本政策对比
    "web_ingestion",          # 网页政策采集（v2 新增）
    "unknown",                # 无法识别
}


# =========================================================
# 第 1 个节点：意图分类
# =========================================================

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """用 LLM 做一次「意图分类」：把用户自然语言问题映射到 6 个离散标签。

    【为什么需要意图分类？】
    不同的用户意图需要走完全不同的处理链路：
    - "奖学金申请条件" → 检索政策 + LLM 回答（policy_qa）
    - "我能不能申请" → 抽取画像 + 规则判断（eligibility_check）
    - "把网页加入知识库" → 抓取 URL + 入库（web_ingestion）
    如果只有一个链路的"万能 Agent"，所有问题都要经过检索 + LLM，浪费且容易出错。

    【Fallback 策略（3 层兜底）】
    1. 如果调用方传了 old_document_id + new_document_id → 强制视为 version_compare
    2. 如果问题中包含 URL + 入库关键词 → 规则直接判定为 web_ingestion（省一次 LLM 调用）
    3. LLM 分类失败或返回非法标签 → 降级为 policy_qa（最通用的链路）
    """

    question = state.get("question", "").strip()

    # ---- 兜底 1：版本对比被显式指定（来自前端选择框） ----
    if state.get("old_document_id") and state.get("new_document_id"):
        return {
            "intent": "version_compare",
            "messages": ["[intent] forced=version_compare"],
        }

    # ---- 兜底 2：URL 采集的快速规则判断（不调 LLM，省钱省时间） ----
    # 正则检查问题中是否包含 URL，同时检查是否包含入库相关关键词。
    # 单纯发一个 URL 不一定是想入库（可能是问"这个网页的内容是什么意思"），
    # 所以必须同时命中 URL 和 入库关键词。
    if re.search(r"https?://", question):
        lower_q = question.lower()
        web_keywords = [
            "加入知识库", "入库", "抓取", "采集", "把这个网页", "这个链接",
            "网页加入", "链接加入", "更新这个网页", "采集这个"
        ]
        if any(kw in lower_q for kw in web_keywords):
            return {
                "intent": "web_ingestion",
                "messages": ["[intent] rule=web_ingestion"],
            }

    # ---- 正常路径：LLM 分类 ----
    prompt = INTENT_CLASSIFY_PROMPT.format(question=question)
    try:
        data = get_llm_client().structured_chat(
            messages=[
                {"role": "system", "content": "你只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,     # 分类任务必须确定，temperature=0 禁止随机性
            max_tokens=100,      # 分类结果很短，限制 token 防输出过长
        )
        intent = data.get("intent", "policy_qa")
    except LLMClientError as e:
        logger.warning(f"intent classify fallback to policy_qa: {e}")
        intent = "policy_qa"
    except Exception as e:
        logger.warning(f"intent classify error, fallback: {e}")
        intent = "policy_qa"

    # ---- 兜底 3：标签不在合法集合中 → 降级 ----
    if intent not in VALID_INTENTS:
        intent = "policy_qa"

    logger.info(f"[node] classify_intent -> {intent}")
    return {"intent": intent, "messages": [f"[intent] {intent}"]}


# =========================================================
# 第 2 个节点：混合检索
# =========================================================

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Hybrid Search：向量 + BM25 两路召回融合。

    【哪些意图不需要检索？】
    - version_compare：直接用 fetch_document_text 拉取整篇文档的全文
    - web_ingestion：不需要检索任何东西，反而要向知识库写入新内容

    跳过不必要的检索能省 embedding 计算和 Qdrant 网络请求。
    """

    intent: IntentType = state.get("intent", "policy_qa")  # type: ignore[assignment]

    if intent in ("version_compare", "web_ingestion"):
        return {
            "retrieved_chunks": [],
            "messages": [f"[retrieve] skipped for {intent}"],
        }

    question = state.get("question", "")
    chunks = toolset.search_policy(question)
    return {
        "retrieved_chunks": chunks,
        "messages": [f"[retrieve] hits={len(chunks)}"],
    }


# =========================================================
# 第 3 个节点：路由占位
# =========================================================

def route_tool_node(state: AgentState) -> Dict[str, Any]:
    """占位节点：不打游戏逻辑，只写日志、留痕迹。

    真正的分支跳转逻辑在 _route() 和 conditional_edges 中。
    把路由逻辑和日志分开，符合"单一职责原则"。
    """
    return {"messages": ["[route] -> " + state.get("intent", "policy_qa")]}


# =========================================================
# 工具节点 1：普通政策问答（RAG）
# =========================================================

def policy_qa_node(state: AgentState) -> Dict[str, Any]:
    """标准 RAG：Hybrid Search 结果 → 拼 Prompt → LLM 生成答案。

    这是最基础也最常用的链路，占所有 Agent 请求的 ~70%。
    """

    chunks = state.get("retrieved_chunks", []) or []

    # 置信度评估（基于 top1 的融合分数）
    confidence, refused = assess_confidence(chunks)
    citations = chunks_to_citations(chunks)

    # 低置信度 → 直接拒答，不会把低质量上下文喂给 LLM
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

    # 将检索到的 chunk 拼接成带编号的上下文块
    context = build_context_block(chunks)
    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        question=state.get("question", ""),
    )

    try:
        answer = get_llm_client().chat(
            messages=[
                {"role": "system", "content": SYSTEM_POLICY_QA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # 低温度减少编造，但保留一点自然性
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


# =========================================================
# 工具节点 2：资格判断（增强版 — 规则引擎 + LLM 依据）
# =========================================================

def eligibility_check_node(state: AgentState) -> Dict[str, Any]:
    """资格判断节点（v2 增强版）。

    【增强前 vs 增强后】
    增强前：chunks → LLM 直接判断 → JSON 结论
            问题：LLM 可能忽略挂科、处分等一票否决条件，或产生随机结论。

    增强后：
    1. 保留原有检索 + 置信度判断 + 引用生成（不变）
    2. 从问题 + user_profile 中通过正则抽取用户画像（确定性，不用 LLM）
    3. 调用 rule_based_scholarship_checker 做确定性规则判断（纯 Python if/else）
    4. 同时调用 LLM 获取政策原文引用（LLM 只负责"从政策里找依据"，不负责判断）
    5. 合并结果：规则判断为主（decision/level），LLM 结果为补充（evidence）
    """

    chunks = state.get("retrieved_chunks", []) or []
    confidence, refused = assess_confidence(chunks)
    citations = chunks_to_citations(chunks)

    # 低置信度 → 拒答（知识库里根本找不到相关政策）
    if refused or not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": citations,
            "confidence": confidence,
            "refused": True,
            "tool_name": "rule_based_scholarship_checker",
            "tool_result": {"hits": len(chunks)},
            "messages": ["[eligibility] refused"],
        }

    # ---- 第 1 步：确定性规则抽取用户画像 ----
    # 不走 LLM，纯正则 + 关键词匹配。结果可复现、可调试。
    profile = extract_scholarship_profile_from_question(
        question=state.get("question", ""),
        user_profile=state.get("user_profile"),
    )

    # ---- 第 2 步：确定性规则引擎判断 ----
    # 同样不走 LLM。一票否决项命中 → not_eligible；排名判断 → 一等/二等/不符合。
    rule_result = rule_based_scholarship_checker(profile)

    # ---- 第 3 步：LLM 补充政策原文引用 ----
    # LLM 只负责"从检索到的政策 chunk 里找证据"，不负责下结论。
    # 这样 LLM 即使幻觉，也只是引用错了条文，不会把"不符合"错判成"符合"。
    llm_result = toolset.check_eligibility(
        question=state.get("question", ""),
        user_profile=state.get("user_profile"),
        chunks=chunks,
    )

    # ---- 第 4 步：合并 ----
    # 规则判断结果为主，LLM 的 evidence 作为政策依据补充
    merged_result = {
        **rule_result,
        "llm_evidence": llm_result.get("evidence", []),
        "llm_explanation": llm_result.get("explanation", ""),
    }

    answer = _format_rule_eligibility_answer(merged_result)
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "refused": False,
        "tool_name": "rule_based_scholarship_checker",
        "tool_result": merged_result,
        "messages": ["[eligibility] ok"],
    }


# =========================================================
# 工具节点 3：材料清单生成（增强版 — Word 导出）
# =========================================================

def checklist_generation_node(state: AgentState) -> Dict[str, Any]:
    """材料清单生成节点（v2 增强版）。

    【增强前 vs 增强后】
    增强前：chunks → LLM 生成 materials/steps/notes → 格式化为文本答案。
    增强后：在原有基础上额外调用 generate_checklist_docx 生成 .docx 文件，
           并将下载 URL 附在 tool_result 和 answer 中。

    【Word 生成失败的处理策略】
    docx 生成失败不应导致整个 Agent 请求失败（毕竟清单文本已经生成好了）。
    所以用 try/except 包裹，失败时仅在 answer 中提示用户，tool_result 中记录错误。
    """

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

    # ---- 第 1 步：LLM 生成材料清单 ----
    result = toolset.generate_checklist(
        question=state.get("question", ""),
        chunks=chunks,
    )

    # ---- 第 2 步：生成 Word 文件（尽力而为） ----
    docx_info = None
    docx_error = None
    try:
        docx_info = generate_checklist_docx(
            task_name=result.get("task_name") or "办事材料清单",
            materials=result.get("materials", []),
            steps=result.get("steps", []),
            notes=result.get("notes", []),
        )
    except Exception as e:
        logger.warning(f"[checklist] docx generation failed: {e}")
        docx_error = str(e)

    # ---- 第 3 步：组装结果 ----
    tool_result = {**result}
    if docx_info:
        tool_result["generated_file"] = {
            "filename": docx_info["filename"],
            "download_url": docx_info["download_url"],
        }
    if docx_error:
        tool_result["generated_file_error"] = docx_error

    answer = _format_checklist_answer(result)
    if docx_info:
        answer += (
            f"\n\n---\n已生成 Word 材料清单，可在前端下载："
            f" `{docx_info['filename']}`"
        )
    elif docx_error:
        answer += "\n\n（Word 文件生成失败，请稍后重试。）"

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "refused": False,
        "tool_name": "generate_checklist",
        "tool_result": tool_result,
        "messages": ["[checklist] ok"],
    }


# =========================================================
# 工具节点 4：版本对比
# =========================================================

def version_compare_node(state: AgentState) -> Dict[str, Any]:
    """对比新旧版本政策差异。

    不需要 Hybrid Search —— 直接从 Qdrant 拉取旧版和新版文档的全部 chunk，
    拼成完整原文，然后交给 LLM 对比差异。
    """

    old_id = state.get("old_document_id")
    new_id = state.get("new_document_id")

    if not old_id or not new_id:
        return {
            "answer": (
                "请指定要对比的旧版与新版政策文档 ID"
                "（old_document_id / new_document_id）。"
            ),
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "compare_policy_versions",
            "tool_result": {"error": "missing document ids"},
            "messages": ["[version_compare] missing ids"],
        }

    # fetch_document_text：从 Qdrant 按 document_id 拉全部 chunk，拼成全文
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


# =========================================================
# 工具节点 5：网页政策采集（v2 新增）
# =========================================================

def web_ingestion_node(state: AgentState) -> Dict[str, Any]:
    """网页政策采集节点（v2 新增）。

    【完整的 Agent 工具调用模式】
    1. 从用户问题中提取 URL（正则）
    2. 调用 ingest_policy_from_url 执行采集 + 入库
    3. 成功 → 返回摘要信息（标题、chunk 数量、来源 URL）
    4. 失败 → 返回清晰的错误信息，refused=True

    【为什么失败也算"refused=True"？】
    语义上：系统拒绝了给出有用答案的请求。但和 RAG 拒答不同：
    - RAG 拒答 = "知识库里没有 → 没法回答"
    - web_ingestion 失败 = "网页没法采集 → 没法入库"
    两者的共同点是：系统在当前状态下无法完成任务。
    """

    question = state.get("question", "")

    # ---- 第 1 步：从问题中提取 URL ----
    # 正则解释：
    #   https?://              → 协议部分
    #   [^\s）\)】。，,]+       → URL 内容，直到遇到空白或中文标点
    #   中文标点（）、】、。）也作为分隔符，因为用户可能写"这个链接：https://xxx。帮我校验"
    url_match = re.search(r"https?://[^\s）\)】。，,]+", question)
    if not url_match:
        return {
            "answer": (
                "请在问题中提供要采集的政策网页链接"
                "（以 http:// 或 https:// 开头）。"
            ),
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "ingest_policy_from_url",
            "tool_result": {"error": "no_url_found"},
            "messages": ["[web_ingestion] no url found"],
        }

    # rstrip 去掉 URL 末尾可能跟的中文标点
    url = url_match.group(0).rstrip(".。,，）)】")

    # ---- 第 2 步：调用采集工具 ----
    # ingest_policy_from_url 内部完成了：请求 → 清洗 → 保存 → 入库
    try:
        result = ingest_policy_from_url(url)
    except ValueError as e:
        # URL 不合法（协议不对、没域名等）
        return {
            "answer": f"URL 不合法：{e}",
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "ingest_policy_from_url",
            "tool_result": {"error": str(e), "url": url},
            "messages": ["[web_ingestion] invalid url"],
        }
    except IngestionError as e:
        # 网页请求失败、内容过短、入库失败等
        return {
            "answer": f"网页采集失败：{e}",
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "ingest_policy_from_url",
            "tool_result": {"error": str(e), "url": url},
            "messages": ["[web_ingestion] ingestion error"],
        }
    except Exception as e:
        # 未知异常（应该尽可能少发生）
        logger.error(f"[web_ingestion] unexpected error: {e}")
        return {
            "answer": f"网页采集时发生未知错误：{e}",
            "citations": [],
            "confidence": "low",
            "refused": True,
            "tool_name": "ingest_policy_from_url",
            "tool_result": {"error": str(e), "url": url},
            "messages": ["[web_ingestion] unexpected error"],
        }

    # ---- 第 3 步：成功 → 返回摘要 ----
    title = result.get("title", "") or result.get("filename", "")
    answer = (
        f"已成功采集并入库网页政策：\n"
        f"- 标题：{title}\n"
        f"- 文件名：{result['filename']}\n"
        f"- 切分 chunk 数量：{result['chunk_count']}\n"
        f"- 来源 URL：{result.get('source_url', url)}\n"
        f"- 状态：{result['status']}\n"
    )

    return {
        "answer": answer,
        "citations": [],
        "confidence": "high",   # 入库操作是确定性的，成功就是 high
        "refused": False,
        "tool_name": "ingest_policy_from_url",
        "tool_result": result,
        "messages": ["[web_ingestion] ok"],
    }


# =========================================================
# 第 8 个节点：最终收尾
# =========================================================

def final_answer_node(state: AgentState) -> Dict[str, Any]:
    """收尾节点：统一日志、可在此做最终格式化/审查。

    目前只打日志，但留下了扩展点：
    - 可以在这里对 answer 做敏感词过滤
    - 可以在这里统一记录分析指标（耗时、token 数等）
    - 可以在这里做最终的安全审查
    """
    logger.info(
        f"[final] intent={state.get('intent')} "
        f"refused={state.get('refused')} "
        f"confidence={state.get('confidence')}"
    )
    return {"messages": ["[final] done"]}


# =========================================================
# 路由函数：根据 intent 决定下一个节点
# =========================================================

def _route(state: AgentState) -> str:
    """条件边的决策函数。

    LangGraph 的行为：
    1. 执行完 route_tool_node 后，调用本函数
    2. 本函数返回一个字符串（如 "eligibility_check_node"）
    3. LangGraph 在 conditional_edges 的映射表中查找这个字符串
    4. 找到对应的节点并执行

    为什么不用 if/elif 链而是 if+return？
    每个 return 都意味着"找到目标，不再往下看"，性能稍好且逻辑更清晰。
    """

    intent: IntentType = state.get("intent", "policy_qa")  # type: ignore[assignment]

    if intent == "eligibility_check":
        return "eligibility_check_node"
    if intent == "checklist_generation":
        return "checklist_generation_node"
    if intent == "version_compare":
        return "version_compare_node"
    if intent == "web_ingestion":
        return "web_ingestion_node"
    # 默认：policy_qa 和 unknown 都走政策问答
    return "policy_qa_node"


# =========================================================
# Graph 构建（把所有节点和边组装起来）
# =========================================================

_compiled_graph = None  # 模块级缓存，避免每次请求都重新 build


def build_graph():
    """构建 LangGraph 状态图。

    【什么是编译（compile）？】
    build_graph 只是"声明"图的结构。
    g.compile() 才是真正把声明转成可执行的运行时对象。
    编译后的图会做验证（例如检查是否有死循环、孤立节点等）。

    【扩展指南：如何新增第 7 个工具节点？】
    1. 写一个 `my_new_node(state) -> dict` 函数
    2. 在 VALID_INTENTS 中新增你的意图标签
    3. 在 classify_intent_node 中添加对应分支（或让 LLM 学习新标签）
    4. 在 retrieve_node 中决定是否需要跳过检索
    5. 在 _route 中新增 if 分支
    6. 在 build_graph 中调用 g.add_node() 注册节点
    7. 在 g.add_conditional_edges 的映射表中添加新映射
    8. 在 g.add_edge 中添加 new_node → final_answer_node 的边
    """

    g = StateGraph(AgentState)

    # ---- 注册所有节点 ----
    g.add_node("classify_intent_node", classify_intent_node)
    g.add_node("retrieve_node", retrieve_node)
    g.add_node("route_tool_node", route_tool_node)
    g.add_node("policy_qa_node", policy_qa_node)
    g.add_node("eligibility_check_node", eligibility_check_node)
    g.add_node("checklist_generation_node", checklist_generation_node)
    g.add_node("version_compare_node", version_compare_node)
    g.add_node("web_ingestion_node", web_ingestion_node)
    g.add_node("final_answer_node", final_answer_node)

    # ---- 固定边：无条件顺序执行 ----
    g.add_edge(START, "classify_intent_node")
    g.add_edge("classify_intent_node", "retrieve_node")
    g.add_edge("retrieve_node", "route_tool_node")

    # ---- 条件边：根据 intent 分发到不同的工具节点 ----
    g.add_conditional_edges(
        "route_tool_node",    # 从哪个节点出发
        _route,               # 决策函数：state → 目标节点名
        {
            # 映射表：_route 的返回值 → 实际节点函数名
            "policy_qa_node": "policy_qa_node",
            "eligibility_check_node": "eligibility_check_node",
            "checklist_generation_node": "checklist_generation_node",
            "version_compare_node": "version_compare_node",
            "web_ingestion_node": "web_ingestion_node",
        },
    )

    # ---- 所有工具节点执行完后都汇聚到 final_answer_node ----
    g.add_edge("policy_qa_node", "final_answer_node")
    g.add_edge("eligibility_check_node", "final_answer_node")
    g.add_edge("checklist_generation_node", "final_answer_node")
    g.add_edge("version_compare_node", "final_answer_node")
    g.add_edge("web_ingestion_node", "final_answer_node")

    # ---- 最终节点 → 结束 ----
    g.add_edge("final_answer_node", END)

    return g.compile()


def get_graph():
    """获取编译后的图（单例模式）。

    为什么缓存？
    编译图是有开销的（验证、优化内部结构），
    每次请求重新编译既浪费 CPU 也不必要。
    """

    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# =========================================================
# 格式化工具：把结构化结果转为用户可读的自然语言答案
# =========================================================
# 为什么要单独写格式化函数？
# - 节点函数保持简洁：只负责"获取数据"，不负责"排版"
# - 格式化逻辑可以独立修改（例如调整措辞、增加 emoji），不影响节点行为
# - 方便做 A/B 测试：两种不同的排版方式可以在格式化函数里切换


def _format_rule_eligibility_answer(
    rule_result: Dict[str, Any],
) -> str:
    """格式化规则判断结果（v2 增强版 eligibility 输出）。

    输出结构：
    【资格判断结论】xxx
    （奖学金等级：xxx）
    【识别到的用户信息】
      - 年级：xxx
      - 成绩排名：前 xx%
      - ...
    【判断原因】
      - ...
    【政策依据】
      - ...
    【仍需补充信息】
      - ...
    【注意】...
    """

    decision = rule_result.get("decision", "need_more_info")
    decision_map = {
        "eligible": "满足申请条件",
        "not_eligible": "不满足申请条件",
        "need_more_info": "信息不足，无法完整判断",
    }
    decision_text = decision_map.get(decision, "结论未知")

    parts = [f"【资格判断结论】{decision_text}"]

    # 奖学金等级（仅 eligible 时有）
    if rule_result.get("level"):
        level_map = {
            "first_class": "一等学业奖学金",
            "second_class": "二等学业奖学金",
        }
        level_text = level_map.get(rule_result["level"], rule_result["level"])
        parts.append(f"（奖学金等级：{level_text}）")

    # 识别到的用户信息
    profile = rule_result.get("profile", {})
    parts.append("\n【识别到的用户信息】")
    info_items = []
    if profile.get("grade"):
        info_items.append(f"年级：{profile['grade']}")
    if profile.get("rank_percent") is not None:
        info_items.append(f"成绩排名：前 {profile['rank_percent']}%")
    if profile.get("has_failed_course"):
        info_items.append(f"挂科/不及格课程数：{profile.get('failed_courses', 0)}")
    if profile.get("has_discipline"):
        info_items.append("有处分记录")
    if profile.get("has_academic_misconduct"):
        info_items.append("有学术不端记录")
    if profile.get("has_fake_material"):
        info_items.append("有材料造假记录")
    if profile.get("tutor_negative"):
        info_items.append("导师评价不合格")
    if info_items:
        for item in info_items:
            parts.append(f"  - {item}")
    else:
        parts.append("  （未能从问题中识别到明确的用户信息）")

    # 判断原因（来自规则引擎）
    reasons = rule_result.get("reasons", [])
    if reasons:
        parts.append("\n【判断原因】")
        for r in reasons:
            parts.append(f"  - {r}")

    # LLM 提供的政策原文引用
    llm_evidence = rule_result.get("llm_evidence", [])
    if llm_evidence:
        parts.append("\n【政策依据】")
        for e in llm_evidence:
            parts.append(f"  - {e}")

    # 缺失信息
    missing = rule_result.get("missing_information", [])
    if missing:
        parts.append("\n【仍需补充信息】")
        for m in missing:
            parts.append(f"  - {m}")

    # 免责声明
    parts.append(
        "\n【注意】以上判断基于确定性规则和已入库的政策文件。"
        "实际申请结果以学校研究生院最终审核为准。"
    )

    return "\n".join(parts)


def _format_eligibility_answer(result: Dict[str, Any]) -> str:
    """保留的旧版格式化函数，用于 tools.py 中 LLM 直接判断的结果。"""
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
    """格式化材料清单答案。"""
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
    """格式化版本对比答案。"""
    if "error" in result:
        return (
            f"版本对比失败："
            f"{result.get('error') or result.get('summary', '')}"
        )
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
