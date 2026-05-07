"""Streamlit Web UI：本项目唯一的前端入口（偏“控制台”样式，够用即可）。

怎么读这个文件更高效？
1）先翻到 `api_*` 那几个函数——它们只是把 `requests` HTTP 的细节藏起来。
2）再看侧边栏：`api_health`、`api_upload`、`api_list_documents` 三件事。
3）主区域核心是 `payload` JSON 拼装 + `mode_path` 选择端点。

Streamlit 重跑模型（重要）：
每次你在页面点按钮/下拉框，**整个脚本从上到下都会重新执行**。
所以不要在这里存重量级模型；所有重计算都放在 FastAPI。

启动命令：`streamlit run streamlit_app.py`
"""


from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")
# Streamlit 单次请求可能触发：模型思考 + 向量下载冷启动，超时设长一点更稳。
TIMEOUT = 120


# =========================================================
# 后端调用
# =========================================================

def api_health(base: str) -> Dict[str, Any]:
    r = requests.get(f"{base}/api/health", timeout=10)
    r.raise_for_status()
    return r.json()


def api_list_documents(base: str) -> Dict[str, Any]:
    r = requests.get(f"{base}/api/documents", timeout=10)
    r.raise_for_status()
    return r.json()


def api_upload(base: str, filename: str, content: bytes, mime: str) -> Dict[str, Any]:
    files = {"file": (filename, content, mime or "application/octet-stream")}
    r = requests.post(f"{base}/api/documents/upload", files=files, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"上传失败：{r.status_code} {r.text}")
    return r.json()


def api_chat(base: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{base}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"接口错误：{r.status_code} {r.text}")
    return r.json()


# =========================================================
# UI helpers
# =========================================================

CONFIDENCE_COLOR = {"high": "#16a34a", "medium": "#f59e0b", "low": "#dc2626"}


def render_confidence(level: str) -> None:
    color = CONFIDENCE_COLOR.get(level, "#6b7280")
    st.markdown(
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:6px;font-size:13px;'>置信度：{level.upper()}</span>",
        unsafe_allow_html=True,
    )


def render_citations(citations: List[Dict[str, Any]]) -> None:
    if not citations:
        st.info("无引用来源。")
        return
    for i, c in enumerate(citations, 1):
        with st.expander(
            f"[{i}] {c.get('filename','')}  ·  chunk_index={c.get('chunk_index','')}  "
            f"·  score={c.get('score', 0):.4f}",
            expanded=(i == 1),
        ):
            st.caption(f"chunk_id: {c.get('chunk_id','')}")
            st.write(c.get("text", ""))


def render_tool_result(
    intent: Optional[str],
    tool_name: Optional[str],
    tool_result: Optional[Dict[str, Any]],
    backend: str = "",
) -> None:
    """展示 Agent 工具调用结果。

    除了原有的 JSON 折叠面板，v2 新增了：
    1. Word 文件下载链接（当 tool_result 包含 generated_file 时）
    2. 文件生成失败提示（当 tool_result 包含 generated_file_error 时）
    """

    if not tool_result:
        return

    st.markdown("---")
    st.markdown(
        f"**Agent 意图**：`{intent}`    **调用工具**：`{tool_name}`"
    )

    # ---- v2 新增：Word 材料清单下载入口 ----
    # 用 st.markdown 的链接语法实现点击下载，不引入额外 JS 依赖。
    # 下载链接 = 后端地址 + /api/files/download/{filename}
    # 注意：backend 可能以 "/" 结尾，需要 rstrip 处理。
    generated_file = tool_result.get("generated_file") if tool_result else None
    if generated_file and generated_file.get("download_url"):
        download_url = backend.rstrip("/") + generated_file["download_url"]
        st.markdown(
            f":page_facing_up: **已生成 Word 材料清单**："
            f"[点击下载 `{generated_file['filename']}`]({download_url})"
        )

    # ---- v2 新增：文件生成失败提示 ----
    # docx 生成失败不应阻塞整个 Agent 响应，所以这里只显示 warning
    if tool_result and tool_result.get("generated_file_error"):
        st.warning(
            f"Word 文件生成失败：{tool_result['generated_file_error']}"
        )

    # ---- 原有的结构化 JSON 展示 ----
    with st.expander("Tool Result（结构化输出）", expanded=False):
        st.code(
            json.dumps(tool_result, ensure_ascii=False, indent=2),
            language="json",
        )


# =========================================================
# 页面布局
# =========================================================

st.set_page_config(
    page_title="高校政策 RAG Agent",
    page_icon=":mortar_board:",
    layout="wide",
)


# ---- 左侧：配置 + 上传 + 文档列表 ----

with st.sidebar:
    st.title(":mortar_board: 高校政策 Agent")
    st.caption("RAG + LangGraph Tool Calling")

    backend = st.text_input("后端地址", value=DEFAULT_BACKEND, help="FastAPI 后端地址")

    # 健康检查
    st.subheader("服务状态")
    try:
        health = api_health(backend)
        cols = st.columns(2)
        cols[0].metric("Qdrant", "已连接" if health.get("qdrant_connected") else "未连接")
        cols[1].metric("文档数", health.get("document_count", 0))
        st.caption(f"chunks: {health.get('chunk_count', 0)}  ·  v{health.get('version','')}")
    except Exception as e:
        st.error(f"无法连接后端：{e}")

    st.divider()

    # 文件上传
    st.subheader("上传政策文件")
    uploaded = st.file_uploader(
        "支持 .pdf / .docx / .txt / .md / .html",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=False,
    )
    if st.button("上传并入库", type="primary", use_container_width=True, disabled=uploaded is None):
        if uploaded is None:
            st.warning("请先选择文件")
        else:
            with st.spinner("解析、切分、向量化中…"):
                try:
                    res = api_upload(
                        backend,
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                    st.success(
                        f"已入库：{res.get('filename')}（{res.get('chunk_count')} 个 chunk）"
                    )
                except Exception as e:
                    st.error(f"上传失败：{e}")

    st.divider()

    # 已入库文档列表
    st.subheader("已入库文档")
    try:
        docs = api_list_documents(backend).get("documents", [])
    except Exception as e:
        docs = []
        st.error(f"获取文档列表失败：{e}")

    if not docs:
        st.caption("暂无文档")
    else:
        for d in docs:
            st.markdown(
                f"**{d.get('filename','')}**  \n"
                f"- id: `{d.get('document_id','')}`\n"
                f"- chunks: {d.get('chunk_count', 0)}  ·  状态: {d.get('status','')}"
            )

    st.divider()
    st.caption(
        "本系统仅基于知识库回答，未命中或低置信度时会拒答。"
        "示例文件位于项目 `examples/` 目录。"
    )


# ---- 主区域：问答 ----

st.title("高校政策问答与办事 Agent")
st.caption(
    "支持普通 RAG 问答 / Agent 模式（自动识别意图 + 工具调用）/ 仅检索调试模式。"
)

mode_label = st.radio(
    "模式",
    options=[
        "普通 RAG 问答",
        "Agent 问答（推荐）",
        "仅检索调试",
    ],
    horizontal=True,
)
mode_map = {
    "普通 RAG 问答": ("rag", "/api/chat/query"),
    "Agent 问答（推荐）": ("agent", "/api/chat/agent"),
    "仅检索调试": ("retrieve", "/api/chat/retrieve"),
}
mode_key, mode_path = mode_map[mode_label]


with st.form("ask_form", clear_on_submit=False):
    question = st.text_area(
        "请输入你的问题",
        height=110,
        placeholder="例如：研究生学业奖学金申请条件是什么？",
    )

    with st.expander("Agent 高级参数（可选）"):
        st.caption("当 Agent 识别为 资格判断 / 版本对比 时使用。")

        # 资格判断 - 用户档案
        st.markdown("**用户档案（资格判断时使用）**")
        profile_text = st.text_area(
            "JSON 格式，例如：{\"年级\":\"研二\",\"成绩排名\":\"前20%\",\"挂科情况\":\"有1门不及格\"}",
            value="",
            height=80,
        )

        # 版本对比 - 文档 ID
        doc_options = [(d.get("filename", ""), d.get("document_id", "")) for d in docs]
        old_label = st.selectbox(
            "旧版政策（version_compare 时使用）",
            options=[("(不选)", "")] + doc_options,
            format_func=lambda x: x[0] if x else "",
            index=0,
        )
        new_label = st.selectbox(
            "新版政策（version_compare 时使用）",
            options=[("(不选)", "")] + doc_options,
            format_func=lambda x: x[0] if x else "",
            index=0,
        )

    submitted = st.form_submit_button("提交", type="primary")


if submitted:
    if not question.strip():
        st.warning("请输入问题")
    else:
        # 组装 payload
        payload: Dict[str, Any] = {"question": question.strip(), "mode": mode_key}

        if mode_key == "agent":
            if profile_text.strip():
                try:
                    payload["user_profile"] = json.loads(profile_text)
                except Exception:
                    st.warning("用户档案 JSON 解析失败，已忽略。")
            old_id = old_label[1] if isinstance(old_label, tuple) else ""
            new_id = new_label[1] if isinstance(new_label, tuple) else ""
            if old_id:
                payload["old_document_id"] = old_id
            if new_id:
                payload["new_document_id"] = new_id

        with st.spinner("思考中…"):
            try:
                resp = api_chat(backend, mode_path, payload)
            except Exception as e:
                st.error(f"调用失败：{e}")
                resp = None

        if resp is not None:
            if mode_key == "retrieve":
                st.subheader("检索结果")
                hits = resp.get("chunks", [])
                st.caption(f"共 {resp.get('total', 0)} 条")
                for i, h in enumerate(hits, 1):
                    with st.expander(
                        f"[{i}] {h.get('filename','')}  ·  final={h.get('final_score',0):.4f}  "
                        f"·  vec={h.get('vector_score',0):.4f}  ·  bm25={h.get('bm25_score',0):.4f}",
                        expanded=(i == 1),
                    ):
                        st.caption(f"chunk_id: {h.get('chunk_id','')}")
                        st.write(h.get("text", ""))
            else:
                # 答案
                st.subheader("回答")
                if resp.get("refused"):
                    st.warning(resp.get("answer", ""))
                else:
                    st.markdown(resp.get("answer", ""))

                cols = st.columns([1, 1, 4])
                with cols[0]:
                    render_confidence(resp.get("confidence", "low"))
                with cols[1]:
                    if resp.get("refused"):
                        st.markdown(
                            "<span style='background:#dc2626;color:white;padding:2px 8px;"
                            "border-radius:6px;font-size:13px;'>已拒答</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<span style='background:#2563eb;color:white;padding:2px 8px;"
                            "border-radius:6px;font-size:13px;'>已回答</span>",
                            unsafe_allow_html=True,
                        )

                # Agent 工具结果
                if mode_key == "agent":
                    render_tool_result(
                        resp.get("intent"),
                        resp.get("tool_name"),
                        resp.get("tool_result"),
                        backend=backend,
                    )

                # 引用
                st.markdown("---")
                st.subheader("引用来源")
                render_citations(resp.get("citations", []))
