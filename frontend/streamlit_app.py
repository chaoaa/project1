"""Streamlit Web UI —— 高校政策 RAG Agent 控制台（v2 重构版）。

=============================================================================
重构目标
=============================================================================
1. 固定视口布局：左侧知识库管理 + 右侧问答工作区，各自内部滚动。
2. 切换模式不阻塞页面（session_state 缓存，不重复请求后端）。
3. 浅色 SaaS 控制台风格，白色卡片 + 浅灰背景。
4. 所有耗时操作只显示局部 spinner。

=============================================================================
Streamlit 的执行模型（理解这个才能写好 Streamlit）
=============================================================================
每次你在页面点击按钮/下拉框/输入框，整个脚本从上到下重新执行一次。
所以：
- 不要在主流程里做任何 HTTP 请求（除了首次加载缓存）
- 所有重操作放在按钮的回调分支里
- 用 st.session_state 保存跨重跑的状态
=============================================================================
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ============================================================================
# 常量
# ============================================================================

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120

MODE_MAP = {
    "普通 RAG 问答": ("rag", "/api/chat/query"),
    "Agent 问答（推荐）": ("agent", "/api/chat/agent"),
    "仅检索调试": ("retrieve", "/api/chat/retrieve"),
}

# ============================================================================
# Session State 初始化（只在首次加载时执行）
# ============================================================================


def init_session_state() -> None:
    """初始化所有 session_state 变量。

    Streamlit 的 session_state 在脚本重跑时保持值不变，
    所以我们用它来缓存后端数据，避免每次交互都请求 API。
    """
    defaults = {
        "backend_url": DEFAULT_BACKEND,
        "selected_mode": "Agent 问答（推荐）",
        "health_cache": None,       # 缓存健康检查结果
        "documents_cache": None,    # 缓存文档列表
        "last_response": None,      # 最近一次问答响应
        "last_error": None,         # 最近一次错误
        "is_submitting": False,     # 是否正在提交（用于局部 spinner）
        "upload_success": None,     # 上传成功消息
        "upload_error": None,       # 上传失败消息
        "ingest_success": None,     # URL 入库成功消息
        "ingest_error": None,       # URL 入库失败消息
        "last_question": "",        # 上一次提交的问题（保留在输入框）
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================================
# CSS 注入（固定视口布局 + SaaS 浅色风格）
# ============================================================================


def inject_css() -> None:
    """注入全局 CSS，实现固定视口布局和 SaaS 控制台风格。

    配色方案：
    - 侧边栏：深色 (#1e293b) + 白色文字，与主区域形成清晰视觉分隔
    - 主区域背景：暖灰 (#f0f2f5)，让白色卡片浮起来
    - 卡片：白色 + 可见阴影 (0 2px 8px rgba(0,0,0,0.08))
    - 强调色：蓝色 #3b82f6 / 绿色 #10b981 / 琥珀 #f59e0b / 红色 #ef4444

    注意：Streamlit 不是专业前端框架，CSS 选择器依赖其内部 DOM 结构。
    如果 Streamlit 版本升级导致选择器失效，需要调整下面的 CSS。
    """

    st.markdown(
        """
<style>
/* ===== 全局 ===== */
body, .stApp {
    background: #f0f2f5;
    color: #1e293b;
}

/* 隐藏 Streamlit 默认顶部栏 */
header[data-testid="stHeader"] {
    display: none;
}

/* 隐藏 Streamlit 默认 footer */
footer {
    display: none;
}

/* 主内容区 padding 收紧 */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ===== 深色侧边栏 ===== */
[data-testid="stSidebar"] {
    background: #1e293b;
    min-width: 340px !important;
    max-width: 380px !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1.2rem !important;
    height: 100vh;
    overflow-y: auto !important;
}

/* 侧边栏内所有文字默认浅色 */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] div {
    color: #cbd5e1 !important;
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    padding-bottom: 0;
    margin-bottom: 0.25rem;
}

section[data-testid="stSidebar"] h2 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-top: 1rem;
}

section[data-testid="stSidebar"] h3 {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* 侧边栏输入框 */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: #64748b !important;
}

/* 侧边栏按钮 */
section[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] button[kind="secondary"] {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #475569 !important;
}

/* 侧边栏分割线 */
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
    margin: 0.75rem 0 !important;
}

/* 侧边栏内上传组件 */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #334155 !important;
    border: 1px dashed #475569 !important;
    border-radius: 8px !important;
}

/* 侧边栏 metric 组件 */
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1rem !important;
}

section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.7rem !important;
}

/* 侧边栏成功/错误消息保持可读 */
section[data-testid="stSidebar"] .stAlert {
    background: transparent !important;
}

/* 文档列表滚动容器 */
.doc-list-container {
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 0.5rem;
    font-size: 0.8rem;
    background: #0f172a;
}

.doc-list-container .doc-item {
    padding: 0.4rem 0;
    border-bottom: 1px solid #334155;
    color: #cbd5e1;
}

.doc-list-container .doc-item:last-child {
    border-bottom: none;
}

.doc-list-container strong {
    color: #f1f5f9;
}

/* ===== 主区域 ===== */
.main-content {
    max-width: 100%;
}

/* 顶部 header 指标 */
.header-metric-item {
    background: #FFFFFF;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.header-metric-item .metric-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
}

.header-metric-item .metric-label {
    font-size: 0.75rem;
    color: #64748b;
}

/* ===== 卡片通用 ===== */
.result-card {
    background: #FFFFFF;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.result-card .card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

/* 结果区滚动容器 */
.results-scroll {
    max-height: calc(100vh - 340px);
    min-height: 200px;
    overflow-y: auto;
    padding-right: 0.3rem;
}

/* 自定义滚动条 */
.results-scroll::-webkit-scrollbar,
.doc-list-container::-webkit-scrollbar {
    width: 6px;
}

.results-scroll::-webkit-scrollbar-track,
.doc-list-container::-webkit-scrollbar-track {
    background: transparent;
    border-radius: 3px;
}

.results-scroll::-webkit-scrollbar-thumb,
.doc-list-container::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

.results-scroll::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* 主区域滚动条 */
[data-testid="stSidebar"] .block-container::-webkit-scrollbar {
    width: 4px;
}

[data-testid="stSidebar"] .block-container::-webkit-scrollbar-track {
    background: transparent;
}

[data-testid="stSidebar"] .block-container::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 2px;
}

/* ===== Badge ===== */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.3rem;
}

.badge-blue {
    background: #dbeafe;
    color: #1d4ed8;
}

.badge-green {
    background: #d1fae5;
    color: #059669;
}

.badge-red {
    background: #fee2e2;
    color: #dc2626;
}

.badge-yellow {
    background: #fef3c7;
    color: #d97706;
}

.badge-gray {
    background: #e2e8f0;
    color: #475569;
}

/* ===== 按钮覆盖 ===== */
div.stButton > button[kind="primary"] {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
}

/* 确保 Streamlit 的 spinner 不阻塞整个页面 */
.stSpinner {
    display: inline-block !important;
}

/* 清空按钮特殊样式 */
.clear-btn button {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    border-radius: 8px !important;
}

.clear-btn button:hover {
    background: #e2e8f0 !important;
    color: #475569 !important;
}

/* ===== 空状态 ===== */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
    color: #94a3b8;
}

.empty-state .empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.empty-state p {
    color: #94a3b8;
    margin: 0.25rem 0;
}

/* ===== 输入框区域 ===== */
.question-area {
    margin-bottom: 0.5rem;
}

/* 让 expander 更紧凑 */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    padding: 0.5rem 0.75rem !important;
}

/* ===== 主区域输入框和文本域 ===== */
.main-content textarea,
.main-content input {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

.main-content textarea:focus,
.main-content input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* ===== Radio 按钮美化 ===== */
div[role="radiogroup"] label {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.8rem !important;
    background: #fff !important;
}

div[role="radiogroup"] label:hover {
    border-color: #3b82f6 !important;
}

/* ===== Expander 美化 ===== */
.streamlit-expanderHeader {
    border-radius: 8px !important;
}

.streamlit-expanderHeader:hover {
    background: #f8fafc !important;
}

/* ===== hr 分割线 ===== */
hr, [data-testid="stMarkdown"] hr {
    border-color: #e2e8f0 !important;
    margin: 0.75rem 0 !important;
}

/* ===== 标题和说明文字 ===== */
h1 {
    color: #0f172a !important;
    font-weight: 700 !important;
}

h2 {
    color: #1e293b !important;
}

small, .stCaption {
    color: #64748b !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================================
# API 调用（纯 HTTP 请求，不涉及 UI）
# ============================================================================


def api_health(base: str) -> Dict[str, Any]:
    r = requests.get(f"{base}/api/health", timeout=10)
    r.raise_for_status()
    return r.json()


def api_list_documents(base: str) -> Dict[str, Any]:
    r = requests.get(f"{base}/api/documents", timeout=10)
    r.raise_for_status()
    return r.json()


def api_upload(
    base: str, filename: str, content: bytes, mime: str
) -> Dict[str, Any]:
    files = {"file": (filename, content, mime or "application/octet-stream")}
    r = requests.post(
        f"{base}/api/documents/upload", files=files, timeout=TIMEOUT
    )
    if r.status_code >= 400:
        raise RuntimeError(f"上传失败：{r.status_code} {r.text}")
    return r.json()


def api_ingest_url(base: str, url: str) -> Dict[str, Any]:
    r = requests.post(
        f"{base}/api/documents/ingest-url",
        json={"url": url},
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"URL 入库失败：{r.status_code} {r.text}")
    return r.json()


def api_chat(
    base: str, path: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    r = requests.post(f"{base}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"接口错误：{r.status_code} {r.text}")
    return r.json()


# ============================================================================
# 数据加载（带缓存，避免每次页面重跑都请求后端）
# ============================================================================


def load_health(force: bool = False) -> Optional[Dict[str, Any]]:
    """获取健康检查结果。

    force=False：使用 session_state 缓存，不会请求后端。
    force=True：强制刷新，更新缓存。
    """
    if force or st.session_state.health_cache is None:
        try:
            st.session_state.health_cache = api_health(
                st.session_state.backend_url
            )
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.health_cache = None
            st.session_state.last_error = str(e)
    return st.session_state.health_cache


def load_documents(force: bool = False) -> List[Dict[str, Any]]:
    """获取文档列表。

    force=False：使用 session_state 缓存。
    force=True：强制刷新。
    """
    if force or st.session_state.documents_cache is None:
        try:
            resp = api_list_documents(st.session_state.backend_url)
            st.session_state.documents_cache = resp.get("documents", [])
        except Exception:
            st.session_state.documents_cache = []
    return st.session_state.documents_cache or []


# ============================================================================
# 渲染辅助（纯 UI，不请求后端）
# ============================================================================


def render_badge(
    text: str, color: str = "gray", inline: bool = True
) -> str:
    """生成一个 HTML badge 标签。

    返回 HTML 字符串，供 st.markdown(unsafe_allow_html=True) 使用。
    """
    return (
        f'<span class="badge badge-{color}">'
        f"{text}"
        f"</span>"
    )


def render_empty_state() -> None:
    """结果区空状态占位。"""
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">&#128269;</div>'
        "<p>提交问题后，回答将显示在这里</p>"
        '<p style="font-size:0.8rem;">'
        "支持普通 RAG 问答 / Agent 智能问答 / 仅检索调试</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# 渲染：Sidebar
# ============================================================================


def render_sidebar() -> None:
    """渲染左侧知识库管理面板。

    所有操作通过按钮触发，按需请求后端。
    session_state 缓存 health 和 documents，避免频繁请求。
    """

    backend = st.session_state.backend_url

    with st.sidebar:
        # ---- 品牌区 ----
        st.title("高校政策 Agent")
        st.caption("RAG · Hybrid Search · LangGraph")

        st.markdown("---")

        # ---- 服务连接卡片 ----
        st.subheader("服务连接")
        new_backend = st.text_input(
            "后端地址",
            value=backend,
            help="FastAPI 后端地址",
            label_visibility="collapsed",
            placeholder="http://localhost:8000",
        )
        if new_backend != backend:
            st.session_state.backend_url = new_backend
            st.session_state.health_cache = None
            st.session_state.documents_cache = None
            st.rerun()

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("检查连接", use_container_width=True):
                load_health(force=True)
                load_documents(force=True)
                st.rerun()

        # 状态展示
        health = load_health()
        if health:
            with col2:
                st.markdown(
                    f'<span class="badge badge-green">'
                    f"已连接</span>",
                    unsafe_allow_html=True,
                )
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Qdrant",
                "在线" if health.get("qdrant_connected") else "离线",
            )
            c2.metric("文档", health.get("document_count", 0))
            c3.metric("Chunks", health.get("chunk_count", 0))
        elif st.session_state.last_error:
            st.error(st.session_state.last_error)
        else:
            with col2:
                st.markdown(
                    '<span class="badge badge-gray">'
                    '未连接</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ---- 文件上传卡片 ----
        st.subheader("上传政策文件")
        uploaded = st.file_uploader(
            "支持 .pdf / .docx / .txt / .md / .html",
            type=["pdf", "docx", "txt", "md", "html", "htm"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.button(
            "上传并入库",
            use_container_width=True,
            disabled=uploaded is None,
        ):
            if uploaded is not None:
                with st.spinner("解析、切分、向量化中…"):
                    try:
                        res = api_upload(
                            backend,
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                        st.session_state.upload_success = (
                            f"已入库：{res.get('filename')} "
                            f"({res.get('chunk_count')} 个 chunk)"
                        )
                        st.session_state.upload_error = None
                        load_documents(force=True)
                        st.rerun()
                    except Exception as e:
                        st.session_state.upload_error = str(e)
                        st.session_state.upload_success = None
                        st.rerun()

        if st.session_state.upload_success:
            st.success(st.session_state.upload_success)
            st.session_state.upload_success = None
        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)
            st.session_state.upload_error = None

        st.markdown("---")

        # ---- URL 政策采集卡片 ----
        st.subheader("URL 政策采集")
        ingest_url = st.text_input(
            "输入政策网页 URL",
            value="",
            placeholder="https://xxx.edu.cn/policy/...",
            label_visibility="collapsed",
            key="sidebar_ingest_url",
        )
        if st.button(
            "采集网页并入库",
            use_container_width=True,
            disabled=not ingest_url.strip(),
        ):
            with st.spinner("抓取、清洗、入库中…"):
                try:
                    res = api_ingest_url(backend, ingest_url.strip())
                    st.session_state.ingest_success = (
                        f"已入库：{res.get('title') or res.get('filename')} "
                        f"({res.get('chunk_count')} 个 chunk)"
                    )
                    st.session_state.ingest_error = None
                    load_documents(force=True)
                    st.rerun()
                except Exception as e:
                    st.session_state.ingest_error = str(e)
                    st.session_state.ingest_success = None
                    st.rerun()

        if st.session_state.ingest_success:
            st.success(st.session_state.ingest_success)
            st.session_state.ingest_success = None
        if st.session_state.ingest_error:
            st.error(st.session_state.ingest_error)
            st.session_state.ingest_error = None

        st.markdown("---")

        # ---- 已入库文档卡片 ----
        st.subheader("已入库文档")
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("刷新", use_container_width=True, key="refresh_docs"):
                load_documents(force=True)
                st.rerun()

        docs = load_documents()
        if not docs:
            st.caption("暂无文档，请上传或采集。")
        else:
            # 将文档列表放入自定义滚动容器
            html_parts = ['<div class="doc-list-container">']
            for d in docs:
                status_badge = (
                    "green" if d.get("status") == "indexed" else "red"
                )
                html_parts.append(
                    f'<div class="doc-item">'
                    f'<strong>{d.get("filename", "")}</strong>'
                    f'<br>'
                    f'<span style="color:#94a3b8;font-size:0.72rem;">'
                    f'id: {d.get("document_id", "")[:20]}...</span>'
                    f'<br>'
                    f'chunks: {d.get("chunk_count", 0)} '
                    f'· <span class="badge badge-{status_badge}">'
                    f'{d.get("status", "")}</span>'
                    f"</div>"
                )
            html_parts.append("</div>")
            st.markdown("\n".join(html_parts), unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "本系统仅基于知识库回答，"
            "未命中或低置信度时会拒答。"
        )


# ============================================================================
# 渲染：主区域 Header
# ============================================================================


def render_header() -> None:
    """渲染主区域顶部：标题 + 模式选择 + 快捷指标。"""

    st.title("高校政策问答与办事 Agent")
    st.caption(
        "基于 RAG、Hybrid Search 与 LangGraph "
        "的高校政策问答和办事辅助系统"
    )

    # 三个快捷指标
    docs = load_documents()
    health = load_health()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    with c1:
        mode_key = MODE_MAP[st.session_state.selected_mode][0]
        st.markdown(
            '<div class="header-metric-item">'
            f'<div class="metric-value" style="font-size:0.9rem;">'
            f'{mode_key.upper()}</div>'
            f'<div class="metric-label">当前模式</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="header-metric-item">'
            f'<div class="metric-value">{len(docs)}</div>'
            f'<div class="metric-label">文档数</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        chunk_count = (
            health.get("chunk_count", 0) if health else 0
        )
        st.markdown(
            '<div class="header-metric-item">'
            f'<div class="metric-value">{chunk_count}</div>'
            f'<div class="metric-label">Chunks</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # 模式选择（不放在 form 里，切换时不触发后端请求）
    st.markdown("---")
    mode_label = st.radio(
        "选择问答模式",
        options=list(MODE_MAP.keys()),
        horizontal=True,
        index=list(MODE_MAP.keys()).index(
            st.session_state.selected_mode
        ),
        label_visibility="collapsed",
    )
    if mode_label != st.session_state.selected_mode:
        st.session_state.selected_mode = mode_label
        st.rerun()


# ============================================================================
# 渲染：提问面板
# ============================================================================


def render_question_panel() -> Optional[Dict[str, Any]]:
    """渲染问题输入区和高级参数。

    使用 st.form 包裹提交按钮，避免每次输入都触发重跑。
    返回 None 表示未提交；返回 dict 表示构建好的 payload。
    """

    mode_label = st.session_state.selected_mode
    mode_key, mode_path = MODE_MAP[mode_label]

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "请输入你的问题",
            height=100,
            placeholder="例如：研究生学业奖学金申请条件是什么？",
            label_visibility="collapsed",
        )

        with st.expander("Agent 高级参数（可选）"):
            st.caption(
                "当 Agent 识别为 资格判断 / 版本对比 / URL 采集 时使用。"
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**用户档案（资格判断）**")
                profile_text = st.text_area(
                    "JSON 格式",
                    value="",
                    height=80,
                    placeholder='{"年级":"研二","成绩排名":"前20%"}',
                    label_visibility="collapsed",
                )
            with c2:
                st.markdown("**版本对比（选择文档）**")
                docs = load_documents()
                doc_options = [
                    (d.get("filename", ""), d.get("document_id", ""))
                    for d in docs
                ]
                old_label = st.selectbox(
                    "旧版政策",
                    options=[("(不选)", "")] + doc_options,
                    format_func=lambda x: x[0] if x else "",
                )
                new_label = st.selectbox(
                    "新版政策",
                    options=[("(不选)", "")] + doc_options,
                    format_func=lambda x: x[0] if x else "",
                )

        c1, c2 = st.columns([3, 1])
        with c1:
            submitted = st.form_submit_button(
                "提交问题", type="primary", use_container_width=True
            )
        with c2:
            clear = st.form_submit_button(
                "清空结果", use_container_width=True
            )

    if clear:
        st.session_state.last_response = None
        st.session_state.last_error = None
        st.rerun()

    if not submitted:
        return None

    if not question.strip():
        st.session_state.last_error = "请输入问题"
        st.rerun()

    # 组装 payload（后端 ChatRequest schema）
    payload: Dict[str, Any] = {
        "question": question.strip(),
        "mode": mode_key,
    }

    if mode_key == "agent":
        if profile_text.strip():
            try:
                payload["user_profile"] = json.loads(profile_text)
            except Exception:
                st.warning("用户档案 JSON 解析失败，已忽略。")

        old_id = (
            old_label[1] if isinstance(old_label, tuple) else ""
        )
        new_id = (
            new_label[1] if isinstance(new_label, tuple) else ""
        )
        if old_id:
            payload["old_document_id"] = old_id
        if new_id:
            payload["new_document_id"] = new_id

    # 执行请求
    with st.spinner("思考中…"):
        try:
            resp = api_chat(
                st.session_state.backend_url, mode_path, payload
            )
            st.session_state.last_response = resp
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.last_response = None
            st.session_state.last_error = str(e)

    st.session_state.last_question = question
    st.rerun()
    return None  # unreachable, st.rerun() 会阻止后续代码执行


# ============================================================================
# 渲染：结果面板
# ============================================================================


def render_results_panel() -> None:
    """渲染右侧下方的结果展示区（可滚动）。

    包含：回答卡片 / 工具结果卡片 / 引用来源卡片 / 检索调试卡片。
    使用自定义 div 包裹，CSS 限制最大高度并启用内部滚动。
    """

    # 开始滚动容器
    st.markdown(
        '<div class="results-scroll" id="results-scroll">',
        unsafe_allow_html=True,
    )

    resp = st.session_state.last_response
    error = st.session_state.last_error

    if error:
        st.error(error)
        st.session_state.last_error = None

    if resp is None and not error:
        render_empty_state()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if resp is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    mode_label = st.session_state.selected_mode
    mode_key = MODE_MAP[mode_label][0]

    # ---- 检索调试模式 ----
    if mode_key == "retrieve":
        render_retrieval_results(resp)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---- 普通回答 / Agent 模式 ----
    render_answer_card(resp)

    if mode_key == "agent":
        render_tool_result_card(resp)

    render_citations_card(resp.get("citations", []))

    st.markdown("</div>", unsafe_allow_html=True)


def render_answer_card(resp: Dict[str, Any]) -> None:
    """渲染回答结果卡片。"""

    confidence = resp.get("confidence", "low")
    refused = resp.get("refused", False)
    intent = resp.get("intent", "")
    tool_name = resp.get("tool_name", "")

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # 标题行
    st.markdown(
        '<div class="card-title">回答结果</div>',
        unsafe_allow_html=True,
    )

    # Badge 行
    badges = []
    if intent:
        badges.append(render_badge(intent, "blue"))
    if tool_name:
        badges.append(render_badge(tool_name, "gray"))
    if confidence:
        color = (
            "green"
            if confidence == "high"
            else "yellow"
            if confidence == "medium"
            else "red"
        )
        badges.append(
            render_badge(f"置信度: {confidence.upper()}", color)
        )
    if refused:
        badges.append(render_badge("已拒答", "red"))
    else:
        badges.append(render_badge("已回答", "green"))

    st.markdown(
        '<div style="margin-bottom:0.8rem;">'
        + " ".join(badges)
        + "</div>",
        unsafe_allow_html=True,
    )

    # 回答内容
    if refused:
        st.warning(resp.get("answer", ""))
    else:
        answer = resp.get("answer", "")
        if answer:
            st.markdown(answer)

    # 如果 tool_result 包含下载链接，在这里额外展示
    tool_result = resp.get("tool_result")
    if tool_result and isinstance(tool_result, dict):
        gen_file = tool_result.get("generated_file")
        if gen_file and gen_file.get("download_url"):
            download_url = (
                st.session_state.backend_url.rstrip("/")
                + gen_file["download_url"]
            )
            st.markdown(
                f":page_facing_up: **已生成 Word 材料清单**："
                f"[点击下载 `{gen_file['filename']}`]"
                f"({download_url})"
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_tool_result_card(resp: Dict[str, Any]) -> None:
    """渲染 Agent 工具调用结果卡片。

    先展示结构化摘要，完整 JSON 放入 expander 默认折叠。
    """

    tool_result = resp.get("tool_result")
    if not tool_result or not isinstance(tool_result, dict):
        return

    tool_name = resp.get("tool_name", "")
    intent = resp.get("intent", "")

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">工具调用结果</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<span class="badge badge-blue">意图: {intent}</span> '
        f'<span class="badge badge-gray">工具: {tool_name}</span>',
        unsafe_allow_html=True,
    )

    # ---- 根据 tool_name 展示结构化摘要 ----
    if tool_name == "rule_based_scholarship_checker":
        _render_eligibility_summary(tool_result)
    elif tool_name == "generate_checklist":
        _render_checklist_summary(tool_result)
    elif tool_name == "ingest_policy_from_url":
        _render_ingestion_summary(tool_result)

    # JSON 原文折叠
    with st.expander("完整 JSON（调试用）", expanded=False):
        st.code(
            json.dumps(tool_result, ensure_ascii=False, indent=2),
            language="json",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_eligibility_summary(tr: Dict[str, Any]) -> None:
    """规则化资格判断的结构化摘要。"""
    decision = tr.get("decision", "unknown")
    decision_map = {
        "eligible": ("green", "满足条件"),
        "not_eligible": ("red", "不符合条件"),
        "need_more_info": ("yellow", "需要更多信息"),
    }
    color, label = decision_map.get(decision, ("gray", decision))

    st.markdown(f"**结论**：{render_badge(label, color)}", unsafe_allow_html=True)

    if tr.get("level"):
        level_names = {
            "first_class": "一等学业奖学金",
            "second_class": "二等学业奖学金",
        }
        st.markdown(
            f"**等级**：{level_names.get(tr['level'], tr['level'])}"
        )

    if tr.get("reasons"):
        st.markdown("**判断原因**：")
        for r in tr["reasons"]:
            st.markdown(f"- {r}")

    if tr.get("missing_information"):
        st.markdown("**需补充信息**：")
        for m in tr["missing_information"]:
            st.markdown(f"- {m}")


def _render_checklist_summary(tr: Dict[str, Any]) -> None:
    """材料清单的结构化摘要。"""
    if tr.get("task_name"):
        st.markdown(f"**任务**：{tr['task_name']}")

    materials = tr.get("materials") or []
    steps = tr.get("steps") or []
    notes = tr.get("notes") or []

    if materials:
        st.markdown(f"**所需材料**：{len(materials)} 项")
    if steps:
        st.markdown(f"**办理步骤**：{len(steps)} 步")
    if notes:
        st.markdown(f"**注意事项**：{len(notes)} 条")

    if tr.get("generated_file_error"):
        st.warning(f"Word 生成失败：{tr['generated_file_error']}")


def _render_ingestion_summary(tr: Dict[str, Any]) -> None:
    """URL 入库的结构化摘要。"""
    if tr.get("title"):
        st.markdown(f"**标题**：{tr['title']}")
    if tr.get("filename"):
        st.markdown(f"**文件**：{tr['filename']}")
    if tr.get("source_url"):
        st.markdown(f"**来源**：{tr['source_url']}")
    if tr.get("chunk_count"):
        st.markdown(f"**Chunks**：{tr['chunk_count']}")
    if tr.get("status"):
        st.markdown(
            f"**状态**：{render_badge(tr['status'], 'green')}",
            unsafe_allow_html=True,
        )


def render_citations_card(citations: List[Dict[str, Any]]) -> None:
    """渲染引用来源卡片。多条引用时使用 expander 折叠，防止撑开页面。"""

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">引用来源</div>',
        unsafe_allow_html=True,
    )

    if not citations:
        st.info("无引用来源。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 使用 Streamlit 原生 expander 逐个展示
    for i, c in enumerate(citations, 1):
        score = c.get("score", 0)
        filename = c.get("filename", "")
        chunk_idx = c.get("chunk_index", "")

        with st.expander(
            f"[{i}] {filename} · score={score:.4f} · chunk={chunk_idx}",
            expanded=(i == 1 and len(citations) <= 3),
        ):
            st.caption(
                f"chunk_id: {c.get('chunk_id', '')}",
            )
            text = c.get("text", "")
            if len(text) > 500:
                st.write(text[:500] + "...")
            else:
                st.write(text)

    st.markdown("</div>", unsafe_allow_html=True)


def render_retrieval_results(resp: Dict[str, Any]) -> None:
    """渲染仅检索调试模式的结果。"""

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">检索结果</div>',
        unsafe_allow_html=True,
    )

    hits = resp.get("chunks", [])
    st.caption(f"查询：{resp.get('query', '')}  |  共 {resp.get('total', 0)} 条命中")

    if not hits:
        st.info("无检索结果。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for i, h in enumerate(hits, 1):
        with st.expander(
            f"[{i}] {h.get('filename', '')} "
            f"· final={h.get('final_score', 0):.4f} "
            f"· vec={h.get('vector_score', 0):.4f} "
            f"· bm25={h.get('bm25_score', 0):.4f}",
            expanded=(i == 1),
        ):
            st.caption(f"chunk_id: {h.get('chunk_id', '')}")
            text = h.get("text", "")
            if len(text) > 400:
                st.write(text[:400] + "...")
            else:
                st.write(text)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    """应用入口。

    执行顺序：
    1. 页面配置（必须第一个 st. 调用）
    2. 初始化 session_state
    3. 注入 CSS
    4. 首次加载数据缓存
    5. 渲染左侧栏
    6. 渲染主区域 header + 提问
    7. 渲染结果区
    """

    # ---- 页面级配置（Streamlit 要求第一个 st 调用） ----
    st.set_page_config(
        page_title="高校政策 RAG Agent",
        page_icon=":mortar_board:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- 初始化 ----
    init_session_state()
    inject_css()

    # ---- 首次加载缓存（只请求一次，后续用 session_state） ----
    load_health()
    load_documents()

    # ---- 左侧栏：知识库管理 ----
    render_sidebar()

    # ---- 主区域 ----
    # 为了 CSS 能正确限制结果区高度，用 div 包裹整个主区域
    st.markdown(
        '<div class="main-content">', unsafe_allow_html=True
    )

    render_header()

    st.markdown("---")

    # 提问面板：内部有 st.form，表单提交会触发 rerun
    render_question_panel()

    st.markdown("---")

    # 结果面板：在固定高度的滚动 div 中
    render_results_panel()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
