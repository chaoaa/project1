"""FastAPI 应用入口（Web 服务器的“总指挥”）。

本文件做了什么（按阅读顺序理解）？
1. 定义 `lifespan`：在服务**启动前后**执行的钩子——适合做“热身”（例如从 Qdrant 拉全体 chunk，
   重建进程内的 BM25 索引）。
2. 创建 `FastAPI(app)`：注册路由、Swagger 文档、跨域策略等。
3. `include_router`：把 `api/` 下面拆出去的各个路由模块挂载到统一前缀上。

为什么是 FastAPI？
- 现代 Python Web 框架，自带 OpenAPI (`/docs`)，适合你做接口调试与演示。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import chat as chat_api
from app.api import documents as documents_api
from app.api import files as files_api
from app.api import health as health_api
from app.config import settings
from app.services.retriever import rebuild_bm25_from_qdrant
from app.services.vector_store import get_vector_store
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子（ startup / shutdown ）。

    - `yield` 之前：**启动瞬间**跑一次（适合做连接检查、预热缓存）。
    - `yield` 之后：进程即将退出时跑一次（当前项目只打日志）。

    为什么启动时要 `rebuild_bm25_from_qdrant`？
    BM25 索引存在**本进程内存**里；重启后端后内存空了，需要从向量库把文本再拉一遍重建索引，
    否则关键词检索会失效（向量检索仍可用）。
    """
    logger.info("=" * 60)
    logger.info(f"School Policy Agent backend starting (v{__version__})")
    logger.info(f"LLM model: {settings.llm_model} @ {settings.llm_base_url}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(
        f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port} / "
        f"collection={settings.qdrant_collection}"
    )
    logger.info("=" * 60)

    try:
        if get_vector_store().is_connected():
            count = rebuild_bm25_from_qdrant()
            logger.info(f"warmup BM25 with {count} chunks from qdrant")
        else:
            logger.warning(
                "Qdrant 暂时无法连接，请确认已运行 docker compose up -d qdrant。"
            )
    except Exception as e:
        logger.warning(f"warmup failed: {e}")

    yield  # 从这里开始，FastAPI 正常对外提供 HTTP 服务

    logger.info("School Policy Agent backend stopped.")


app = FastAPI(
    title="School Policy Agent API",
    description=(
        "基于 RAG + LangGraph + Tool Calling 的高校政策问答与办事 Agent 系统"
    ),
    version=__version__,
    lifespan=lifespan,
)

# CORS：允许浏览器里运行的前端（Streamlit 或本地静态页）跨域访问本 API。
# 生产环境建议把 allow_origins 改成你的前端域名白名单，而不是 "*"。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_api.router)
app.include_router(documents_api.router)
app.include_router(chat_api.router)
app.include_router(files_api.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    """极简“欢迎页”，告诉你服务名与 Swagger 文档地址。"""
    return {
        "service": "school-policy-agent",
        "version": __version__,
        "docs": "/docs",
    }
