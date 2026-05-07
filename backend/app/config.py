"""集中管理所有运行期配置（运行时“一本通”）。

关键点（小白建议按这个顺序理解）：
1. `pydantic-settings` 会从环境变量 + `.env 文件` 里读取配置，并做类型校验。
2. `Field(..., alias="LLM_BASE_URL")` 表示：Python 里用小写蛇形名 `llm_base_url`，
   但 `.env` / 系统环境变量里写大写 `LLM_BASE_URL` 也能自动映射进来。
3. 路径常量 `BACKEND_DIR` 等用 `Path(__file__)` 相对定位，避免写死绝对路径。

不要把 API Key、内网地址写死在代码里——全部走 `.env`，方便换机器、换模型、换向量库。
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 下面这些 Path 是“相对本文件”算出来的，便于项目挪目录也不炸：
# config.py 位于 backend/app/config.py → parent.parent 就是 backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
STORAGE_DIR = BACKEND_DIR / "app" / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploaded_files"
METADATA_PATH = STORAGE_DIR / "metadata.json"


class Settings(BaseSettings):
    """全局应用配置。

    小技巧：改完 `.env` 后需要**重启 uvicorn** 才会生效（本文件用 `lru_cache` 单例缓存了配置）。
    """

    # ---- LLM ----
    llm_base_url: str = Field("https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_model: str = Field("deepseek-chat", alias="LLM_MODEL")

    # ---- Embedding ----
    embedding_model: str = Field("BAAI/bge-m3", alias="EMBEDDING_MODEL")

    # ---- Qdrant ----
    qdrant_host: str = Field("localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field("school_policy_docs", alias="QDRANT_COLLECTION")

    # ---- 切分 ----
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")

    # ---- 检索 ----
    top_k: int = Field(6, alias="TOP_K")

    # ---- 拒答阈值 ----
    score_threshold_high: float = Field(0.75, alias="SCORE_THRESHOLD_HIGH")
    score_threshold_medium: float = Field(0.45, alias="SCORE_THRESHOLD_MEDIUM")

    # 项目目录
    backend_dir: Path = BACKEND_DIR
    project_root: Path = PROJECT_ROOT
    storage_dir: Path = STORAGE_DIR
    upload_dir: Path = UPLOAD_DIR
    metadata_path: Path = METADATA_PATH

    model_config = SettingsConfigDict(
        env_file=[
            str(PROJECT_ROOT / ".env"),  # 推荐：与 docker-compose.yml 同级的根目录 .env
            str(BACKEND_DIR / ".env"),  # 备选：仅在 backend 目录放 .env（适合单目录部署）
        ],
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例：`get_settings()` 第一次调用时读盘，之后全程复用同一个对象。

    同时确保上传目录存在，避免第一次写文件时报 `FileNotFoundError`。
    """
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
