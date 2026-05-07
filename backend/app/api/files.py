"""文件下载接口：`/api/files/*`。

提供生成文件（如 Word 材料清单）的下载服务。

=============================================================================
安全设计（为什么要有这么多重检查？）
=============================================================================
文件下载接口是 Web 应用中最容易出安全问题的端点之一。

常见攻击方式：路径穿越（Path Traversal）
攻击者尝试 GET /api/files/download/../../../etc/passwd
如果只做简单的文件拼接（GENERATED_DIR + filename），就能读到服务器上任意文件。

本接口的防护策略（纵深防御）：
1. 黑名单字符过滤：拒绝包含 "/"、"\\"、".." 的文件名
   （很多人只检查 "../"，但 Windows 上 "..\\" 也能穿越）
2. resolve() + startswith 校验：即使第 1 步被绕过，
   最终拼接出的绝对路径也必须以 GENERATED_DIR 开头
3. 文件存在检查：不存在的文件直接 404，不给攻击者探测信息

这三层检查合在一起，能有效防御绝大多数路径穿越攻击。
=============================================================================
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.utils.logger import logger

router = APIRouter(prefix="/api/files", tags=["files"])

# 生成文件的存储目录
# 与 uploaded_files (用户上传的) 分开，职责清晰
GENERATED_DIR = settings.storage_dir / "generated_files"


@router.get("/download/{filename}")
def download_file(filename: str):
    """下载生成的文件（如 Word 材料清单）。

    FastAPI 会自动将 URL 路径中的 {filename} 注入为函数参数。
    例如：GET /api/files/download/checklist_20260507_153012.docx
    → filename = "checklist_20260507_153012.docx"

    返回值：
        FileResponse: FastAPI 会设置正确的 Content-Type 和 Content-Disposition
        浏览器收到后会自动触发下载。
    """

    # ---- 第 1 层防护：黑名单字符过滤 ----
    # 拒绝包含路径穿越关键字符的文件名
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="filename 不能为空")

    for bad in ("/", "\\", ".."):
        if bad in filename:
            raise HTTPException(
                status_code=400,
                detail="非法文件名"
            )

    # ---- 第 2 层防护：路径 resolve + 前缀检查 ----
    # Path.resolve() 会把 "../" 等相对路径符号全部解析为绝对路径
    # 然后检查解析后的路径是否确实在 GENERATED_DIR 内
    file_path = Path(GENERATED_DIR / filename).resolve()
    resolved_base = GENERATED_DIR.resolve()

    if not str(file_path).startswith(str(resolved_base)):
        logger.warning(
            f"[files] blocked path traversal attempt: {filename}"
        )
        raise HTTPException(status_code=400, detail="非法文件路径")

    # ---- 第 3 层防护：文件存在检查 ----
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    logger.info(f"[files] downloading: {file_path}")

    # ---- 返回文件 ----
    # media_type 设置正确的 MIME 类型，确保浏览器识别为 docx 文件
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=(
            "application/"
            "vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
    )
