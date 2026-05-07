"""政策网页采集并入库工具。

从 HTTP/HTTPS URL 抓取政策网页正文，清洗后切分、向量化并入库。

=============================================================================
设计决策（为什么选择 BeautifulSoup + 规则，而不是 LLM 抽取？）
=============================================================================
1. 成本：网页正文抽取是高频操作，用 LLM 会消耗 token。
2. 准确度：政策网页通常是服务器端渲染的 HTML，结构化程度高，
   BeautifulSoup 的 CSS 选择器能精确命中正文容器。
3. 速度：纯 HTML 解析毫秒级，LLM 需要秒级。

对于 JS 动态渲染的页面（SPA），当前的 httpx + BeautifulSoup 方案会失效。
这是因为 httpx 只能拿到初始 HTML，不会执行 JavaScript。
如果后续需要支持 SPA 页面，可选方案：
- Playwright / Selenium（启动真实浏览器，但资源消耗大）
- 分析 API 请求（很多 SPA 页面背后有 JSON API，直接调 API 比渲染页面更高效）

=============================================================================
正文抽取策略（从精准到宽泛的 3 级降级）
=============================================================================
第 1 级：CSS 选择器精确匹配正文容器
  优先尝试 main, article, .content, #content, .article 等常见容器。
  如果命中且文本 >= 100 字符 → 直接返回。

第 2 级：降级到 <body> 全文
  找不到特定容器 → 取整个 body 的文本。
  在此之前 script/style/nav/footer/header/aside 已被删除，所以 body 文本相对干净。

第 3 级：降级到整个 HTML
  body 也不存在（极少见）→ 取整个 soup 的文本。
=============================================================================
"""

from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.services.ingestion_service import IngestionError, ingest_file
from app.utils.logger import logger

# ---- 模拟浏览器 User-Agent，降低被反爬的概率 ----
# 部分网站会拒绝没有 User-Agent 的请求（返回 403），所以必须设置。
# 使用 Chrome 120 的 UA 字符串，这是最常见的浏览器标识。
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---- 常见正文容器 CSS 选择器 ----
# 按优先级排序：先匹配语义化标签（main/article），再匹配常见 class/id。
# 这些选择器覆盖了绝大多数高校网站和政府网站的正文区域。
# 如果你的目标网站有特殊容器，在这里添加即可。
CONTENT_SELECTORS = [
    # HTML5 语义标签（优先级最高，语义最准确）
    "main",
    "article",
    # 常见 class 选择器
    ".content",
    ".article",
    ".news-content",
    ".wp_articlecontent",   # WordPress 文章正文
    ".post-content",         # WordPress/博客
    ".entry-content",        # WordPress
    ".news_content",
    ".main-content",
    ".detail-content",
    ".text-content",
    # 常见 id 选择器
    "#content",
    "#article",
    "#news_content",
    "#main-content",
    "#detail-content",
    "#text-content",
]


def ingest_policy_from_url(url: str) -> dict:
    """从 URL 采集政策网页，清洗后切分、向量化并入库。

    【完整流程】
    1. URL 合法性校验（协议、域名）
    2. httpx GET 请求（超时 20s，跟随重定向）
    3. BeautifulSoup 清洗 HTML（去掉 script/style/nav/footer/header/aside）
    4. 提取标题（优先 <title>，降级 h1，再降级 URL 本身）
    5. 正文字段抽取（3 级降级：容器 → body → 全文）
    6. 长度检查（< 100 字符则拒绝，可能不是正文页）
    7. 保存为 txt 文件到 uploaded_files/
    8. 调用 ingest_file 完成入库（复用统一入库逻辑）

    参数：
        url: 要采集的网页地址（必须以 http:// 或 https:// 开头）

    返回：
        dict: {document_id, filename, title, source_url, chunk_count, status, message}

    异常：
        ValueError:      URL 协议不合法或缺少域名
        IngestionError:  网页请求失败、内容过短、或入库失败
    """

    # ---- 第 1 步：URL 合法性校验 ----
    # 只允许 http/https，防止 SSRF 攻击（如 file:///etc/passwd）
    if not url or not isinstance(url, str):
        raise ValueError("URL 不能为空")
    url = url.strip()

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"不支持的协议：{parsed.scheme}，仅支持 http 和 https"
        )
    if not parsed.netloc:
        raise ValueError("URL 缺少域名")

    logger.info(f"[url_ingestion] fetching: {url}")

    # ---- 第 2 步：HTTP 请求 ----
    # follow_redirects=True：如果目标地址 301/302 跳转，自动跟随
    # timeout=20：防止某个慢响应页面一直挂着占资源
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
            follow_redirects=True,
        )
        resp.raise_for_status()  # 4xx/5xx → 抛异常
    except httpx.HTTPError as e:
        raise IngestionError(f"网页请求失败：{e}")
    except Exception as e:
        raise IngestionError(f"网页请求异常：{e}")

    # ---- 第 3 步：长度初筛 ----
    # 如果 HTTP 返回的 HTML 本身就很短（< 50 字符），
    # 大概率是空白页、错误页或重定向页，不值得继续处理。
    html = resp.text
    if not html or len(html) < 50:
        raise IngestionError("网页内容过短，可能不是正文页")

    # ---- 第 4 步：BeautifulSoup 解析 ----
    # 使用 lxml 解析器（速度快、容错性好）
    soup = BeautifulSoup(html, "lxml")

    # 4a. 提取标题
    title = _extract_title(soup, url)

    # 4b. 删除噪声标签（这些标签里的文字不是正文内容）
    for tag_name in (
        "script", "style", "noscript", "nav", "footer", "header", "aside"
    ):
        for tag in soup.find_all(tag_name):
            tag.decompose()  # decompose() 比 extract() 更彻底，释放内存

    # ---- 第 5 步：正文抽取 ----
    body_text = _extract_content(soup)

    # ---- 第 6 步：长度终筛 ----
    # 清洗后的正文必须 >= 100 字符，否则很可能是导航页/列表页，不是政策正文
    if not body_text or len(body_text.strip()) < 100:
        raise IngestionError(
            f"网页正文过短（{len(body_text) if body_text else 0} 字符），"
            f"可能不是政策正文页面"
        )

    logger.info(
        f"[url_ingestion] extracted {len(body_text)} chars from {url}"
    )

    # ---- 第 7 步：保存为 txt 文件 ----
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = _url_to_safe_filename(url, timestamp)
    file_path = settings.upload_dir / safe_name
    # encoding="utf-8" 确保中文不会被乱码存储
    file_path.write_text(body_text, encoding="utf-8")
    logger.info(f"[url_ingestion] saved to {file_path}")

    # ---- 第 8 步：复用统一入库逻辑 ----
    # ingest_file 会完成：生成 ID → 解析 → 清洗 → 切分 → 向量化 → BM25 → metadata
    result = ingest_file(
        file_path=file_path,
        filename=safe_name,
        source_url=url,
        title=title,
    )

    return result


def _extract_title(soup: BeautifulSoup, fallback_url: str) -> str:
    """从 HTML 中提取文档标题。

    优先级：
    1. <title> 标签的内容（最可靠，HTML 标准字段）
    2. <h1> 标签的内容（通常就是页面主标题，取前 100 字符防止过长）
    3. URL 本身（最后的兜底）

    为什么不用 LLM 提取？
    <title> 和 <h1> 是 HTML 标准结构，规则提取零成本、零延迟。
    """
    # 优先取 <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    # 降级取第一个 <h1>
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)[:100]

    # 最后的兜底：用 URL 作为标题
    return fallback_url


def _extract_content(soup: BeautifulSoup) -> str:
    """从 BeautifulSoup 树中抽取正文（3 级降级策略）。

    为什么不用 .get_text() 一把梭？
    直接 soup.get_text() 会把侧边栏、导航、页脚等所有文字混在一起。
    第 1 级先尝试找正文容器，能显著提高信噪比。
    """

    # ---- 第 1 级：CSS 选择器匹配正文容器 ----
    # select_one() 返回第一个匹配元素（按 CSS 优先级）
    for selector in CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n")
            # 必须 >= 100 字符才算"真的找到了正文"
            # 否则可能是空容器或只有一句话的导航提示
            if len(text.strip()) >= 100:
                return text

    # ---- 第 2 级：降级到整个 <body> ----
    # 此时 script/style/nav/footer/header/aside 已经被 decompose 了
    # 所以 body.get_text() 的信噪比已经比原始 HTML 好很多
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n")

    # ---- 第 3 级：降级到整个 HTML 文档 ----
    # 极少见的情况（如 HTML 不规范、没有 body 标签）
    return soup.get_text(separator="\n")


def _url_to_safe_filename(url: str, timestamp: str) -> str:
    """将 URL 转换为安全的文件名。

    【为什么需要这个函数？】
    URL 中可能包含 / ? & = % 等文件系统不接受的字符，
    直接用作文件名会报错。本函数将其转换为仅含字母数字下划线的安全名称。

    【命名规则】
    - 从 URL 路径最后一段提取基础名
    - 去掉 .html .php .aspx 等扩展名
    - 去掉不安全字符，保留字母数字中文和下划线
    - 限制 40 字符以内防止文件名过长
    - 格式：url_{基础名}_{时间戳}.txt

    示例：
    "https://example.edu.cn/notice/2024/scholarship.html"
    → "url_scholarship_20260507_153012.txt"
    """
    parsed = urlparse(url)
    path_part = parsed.path.strip("/")

    if path_part:
        # 取 URL 路径最后一段
        segments = path_part.split("/")
        name_base = segments[-1]

        # 去掉扩展名（.html .htm .php .aspx .jsp 等）
        name_base = re.sub(r"\.\w+$", "", name_base)

        # 去掉不安全字符，保留：字母、数字、中文、下划线
        # [^\w一-鿿-]：\w = [a-zA-Z0-9_]，一-鿿 = CJK 统一汉字区间
        name_base = re.sub(r"[^\w一-鿿-]", "_", name_base)

        # 文件名长度限制，防止 Windows 路径过长
        if len(name_base) > 40:
            name_base = name_base[:40]
    else:
        # URL 没有路径（如 https://example.com），用默认名
        name_base = "policy_page"

    return f"url_{name_base}_{timestamp}.txt"
