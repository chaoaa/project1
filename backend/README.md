# 后端：School Policy Agent Backend

## 概览

- 框架：FastAPI + LangGraph + Qdrant + sentence-transformers + rank-bm25
- 入口：`app/main.py`
- 默认端口：8000
- API 文档：启动后访问 `http://localhost:8000/docs`

## 启动

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# 配置环境变量（在项目根目录复制 .env.example 为 .env）
copy ..\.env.example ..\.env       # Windows
# cp ../.env.example ../.env       # macOS/Linux
# 然后修改 .env 中的 LLM_API_KEY

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | /api/health           | 健康检查 |
| POST | /api/documents/upload | 上传文档（multipart/form-data, field=`file`） |
| GET  | /api/documents        | 已入库文档列表 |
| POST | /api/chat/query       | 普通 RAG 问答 |
| POST | /api/chat/agent       | Agent 问答（自动识别意图 + 调用工具） |
| POST | /api/chat/retrieve    | 仅检索（调试用） |

## 目录结构

```
backend/app
├── api/             # FastAPI 路由
├── services/        # 解析 / 清洗 / 切分 / 检索 / LLM / 引用
├── agent/           # LangGraph 节点、工具、prompt、状态
├── storage/         # 上传文件 + metadata.json
└── utils/           # logger / 文件工具
```
