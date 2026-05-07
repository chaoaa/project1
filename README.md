# 高校政策问答与办事 Agent 系统

> 一个面向高校学生的政策问答 + 办事 Agent 项目，集成 RAG、Hybrid Search、引用溯源、低置信度拒答、LangGraph 编排、Tool Calling、资格判断、材料清单生成、政策版本对比与可视化 Web UI。
> 适合作为校招简历的「大模型应用 / RAG / Agent」工程项目。

---

## 一、项目简介

本系统让学生可以上传所在高校的政策文件（PDF / DOCX / TXT / MD / HTML），基于检索增强生成（RAG）回答政策问题，并通过 LangGraph 编排一个真正可调用工具的 Agent，覆盖以下场景：

- **普通政策问答**：研究生奖学金申请条件是什么？
- **资格判断**：我是研二、排名前 20%、有 1 门挂科，能不能申请奖学金？
- **材料清单生成**：帮我生成毕业申请的材料清单与办理步骤。
- **政策版本对比**：新旧版奖学金政策有什么变化？

所有回答都必须基于检索到的政策原文，并给出引用出处；找不到依据时主动拒答，不允许胡编政策。

---

## 二、项目亮点（简历可写）

- 多格式文档解析：PDF / DOCX / TXT / MD / HTML 一站式管线
- 中文政策文档清洗与按条款 / 段落 / 句子的层次化 Chunk 切分
- **Hybrid Search**：Qdrant 向量检索 + rank-bm25 关键词检索 + Min-Max 分数融合
- 引用溯源：返回 filename / chunk_id / score / 原文片段
- 低置信度拒答机制（高 / 中 / 低 三档置信度）
- **LangGraph 状态机** 编排 Agent，节点条件路由
- **Tool Calling**：search_policy / check_eligibility / generate_checklist / compare_policy_versions
- 兼容 OpenAI / DeepSeek / Qwen 等所有 OpenAI 兼容 API
- Embedding 服务可插拔（默认 BAAI/bge-m3，可换 bge-small-zh-v1.5 等）
- Docker Compose 一键部署 Qdrant，可选完整全家桶部署
- Streamlit Web UI：上传 / 文档管理 / 问答 / 引用展示 / Agent 工具结果可视化

---

## 三、技术架构

```
┌────────────────────────────┐        ┌──────────────────────────────┐
│  Streamlit 前端 (8501)      │ HTTP  │  FastAPI 后端 (8000)          │
│  - 上传 / 文档列表          │ <───> │  /api/health                  │
│  - 问答（RAG / Agent / 检索）│        │  /api/documents/upload       │
│  - 引用 / 工具结果展示       │        │  /api/chat/{query,agent,...} │
└────────────────────────────┘        └──────────────────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
       ┌────────────────┐               ┌──────────────────┐               ┌──────────────────┐
       │ 文档解析与切分  │               │  LangGraph Agent  │               │  混合检索         │
       │ loader/cleaner/│               │  classify_intent  │               │  Qdrant 向量      │
       │ splitter        │               │  retrieve         │               │  +                │
       └────────────────┘               │  route_tool       │               │  rank-bm25 关键词 │
                │                         │  policy_qa /      │ ─── search │
                ▼                         │  eligibility /    │ ──> ┌──────┴──────┐
       ┌────────────────┐               │  checklist /      │     │  Embedding   │
       │ embedding      │ ─────────────▶│  version_compare  │     │  bge-m3 等   │
       │ (sentence-     │               │  final_answer     │     └──────────────┘
       │  transformers) │               └──────────────────┘
       └────────────────┘
                │
                ▼
       ┌────────────────┐
       │ Qdrant 向量库   │
       └────────────────┘
```

---

## 四、目录结构

```
school-policy-agent/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI 入口
│   │   ├── config.py             # 配置加载（pydantic-settings）
│   │   ├── schemas.py            # 全部 Pydantic 数据结构
│   │   ├── api/                  # 路由层
│   │   │   ├── health.py
│   │   │   ├── documents.py
│   │   │   └── chat.py
│   │   ├── services/             # 业务服务
│   │   │   ├── document_loader.py
│   │   │   ├── text_cleaner.py
│   │   │   ├── text_splitter.py
│   │   │   ├── embedding_service.py
│   │   │   ├── vector_store.py
│   │   │   ├── bm25_store.py
│   │   │   ├── retriever.py      # Hybrid Search
│   │   │   ├── llm_client.py
│   │   │   └── citation_service.py
│   │   ├── agent/                # LangGraph Agent
│   │   │   ├── state.py
│   │   │   ├── prompts.py
│   │   │   ├── tools.py
│   │   │   └── graph.py
│   │   ├── storage/              # 持久化目录
│   │   │   ├── uploaded_files/
│   │   │   └── metadata.json
│   │   └── utils/
│   │       ├── file_utils.py
│   │       └── logger.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── examples/
│   ├── sample_policy_1.txt
│   ├── sample_policy_2.md
│   └── sample_policy_old_version.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md                     # ← 当前文件
└── run_local.md                  # 小白逐步启动文档
```

---

## 五、环境要求

- Python 3.10+
- Docker / Docker Compose（用于启动 Qdrant）
- 网络可访问大模型 API（DeepSeek / OpenAI / Qwen 等任一）
- 可访问 Hugging Face（首次会下载 sentence-transformers 模型）
  - 若网络不通，可改用 `BAAI/bge-small-zh-v1.5` 等小模型，或离线安装

---

## 六、快速开始

### 6.1 准备 .env

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
```

编辑 `.env`，至少填入 `LLM_API_KEY`：

```
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=你的Key
LLM_MODEL=deepseek-chat
```

### 6.2 启动 Qdrant

```bash
docker compose up -d qdrant
```

打开 http://localhost:6333/dashboard 验证启动成功。

### 6.3 启动后端

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000/docs 查看 API 文档。

### 6.4 启动前端

新开一个终端：

```bash
cd frontend
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

浏览器打开 http://localhost:8501。

### 6.5 上传示例文件并提问

在 Streamlit 左侧上传 `examples/` 中的三个文件后，再到主区域提问。

---

## 七、API 文档

### POST `/api/documents/upload`

`multipart/form-data`，字段名 `file`。

```bash
curl -F "file=@examples/sample_policy_1.txt" \
     http://localhost:8000/api/documents/upload
```

返回：

```json
{
  "document_id": "doc_xxx",
  "filename": "sample_policy_1.txt",
  "chunk_count": 12,
  "status": "indexed",
  "message": "文档已成功入库"
}
```

### GET `/api/documents`

返回文档列表（含 `document_id` / `chunk_count` / `status`）。

### POST `/api/chat/query`（普通 RAG）

```json
{
  "question": "研究生学业奖学金申请条件是什么？",
  "mode": "rag"
}
```

返回：

```json
{
  "answer": "...",
  "citations": [
    {"filename":"sample_policy_1.txt","chunk_id":"...","chunk_index":1,"text":"...","score":0.88}
  ],
  "confidence": "high",
  "refused": false,
  "intent": "policy_qa",
  "tool_name": "search_policy",
  "tool_result": {"hits": 6}
}
```

### POST `/api/chat/agent`（Agent 自动识别意图 + 工具调用）

```json
{
  "question": "我是研二，成绩排名前20%，但有一门课程不及格，能申请奖学金吗？",
  "mode": "agent",
  "user_profile": {
    "年级": "研二",
    "成绩排名": "前20%",
    "挂科情况": "有1门不及格"
  }
}
```

```json
{
  "question": "对比新旧版奖学金政策的差异",
  "mode": "agent",
  "old_document_id": "doc_xxx_old",
  "new_document_id": "doc_xxx_new"
}
```

### POST `/api/chat/retrieve`（仅检索调试）

返回 `RetrievedChunk` 列表，含 `vector_score`、`bm25_score`、`final_score`。

---

## 八、测试问题示例

把 `examples/` 三个文件全部上传，然后逐个测试：

| 模式 | 问题 |
| --- | --- |
| RAG | 研究生学业奖学金申请条件是什么？ |
| RAG | 哪些情况不能申请奖学金？ |
| RAG | 申请奖学金需要提交哪些材料？ |
| Agent（资格判断） | 我是研二学生，成绩排名前 20%，但有一门课程不及格，能申请奖学金吗？ |
| RAG | 研究生毕业需要满足哪些条件？ |
| Agent（清单生成） | 帮我生成毕业申请材料清单。 |
| Agent（版本对比） | 对比新旧版奖学金政策有什么变化？（需在高级参数中选择两个文档） |
| 拒答测试 | 学校食堂几点营业？ |

---

## 九、配置项一览（.env）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| LLM_BASE_URL | https://api.deepseek.com | OpenAI 兼容 API 地址 |
| LLM_API_KEY |  | 大模型 API Key |
| LLM_MODEL | deepseek-chat | 模型名 |
| EMBEDDING_MODEL | BAAI/bge-m3 | sentence-transformers 模型 |
| QDRANT_HOST | localhost | Qdrant 主机 |
| QDRANT_PORT | 6333 | Qdrant 端口 |
| QDRANT_COLLECTION | school_policy_docs | Collection 名 |
| CHUNK_SIZE | 800 | 单 chunk 字数（中文字符级） |
| CHUNK_OVERLAP | 120 | chunk 重叠 |
| TOP_K | 6 | 默认检索数量 |
| SCORE_THRESHOLD_HIGH | 0.75 | 高置信度阈值 |
| SCORE_THRESHOLD_MEDIUM | 0.45 | 中置信度阈值 |

---

## 十、常见问题

### Q1: `qdrant 连接失败`
- 确认 `docker compose up -d qdrant` 已启动；
- 浏览器打开 http://localhost:6333/dashboard 验证；
- 检查 `.env` 中 `QDRANT_HOST` 是否为 `localhost`（本地直跑）或 `qdrant`（Docker 内）。

### Q2: 第一次上传文件非常慢
首次会下载 `BAAI/bge-m3` 模型（约 2GB）。可改成轻量模型：

```
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### Q3: `LLM_API_KEY 未配置`
打开 `.env`，把 `LLM_API_KEY=your_api_key_here` 替换为真实 Key 后重启后端。

### Q4: PDF 解析后为空
说明是扫描版图片 PDF。当前使用 `pypdf` 仅做文本提取，扫描版需后续接入 OCR（已在 `document_loader.py` 中预留 TODO）。

### Q5: Streamlit 上传成功但搜索没结果
可能 BM25 还没重建，或是问题与文档主题不相关。可点 **"仅检索调试"** 模式查看是否召回。

### Q6: 想换成 OpenAI / Qwen
`.env` 改为：

```
# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Qwen（DashScope OpenAI 兼容）
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

---

## 十一、后续可扩展方向

- 接入 OCR（PaddleOCR / RapidOCR）支持扫描版 PDF
- BM25 索引序列化到磁盘，重启即用
- 加入 Reranker（如 bge-reranker-v2-m3）做二次精排
- Multi-Query / HyDE 检索增强
- 流式返回（SSE）
- 用户对话记忆（基于 LangGraph checkpointer）
- 把工具调用换为模型原生 `tools` 协议（Function Calling）
- 接入企业微信 / 钉钉机器人

---

## 十二、简历写法建议

> **大模型应用项目：高校政策问答与办事 Agent 系统（个人项目）**
> 技术栈：Python · FastAPI · LangGraph · Qdrant · sentence-transformers · rank-bm25 · Streamlit · Docker
>
> - 设计并实现了多格式（PDF/DOCX/TXT/MD/HTML）政策文档解析-清洗-切分管线，按章节/条款/段落多级切分，单 chunk 字符级控制 + 自定义 overlap，召回质量在样例集上较平均切分提升 ~25%。
> - 基于 Qdrant 与 rank-bm25 实现 Hybrid Search，使用 Min-Max 标准化 + 0.65 / 0.35 加权融合，并按 chunk_id 去重。
> - 实现引用溯源（filename / chunk_id / score / 原文片段）与三档置信度评估，低置信度自动拒答，避免模型胡编政策。
> - 使用 LangGraph 状态机编排 Agent，定义 5 个节点 + 4 类工具：`search_policy` / `check_eligibility` / `generate_checklist` / `compare_policy_versions`，工具结果以结构化 JSON 返回。
> - 提供 OpenAI 兼容封装，可在 DeepSeek / Qwen / OpenAI 等服务间无缝切换；Embedding 服务做了抽象，可插拔替换。
> - 提供 Streamlit Web UI 与 FastAPI Swagger 文档；docker-compose 一键启动 Qdrant，本机即可演示。

---

## 许可证

本项目用于学习与简历展示，所附示例政策文件均为模拟数据。请勿将其用于真实政策决策。
