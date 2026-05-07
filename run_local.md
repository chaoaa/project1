# 本地运行指南（小白逐步版）

> 目标：在你自己的 Windows / macOS / Linux 电脑上把项目完整跑起来。
> 总共 4 个终端：1 个 Docker、1 个后端 uvicorn、1 个前端 streamlit、1 个备用（用来 curl 测试可省）。

---

## 0. 前置依赖

- **Python 3.10+**：在终端输入 `python --version` 检查；没有就去 https://www.python.org/downloads/ 安装。
- **Docker Desktop**：https://www.docker.com/products/docker-desktop/ ，安装完启动 Docker。
- 一个可用的大模型 API Key（推荐 DeepSeek，1 元能用很久）：https://platform.deepseek.com/

> Windows 用户注意：以下命令在 PowerShell 中执行；如出错请尝试用 `cmd`。

---

## 1. 创建虚拟环境（后端）

```bash
cd backend
python -m venv .venv
```

**激活虚拟环境：**

- Windows PowerShell：`.\.venv\Scripts\Activate.ps1`
- Windows CMD：`.venv\Scripts\activate.bat`
- macOS / Linux：`source .venv/bin/activate`

激活成功后，命令行最前面会出现 `(.venv)`。

---

## 2. 安装后端依赖

```bash
pip install -r requirements.txt
```

> 第一次安装较慢，因为 `sentence-transformers`、`torch` 体积较大，请耐心等待。
> 国内网络可加镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

## 3. 安装前端依赖（新开一个终端）

```bash
cd frontend
python -m venv .venv
# 激活（参考第 1 步）
pip install -r requirements.txt
```

---

## 4. 配置 .env

在**项目根目录**（也就是 `agent/`，不是 `backend/`）执行：

- Windows：

  ```
  copy .env.example .env
  notepad .env
  ```

- macOS / Linux：

  ```
  cp .env.example .env
  nano .env
  ```

**至少修改：**

```
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=替换为你的真实Key
LLM_MODEL=deepseek-chat
```

> 如果你的网络无法下载 `BAAI/bge-m3`（首次会自动下载约 2GB），就把 `EMBEDDING_MODEL` 改成轻量模型：
>
> ```
> EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
> ```

---

## 5. 启动 Qdrant（向量数据库）

在项目根目录新开一个终端：

```bash
docker compose up -d qdrant
```

验证：浏览器访问 http://localhost:6333/dashboard
看到 Qdrant 的 Web 控制台就 OK。

---

## 6. 启动后端

回到第 1 步的后端终端（已激活 venv）：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动成功后访问：

- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/health

预期 health 返回：

```json
{
  "status": "ok",
  "qdrant_connected": true,
  "document_count": 0,
  "chunk_count": 0
}
```

---

## 7. 启动前端

回到第 3 步的前端终端（已激活 venv）：

```bash
streamlit run streamlit_app.py
```

浏览器自动打开 http://localhost:8501。

---

## 8. 上传示例政策文件

在 Streamlit 左侧：

1. 点 **"上传政策文件"**，选择 `examples/sample_policy_1.txt`，点 **"上传并入库"**
2. 重复上传 `examples/sample_policy_2.md` 与 `examples/sample_policy_old_version.txt`
3. 左下方 **"已入库文档"** 应当显示 3 个文档，每个 status=indexed

> 第一次上传会触发 Embedding 模型下载，请耐心等待 1~5 分钟。

---

## 9. 提问测试

主区域选择 **"Agent 问答（推荐）"** 模式，依次试这些：

**普通政策问答：**

- 研究生学业奖学金申请条件是什么？
- 申请奖学金需要提交哪些材料？
- 研究生毕业需要满足哪些条件？

**资格判断（展开"Agent 高级参数" → 用户档案）：**

```json
{"年级":"研二","成绩排名":"前20%","挂科情况":"有1门不及格"}
```

问：我能申请奖学金吗？

**材料清单生成：**

- 帮我生成毕业申请材料清单。

**版本对比（高级参数中选择 旧版/新版）：**

- 旧版：sample_policy_old_version.txt
- 新版：sample_policy_1.txt
- 问：对比新旧版奖学金政策有什么变化？

**拒答测试：**

- 学校食堂几点营业？（应当返回拒答 + low 置信度）

---

## 10. 常见报错与解决

### 报错：`Connection refused (qdrant)`
没启动 Qdrant。回到第 5 步执行 `docker compose up -d qdrant`。

### 报错：`LLM_API_KEY 未配置`
`.env` 中没填 Key 或 backend 启动时没读到 `.env`。
- 确认 `.env` 在 **项目根目录**（与 `docker-compose.yml` 同级）
- 重启 `uvicorn`

### 报错：`OSError: model not found / We couldn't connect to huggingface.co`
网络无法下载 Embedding 模型。两种解决：
- 设置代理：`set HF_ENDPOINT=https://hf-mirror.com` 后重启后端（Linux/Mac 用 `export`）
- 改用小模型：`EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`

### 报错：`response_format json_object is not supported`
某些模型不支持 JSON 模式。代码已自动 fallback 到普通 chat + 正则提取 JSON，无需处理。

### 上传 PDF 后 chunk_count=0
扫描版 PDF。请改用文本版 PDF，或后续接入 OCR（`backend/app/services/document_loader.py` 有 TODO）。

### Windows 上 `Activate.ps1` 报"无法运行脚本"
管理员 PowerShell 执行：

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 端口已占用
- 后端：把 `--port 8000` 改为别的，例如 `--port 8010`，前端侧栏后端地址同步改
- 前端：`streamlit run streamlit_app.py --server.port 8502`
- Qdrant：改 `docker-compose.yml` 中 `6333:6333` 为 `16333:6333` 等

---

## 11. 关闭服务

- Streamlit：终端按 `Ctrl + C`
- 后端：终端按 `Ctrl + C`
- Qdrant：`docker compose down`

## 12. 新功能测试（v2 增强版 Agent）

### 12.1 规则化奖学金资格判断测试

此功能将资格判断从"纯 LLM 猜测"改为"确定性规则判断"，LLM 只负责解释和表达。

**测试问题：**

```
我是研二学生，上一学年成绩排名前20%，但有一门课程不及格，能申请研究生学业奖学金吗？
```

**预期结果：**
- Agent 意图识别为 `eligibility_check`
- 调用工具：`rule_based_scholarship_checker`
- 结论：**不符合**（not_eligible）
- 原因：有 1 门课程不及格
- 输出包含：【资格判断结论】【识别到的用户信息】【判断原因】【政策依据】【注意】

**更多测试用例：**

| 问题 | 预期结论 |
|------|----------|
| 我是研二，排名前15%，无挂科无处分，能申请吗？ | eligible（一等） |
| 我是研二，排名前35%，没有挂科，能申请吗？ | eligible（二等） |
| 我是研二，排名前60%，没有挂科，能申请吗？ | not_eligible |
| 我是研二，排名未知，没有挂科，能申请吗？ | need_more_info |
| 我是研二，排名前20%，但有处分，能申请吗？ | not_eligible |
| 我是研二，有学术不端记录，能申请吗？ | not_eligible |

也可以通过"Agent 高级参数 → 用户档案"传入 JSON：

```json
{"年级":"研二","成绩排名":"前20%","挂科情况":"有1门不及格"}
```

### 12.2 材料清单 Word 导出测试

此功能在生成材料清单的同时，自动生成 Word 文件并提供下载链接。

**测试问题：**

```
帮我生成研究生毕业申请材料清单和办理步骤。
```

**预期结果：**
- Agent 意图识别为 `checklist_generation`
- 调用工具：`generate_checklist`
- 回答底部显示："已生成 Word 材料清单，可在前端下载。"
- Tool Result 展开后包含 `generated_file` 字段（filename + download_url）
- 前端显示下载链接，点击可下载 `.docx` 文件
- Word 文件保存在 `backend/app/storage/generated_files/` 目录

**验证 Word 文件：**

1. 浏览器打开 http://localhost:8000/api/files/download/checklist_xxx.docx
2. 文件应包含：标题、所需材料、办理步骤、注意事项、生成时间

### 12.3 网页政策采集测试

此功能让 Agent 通过 URL 抓取政策网页正文，清洗后入库。

**测试问题：**

```
帮我把这个网页加入知识库：https://example.com/test-policy.html
```

（建议使用真实的政策网页 URL 测试，例如学校官网的通知页面）

**预期结果：**
- Agent 意图识别为 `web_ingestion`
- 调用工具：`ingest_policy_from_url`
- 返回：已成功采集并入库网页政策（标题、文件名、chunk 数量、来源 URL）
- 左侧"已入库文档"中新增一条记录

**也可直接调用 API：**

```bash
curl -X POST http://localhost:8000/api/documents/ingest-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.edu.cn/notice/xxx.html"}'
```

返回示例：

```json
{
  "document_id": "doc_xxx",
  "filename": "url_xxx.txt",
  "title": "关于...的通知",
  "source_url": "https://...",
  "chunk_count": 8,
  "status": "indexed",
  "message": "文档已成功入库"
}
```

### 12.4 新增 API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/documents/ingest-url | URL 政策网页采集入库 |
| GET  | /api/files/download/{filename} | 下载生成的 Word 材料清单 |

---

数据持久化目录：

- 上传文件：`backend/app/storage/uploaded_files/`
- 生成文件：`backend/app/storage/generated_files/`
- 元数据：`backend/app/storage/metadata.json`
- 向量数据：`qdrant_storage/`（Docker 数据卷，可删除以重置）
