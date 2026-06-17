# AI Agent Chat — 万事通智能助手

> 基于 FastAPI + LangChain + DeepSeek 的知识库问答系统，支持 RAG 检索、工具调用、记忆持久化、SSE 流式对话。

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)]()
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [工具清单](#工具清单-20-个)
- [RAG 检索架构](#rag-检索架构)
- [数据存储](#数据存储)
- [记忆体系](#记忆体系)
- [安全设计](#安全设计)
- [超时与降级](#超时与降级)
- [API 接口](#api-接口)
- [环境变量](#环境变量)
- [测试](#测试)

---

## 快速开始

### 前置依赖

- Python 3.12+
- MySQL 8.0+（运行中）
- Qdrant 向量数据库（可选，自动降级）
- DeepSeek / DashScope API Key

### 启动服务

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
# 编辑 .env，填入 API Key：
#   OPENAI_API_KEY=sk-xxx       (DeepSeek)
#   DASHSCOPE_API_KEY=sk-xxx    (DashScope 文本嵌入)

# 3. 启动服务（带热重载）
uvicorn src.server.api_server:app --host 0.0.0.0 --port 8000 --reload

# 4. 打开浏览器
open http://localhost:8000
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                    前端（单页 HTML）                           │
│     chat.html — SSE 流式对话 / 知识库管理 / Markdown 渲染    │
├──────────────────────────────────────────────────────────────┤
│                  FastAPI 服务层（api_server.py）               │
│        12 条 REST API + 1 条 SSE 端点 + Pydantic 校验        │
├──────────────┬───────────────────────────────┬───────────────┤
│  Agent 层    │  工具层                      │  数据层        │
│ agent_chain  │  tool_registry (单例)        │  MySQL         │
│  ┌───────┐   │  ├── calculator             │  ├─ chunks     │
│  │ Prompt│   │  ├── web_search/fetch       │  ├─ terms      │
│  │ Tools │   │  ├── rag_retrieve           │  ├─ KB 元数据  │
│  │Memory │   │  ├── file_tools (4)         │  ├─ 对话历史   │
│  │滑动窗口│   │  ├── shell_tools           │  ├─ 情景记忆   │
│  └───────┘   │  ├── system_tools (2)       │  ├─ 语义记忆   │
│              │  ├── app_tools (2)          │  Qdrant        │
│  安全层      │  ├── episodic_memory (3)    │  ├─ knowledge  │
│  ├─白名单    │  └── semantic_memory (4)    │  └─ memory     │
│  ├─黑名单    │                               │               │
│  ├─超时兜底  │   外部依赖                    │  DeepSeek API │
│  └─审计日志  │   ├─ DuckDuckGo (搜索)       │  DashScope API│
│              │   └─ psutil (系统信息)       │               │
└──────────────┴───────────────────────────────┴───────────────┘
```

### 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Agent 框架 | 自实现 ReAct，非 LangGraph | 当前场景线性对话，不需要多智能体编排；自实现依赖更少、控制更精细 |
| 检索融合 | RRF（Reciprocal Rank Fusion） | 消除 Qdrant cosine 与 BM25 分数不可比的问题，排名比分数更稳定 |
| 记忆 | 三层分离（对话/情景/语义） | 三种信息的查询方式不同，混在一层会互相干扰 |
| 流式 | SSE 非 WebSocket | 单向推送够了，不需要双向通信，HTTP 协议对 Nginx 代理友好 |
| LLM 超时 | 三层嵌套：30s → 60s → 120s | 内层超时先触发，外层兜底，防止单点卡死整个请求 |

---

## 功能特性

### 核心能力

- **RAG 三路检索** — Qdrant 语义向量 + BM25 稀疏倒排 + MySQL LIKE 关键词，RRF 融合
- **20 个可调用工具** — 计算、搜索、文件操作、Shell 命令、系统信息、应用控制
- **三层记忆体系** — 对话历史持久化 + 情景记忆（关键词）+ 语义记忆（向量）
- **SSE 流式对话** — 逐 token 推送，打字机效果，来源引用卡片
- **多格式文档解析** — PDF / DOCX / TXT / MD / CSV 自动转换、分块、入库
- **滑动窗口上下文** — 保留最近 N 轮对话，控制 LLM 上下文长度恒定
- **优雅降级** — Qdrant / BM25 / MySQL 任一不可用时，自动降级不影响主服务
- **审计日志** — 所有工具调用记录 + 敏感参数自动脱敏

### 安全保障

- 命令白名单 + 黑名单 + python/pip 特殊处理
- 文件路径 `os.path.realpath` 防穿越 + 白名单目录限写
- 读取文件 100KB 上限
- 三层超时兜底（LLM 30s → 迭代 60s → SSE 120s）
- API Key 启动校验

---

## 项目结构

```
├── src/
│   ├── server/
│   │   ├── api_server.py          # FastAPI 服务（16 条路由 + SSE）
│   │   └── chat_store.py          # 对话历史 MySQL 存储
│   │
│   ├── agent/
│   │   └── agent_chain.py         # Agent ReAct 循环 + Session 管理
│   │
│   ├── tools/
│   │   ├── tool_registry.py       # 工具注册中心（单例，20 个工具）
│   │   ├── calculator_tool.py     # 数学计算
│   │   ├── web_search_tool.py     # 联网搜索（DuckDuckGo）+ 网页抓取
│   │   ├── rag_retriever_tool.py  # RAG 三路检索 + RRF 融合 + 重排
│   │   ├── file_tools.py          # 读/写/搜索/列目录（路径白名单）
│   │   ├── shell_tools.py         # Shell 命令（三层安全防护）
│   │   ├── app_tools.py           # 打开/列出应用（预定义映射表）
│   │   ├── system_tools.py        # 系统信息 + 进程列表（psutil）
│   │   └── memory/
│   │       ├── episodic_memory_tool.py   # 情景记忆（MySQL）
│   │       └── semantic_memory_tool.py   # 语义记忆（MySQL+Qdrant 双写）
│   │
│   ├── knowledge_ingest/
│   │   ├── document_processor.py  # 文档分块（500 字，overlap 50）
│   │   ├── file_converter.py      # 格式转换（PDF/DOCX/TXT/MD/CSV）
│   │   ├── chunk_store.py         # MySQL document_chunks 存储
│   │   ├── vector_store.py        # Qdrant 向量存储（DashScope 嵌入）
│   │   ├── bm25_index.py          # BM25 稀疏检索引擎（内存缓存）
│   │   ├── bm25_indexer.py        # 倒排索引构建器（jieba 分词）
│   │   └── inverted_index_schema.py # 倒排索引表 DDL
│   │
│   └── utils/
│       ├── config_loader.py       # 集中配置 + 路径安全校验
│       ├── logger.py              # 错误日志（RotatingFileHandler）
│       └── audit.py               # 审计日志 + 降级事件记录
│
├── static/
│   └── chat.html                  # 单页聊天 UI（暗色主题 + Markdown 渲染）
│
├── tests/
│   └── regression_test.py         # 全量回归测试（79 项）
│
├── .env                           # 环境变量（不含 Git）
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

---

## 工具清单（20 个）

| 工具 | 功能 | 权限 | 安全措施 |
|------|------|------|---------|
| `calculator` | 加/减/乘/除/幂/开方/对数/三角/阶乘 | auto | 纯计算，无副作用 |
| `web_search` | DuckDuckGo 联网搜索 | auto | 只读，HTTP GET |
| `web_fetch` | 抓取指定 URL 的文本内容 | auto | 只读，限 3000 字符 |
| `rag_retrieve` | RAG 三路检索 + RRF + 重排 | auto | 只读知识库 |
| `read_file` | 读取文件（默认 100KB 上限） | auto | 路径 `os.path.realpath` 防穿越 |
| `write_file` | 写入/追加文件 | user_confirm | 限白名单目录（./data, ./output, ./static） |
| `search_files` | glob 模式搜索文件名 | auto | 限项目目录 |
| `list_directory` | 列出目录内容（非递归） | auto | 只读 |
| `run_command` | 执行 Shell 命令 | user_confirm | 白名单 + 黑名单 + python/pip 特殊处理 |
| `open_application` | 打开记事本/计算器/Chrome 等 | user_confirm | 预定义应用映射表 |
| `list_applications` | 列出可打开的应用 | auto | 只读 |
| `get_system_info` | CPU/内存/磁盘/OS 信息 | auto | 只读 |
| `get_process_list` | 进程列表（按 CPU 排序） | auto | 只读 |
| `episodic_memory_save` | 保存情景记忆 | auto | MySQL 写入 |
| `episodic_memory_search` | 搜索情景记忆 | auto | MySQL 查询 |
| `episodic_memory_delete` | 删除情景记忆 | auto | 需先查得 ID |
| `semantic_memory_save` | 保存语义记忆（MySQL+Qdrant 双写） | auto | 长期知识存储 |
| `semantic_memory_search` | 搜索语义记忆（向量/关键词/混合） | auto | 三种检索模式 |
| `semantic_memory_delete` | 删除语义记忆 | auto | 同时清理 MySQL+Qdrant |
| `semantic_memory_count` | 统计语义记忆条数 | auto | 只读 |

---

## RAG 检索架构

```
用户查询
  │
  ▼
① Query Rewrite
  ├── LLM 标准化、润色、纠错
  └── LLM 扩写为多条语义相似表达
  │
  ▼
② 三路并行检索（取 Top-30 候选）
  ├── Qdrant 稠密向量       ← DashScope text-embedding-v3 (1024维)
  ├── BM25 稀疏倒排         ← 自建 document_terms 表 + BM25Okapi
  └── MySQL LIKE 关键词     ← 同义词扩展，保留向下兼容
  │
  ▼
③ RRF 融合
  └── score = 1/(60+rank₁) + 1/(60+rank₂) + 1/(60+rank₃)
  │
  ▼
④ LLM 重排（可选）
  └── DeepSeek 对 Top-30 精排 → Top-5
  │
  ▼
⑤ 返回结果 + 来源标签
```

### BM25 倒排索引

| 表 | 行数 | 用途 | 更新时机 |
|----|------|------|---------|
| `document_terms` | ~2100 | term → {chunk_id → term_freq} 映射 | 上传文档时增量 |
| `doc_stats` | ~29 | chunk_id → 总词数（BM25 长度归一化用） | 上传文档时增量 |

### 优雅降级

| 组件不可用时 | 系统行为 |
|-------------|---------|
| Qdrant 连接失败 | 自动退化为 BM25 + LIKE 检索，用户无感知 |
| BM25 索引加载失败 | 退化为 Qdrant + LIKE 检索 |
| MySQL 不可用 | 跳过记忆加载，对话从空白开始 |

---

## 数据存储

| 数据库 | 表/集合 | 用途 | 数据量 |
|--------|---------|------|--------|
| MySQL `fojiao_db` | `document_chunks` | 分块原文（~500 字/块） | 29 行 |
| | `document_terms` | 倒排索引（term → chunk_id） | ~2100 行 |
| | `doc_stats` | 文档统计（BM25 归一化） | 29 行 |
| | `knowledge_bases` | 知识库元数据 | 3 行 |
| MySQL `ai_agent_db` | `conversation_history` | 对话历史（刷新不丢） | 按会话增长 |
| | `episodic_memories` | 情景记忆 | 按使用增长 |
| | `semantic_memories` | 语义记忆（含关键词索引） | 按使用增长 |
| Qdrant | `knowledge_vectors` | 文档向量（1024 维） | 按上传增长 |
| | `semantic_memory` | 语义记忆向量 | 按使用增长 |

---

## 记忆体系

| 层级 | 存储 | 查询方式 | 用途 | 生命周期 |
|------|------|---------|------|---------|
| 对话历史 | MySQL `conversation_history` | id ASC 顺序读取 | 页面刷新恢复对话 | 随会话 |
| 情景记忆 | MySQL `episodic_memories` | 关键词 LIKE 搜索 | "我们聊过什么" | 用户管理 |
| 语义记忆 | MySQL + Qdrant（双写） | 向量/关键词/混合 | "我记得的知识点" | 用户管理 |

### 上下文管理

滑动窗口策略：`MAX_HISTORY_ROUNDS=10`（默认）

```
对话 50 轮时：
  MySQL 全量存储（50 轮完整历史）
  发送给 LLM 的仅：第一轮 + 最近 9 轮 = 20 条消息恒定
  超出部分自动裁剪，不占用 LLM 上下文
```

---

## 安全设计

### 命令执行 — 三层防护

```
第一层：白名单
  ├── dir, type, find, echo, where, whoami
  ├── systeminfo, netstat, tasklist, ipconfig, ping
  ├── git, npm, curl, wget
  └── ← 不在白名单中的命令直接拒绝

第二层：黑名单
  ├── del, rd, rm, format, shutdown
  ├── taskkill, regedit, sc, wmic, runas
  └── ← 即使路径匹配也拦截

第三层：python/pip 特殊处理
  ├── python script.py       → 允许（脚本必须在项目目录内）
  ├── python -c "code"       → 拦截（任意代码执行）
  ├── pip list / pip freeze  → 允许
  └── pip install/uninstall  → 拦截（供应链攻击防护）
```

### 文件路径防护

```python
# 所有文件操作前执行
target = os.path.realpath(os.path.abspath(path))
if not target.startswith(base):
    raise PermissionError("路径越权")

# write_file 额外检查
validate_write_path(path)  # 限 ./data, ./output, ./static
```

### 审计日志

- 每次工具调用记录：时间、工具名、参数（脱敏）、耗时、结果状态
- 敏感字段模糊匹配：password / token / api_key / secret / auth / credential 等 16 个关键词
- 降级事件记录：组件名、原因、影响范围
- 日志文件：`src/logs/agent_audit.log`（10MB × 3 轮转）

---

## 超时与降级

```
┌─── CHAT_TIMEOUT = 120s ─────────── SSE 总超时（最外层兜底）
│
├─── ITERATION_TIMEOUT = 60s ─────── Agent 每轮迭代超时
│
├─── LLM_TIMEOUT = 30s ───────────── 单次 API 请求超时
│
└─── Shell 命令超时 = 15s ───────── subprocess.run(timeout=15)
```

---

## API 接口

| 方法 | 路径 | 说明 | 请求/响应 |
|------|------|------|-----------|
| `GET` | `/` | 聊天页面 | → `text/html` |
| `GET` | `/api/health` | 健康检查 | → `{"status":"healthy","model":"..."}` |
| `POST` | `/api/session` | 创建会话 | → `{"session_id":"abc123"}` |
| `POST` | `/api/chat` | SSE 流式对话 | `{"message":"...","session_id":"..."}` → SSE 事件流 |
| `GET` | `/api/history/{id}` | 会话历史 | → `{"session_id":"...","history":[...]}` |
| `DELETE` | `/api/history/{id}` | 删除会话 | → `{"deleted":"..."}` |
| `GET` | `/api/sessions/list` | 历史会话列表 | → `{"sessions":[...]}` |
| `POST` | `/api/upload` | 上传文档入库 | `multipart/form-data` → 6 步流水线结果 |
| `GET` | `/api/kb/list` | 知识库列表 | → `{"knowledge_bases":[...]}` |
| `POST` | `/api/kb/create` | 创建知识库 | `kb_name=...` → `{"kb_id":...}` |
| `GET` | `/api/kb/{id}/docs` | 知识库文档列表 | → `{"documents":[...]}` |
| `DELETE` | `/api/kb/{id}/doc/{did}` | 删除文档 | → `{"deleted_chunks":...}` |

---

## 环境变量

| 变量 | 默认值 | 必填 | 说明 |
|------|--------|------|------|
| `OPENAI_API_KEY` | — | ✅ | DeepSeek API Key |
| `DASHSCOPE_API_KEY` | — | ✅ | DashScope 文本嵌入 Key |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | ❌ | API 地址 |
| `LLM_MODEL` | `deepseek-v4-flash` | ❌ | 对话模型名 |
| `MAX_AGENT_ITERATIONS` | `12` | ❌ | Agent 最大循环轮次 |
| `MAX_HISTORY_ROUNDS` | `10` | ❌ | 上下文保留的对话轮数 |
| `LLM_TIMEOUT` | `30` | ❌ | API 请求超时（秒） |
| `ITERATION_TIMEOUT` | `60` | ❌ | Agent 迭代超时（秒） |
| `CHAT_TIMEOUT` | `120` | ❌ | SSE 总超时（秒） |
| `RERANK_ENABLED` | `false` | ❌ | 是否开启 LLM 重排 |
| `RRF_K` | `60` | ❌ | RRF 融合参数 |
| `RETRIEVER_TOP_K` | `5` | ❌ | 检索返回条数 |

---

## 测试

```bash
# 全量回归测试（79 项，覆盖所有工具 + 核心逻辑）
python tests/regression_test.py

# 手动验证
python -c "
import http.client, json
c = http.client.HTTPConnection('127.0.0.1', 8000, timeout=5)
c.request('GET', '/api/health')
print(c.getresponse().read().decode())
"
```

---

## 许可

MIT License
