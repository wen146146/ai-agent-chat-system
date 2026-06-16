# AI Agent Chat — 智能面试知识助手

基于 FastAPI + LangChain + MySQL + Qdrant 的面试知识库智能问答系统。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
# 复制 .env 并填入你的 API Key
# 需要: OPENAI_API_KEY (DeepSeek), DASHSCOPE_API_KEY (DashScope 文本嵌入)

# 3. 启动服务
uvicorn src.server.api_server:app --host 0.0.0.0 --port 8000

# 4. 打开浏览器
# http://localhost:8000
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Web 框架 | FastAPI + uvicorn | REST API + SSE 流式输出 |
| Agent | LangChain (bind_tools) | Function Calling + ReAct 循环 |
| 向量数据库 | Qdrant | 语义相似度检索 |
| 关系数据库 | MySQL 8.0+ | 知识库存储 + 记忆 + 倒排索引 |
| 中文分词 | jieba | BM25 索引 + 关键词提取 |
| LLM | DeepSeek (OpenAI 兼容) | 对话 + 重排 + 查询改写 |
| 嵌入模型 | DashScope text-embedding-v3 | 文本向量化 |
| 前端 | 原生 HTML/CSS/JS | Chat UI + 知识库管理 |

## 项目结构

```
src/
├── server/api_server.py        # FastAPI 服务（16 条路由）
├── agent/
│   ├── agent_chain.py          # Agent ReAct 循环（12 轮上限）
│   └── agent_prompt.py         # System Prompt
├── tools/
│   ├── tool_registry.py        # 工具注册中心（20 个工具）
│   ├── calculator_tool.py      # 数学计算
│   ├── web_search_tool.py      # 联网搜索 + 网页抓取
│   ├── rag_retriever_tool.py   # RAG 知识库检索（三路融合）
│   ├── file_tools.py           # 文件系统工具（读/写/搜索/列表）
│   ├── shell_tools.py          # Shell 命令工具（安全白名单）
│   ├── app_tools.py            # 应用控制工具（打开/列表）
│   ├── system_tools.py         # 系统信息工具（CPU/内存/进程）
│   └── memory/
│       ├── episodic_memory_tool.py  # 情景记忆
│       └── semantic_memory_tool.py  # 语义记忆（MySQL+Qdrant 双写）
├── knowledge_ingest/
│   ├── document_processor.py   # 文档分块
│   ├── file_converter.py       # 格式转换（PDF/DOCX/TXT/MD/CSV）
│   ├── chunk_store.py          # MySQL 分块存储
│   ├── vector_store.py         # Qdrant 向量存储
│   ├── bm25_index.py           # BM25 稀疏检索引擎
│   ├── bm25_indexer.py         # 倒排索引构建器
│   └── inverted_index_schema.py # 倒排索引表结构
└── utils/
    ├── config_loader.py        # 集中配置管理 + 路径校验
    ├── logger.py               # 错误日志
    └── audit.py                # 审计日志（参数脱敏）

static/
└── chat.html                   # 聊天 UI（来源卡片 + 打字机动画）

tests/
├── regression_test.py          # 全量回归测试（67 项）
├── debug_capture.py
├── dump_full_messages.py
└── dump_memory.py
```

## 工具清单（20 个）

### 已有工具（11 个）

| 工具 | 说明 | 权限 |
|------|------|------|
| `calculator` | 数学计算（加/减/乘/除/三角/对数/阶乘） | auto |
| `web_search` | DuckDuckGo 联网搜索 | auto |
| `web_fetch` | 网页内容抓取 | auto |
| `rag_retrieve` | RAG 知识库检索（三路融合） | auto |
| `episodic_memory_save` | 保存情景记忆 | auto |
| `episodic_memory_search` | 搜索情景记忆 | auto |
| `episodic_memory_delete` | 删除情景记忆 | auto |
| `semantic_memory_save` | 保存语义记忆 | auto |
| `semantic_memory_search` | 搜索语义记忆 | auto |
| `semantic_memory_delete` | 删除语义记忆 | auto |
| `semantic_memory_count` | 统计语义记忆数量 | auto |

### 新增工具（9 个）

| 工具 | 说明 | 权限 | 安全设计 |
|------|------|------|---------|
| `read_file` | 读取文件（100KB 上限） | auto | 路径规范化防穿越 |
| `write_file` | 写入文件（覆盖/追加） | user_confirm | 路径白名单 + 规范化 |
| `search_files` | glob 模式搜索文件 | auto | 限制搜索范围 |
| `list_directory` | 列出目录内容 | auto | 非递归 |
| `run_command` | 执行 Shell 命令 | user_confirm | 白名单 + 黑名单 + python/pip 特殊处理 |
| `open_application` | 打开应用程序 | user_confirm | 预定义映射表 |
| `list_applications` | 列出可打开的应用 | auto | 只读 |
| `get_system_info` | 系统信息（CPU/内存/磁盘） | auto | 只读 |
| `get_process_list` | 进程列表 | auto | 只读 |

## RAG 检索架构

```
用户查询
  ↓
 ① Query Rewrite（LLM 改写 + 扩写）
  ↓
 ② 三路并行检索
    ├── Qdrant 语义向量（DashScope 嵌入）
    ├── BM25 稀疏检索（jieba 分词 + 倒排索引）
    └── MySQL LIKE 关键词（同义词扩展，保留兼容）
  ↓
 ③ RRF 融合（Reciprocal Rank Fusion, k=60）
  ↓
 ④ LLM 重排（可选，DeepSeek 精排）
  ↓
 ⑤ 返回 Top-5 + 来源标签
```

### BM25 倒排索引

| 表 | 用途 | 更新时机 |
|----|------|---------|
| `document_terms` | term → {chunk_id → tf} | 上传文档时增量 |
| `doc_stats` | chunk_id → 总词数 + avgdl | 上传文档时增量 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 返回聊天页面 |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/session` | 创建新会话 |
| `POST` | `/api/chat` | SSE 流式对话 |
| `GET` | `/api/history/{id}` | 查询会话历史 |
| `DELETE` | `/api/history/{id}` | 删除会话 |
| `GET` | `/api/sessions` | 列出活跃会话 |
| `POST` | `/api/upload` | 上传文件到知识库 |
| `GET` | `/api/kb/list` | 知识库列表 |
| `POST` | `/api/kb/create` | 创建知识库 |
| `GET` | `/api/kb/{kb_id}/docs` | 文档列表 |
| `DELETE` | `/api/kb/{kb_id}/doc/{doc_id}` | 删除文档 |

## 安全设计

### 命令执行三层防护

1. **白名单** — 只允许 `dir`、`type`、`git`、`npm` 等预设命令
2. **黑名单** — `del`、`rm`、`shutdown`、`regedit` 等高危命令拦截
3. **python/pip 特殊处理** — `python -c "代码"` 拦截, `pip install` 拦截

### 文件路径防护

- `os.path.realpath()` 规范化路径防止 `../` 穿越
- `write_file` 额外检查写入目录白名单
- `read_file` 大小限制（默认 100KB）

### 审计日志

- 所有工具调用记录到 `src/logs/agent_audit.log`
- 敏感参数自动脱敏（password/token/api_key/secret 等 16 个关键词模糊匹配）

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | DeepSeek API Key |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | API Base URL |
| `DASHSCOPE_API_KEY` | — | DashScope 嵌入 Key |
| `LLM_MODEL` | `deepseek-v4-flash` | 对话模型 |
| `MAX_AGENT_ITERATIONS` | `12` | Agent 最大循环轮次 |
| `RERANK_ENABLED` | `false` | 是否开启 LLM 重排 |
| `RRF_K` | `60` | RRF 融合参数 |
| `RETRIEVER_TOP_K` | `5` | 检索返回条数 |

## 测试

```bash
# 全量回归测试（67 项）
python tests/regression_test.py
```
