# -*- coding: utf-8 -*-
# ============================================================================
# FastAPI Web服务层（API Server）
# ============================================================================
# 作用：
#   整个项目的 HTTP 入口，对外暴露 RESTful API。
#   启动命令：uvicorn src.server.api_server:app --host 0.0.0.0 --port 8000
#
# 接口清单（12个）：
#   GET  /                        — 返回聊天页面（chat.html）
#   GET  /api/health              — 健康检查
#   POST /api/session             — 创建新会话
#   POST /api/chat                — 核心对话接口（SSE流式返回）
#   GET  /api/history/{id}        — 查询会话历史
#   DELETE /api/history/{id}      — 删除会话
#   GET  /api/sessions            — 列出所有活跃会话
#   POST /api/upload              — 知识库文件上传（5步流水线）
#   GET  /api/kb/list              — 知识库列表
#   POST /api/kb/create            — 创建知识库
#   GET  /api/kb/{kb_id}/docs     — 获取知识库文档列表
#   DELETE /api/kb/{kb_id}/doc/{doc_id} — 删除文档
# ============================================================================

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import sys
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv

# 把项目根目录加入 sys.path，确保 from src.xxx 能正确导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# 启动时检查必要环境变量（仅打印警告，不阻塞启动）
from src.utils.config_loader import validate_keys_at_startup
_startup_warnings = validate_keys_at_startup()
for _w in _startup_warnings:
    print(_w)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import tempfile
import shutil
import time

# 从Agent编排模块导入核心功能
from src.agent.agent_chain import stream_chat, SessionManager, create_default_session

# 日志模块
from src.utils.logger import logger

# ============================================================================
# 服务配置
# ============================================================================
HOST = os.getenv("HOST", "0.0.0.0")  # 监听地址，0.0.0.0=接受所有网卡请求
PORT = int(os.getenv("PORT", "8000"))  # 监听端口

# 创建 FastAPI 应用实例
app = FastAPI(title="AI Agent Chat API", version="1.0")

# ============================================================================
# CORS 跨域配置（允许前端任意域名访问）
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源
    allow_credentials=True,        # 允许携带Cookie
    allow_methods=["*"],           # 允许所有HTTP方法
    allow_headers=["*"],           # 允许所有请求头
)


# ============================================================================
# 请求模型定义
# ============================================================================

class ChatRequest(BaseModel):
    """
    对话请求体。
    前端每次发送消息时提交的JSON格式：
    {"message": "你好", "session_id": "abc123"}

    字段说明：
      - message: 用户输入的文本（必填，不能为空）
      - session_id: 会话ID，用于保持多轮对话上下文。
                    首次对话留空，服务端自动创建并返回。
                    后续对话传同一个ID，服务端会复用历史记录。
    """
    message: str
    session_id: str = ""


class HistoryRequest(BaseModel):
    """
    会话历史查询请求体（当前未使用，预留给扩展接口）。
    """
    session_id: str


# ============================================================================
# 接口1: GET /  — 返回前端聊天页面
# ============================================================================

@app.get("/")
async def root():
    """
    访问根路径时，直接返回 static/chat.html 静态文件。
    用户打开浏览器访问 http://localhost:8000 就能看到聊天界面。

    返回值：
      - FileResponse: 发送 static/chat.html 文件内容到浏览器
    """
    # 通过 Path(__file__).parent.parent.parent 定位到项目根目录
    # 再拼接 static/chat.html 路径
    return FileResponse(
        Path(__file__).parent.parent.parent / "static" / "chat.html"
    )


# ============================================================================
# 接口2: GET /api/health  — 健康检查
# ============================================================================

@app.get("/api/health")
async def health():
    """
    服务存活探测，运维/监控用。
    返回当前服务状态、活跃会话数、使用的LLM模型。

    返回值示例：
      {"status": "healthy", "sessions": 3, "model": "deepseek-v4-flash"}

    调用链路：
      → SessionManager.list_sessions() 返回所有活跃会话列表
      → len() 统计数量
      → os.getenv("LLM_MODEL") 读取当前模型名
    """
    return {
        "status": "healthy",
        "sessions": len(SessionManager().list_sessions()),
        "model": os.getenv("LLM_MODEL", "unknown"),
    }


# ============================================================================
# 接口3: POST /api/session  — 创建新会话
# ============================================================================

@app.post("/api/session")
async def create_session():
    """
    创建全新的对话会话，返回一个8位随机session_id。
    前端在首次打开聊天页面时调用此接口，拿到session_id后
    所有后续对话请求都带上这个ID，实现多轮对话记忆。

    返回值示例：
      {"session_id": "a1b2c3d4"}

    调用链路：
      → create_default_session()
        → uuid.uuid4() 生成随机UUID，取前8位
        → SessionManager.get_session(id) 创建 ChatSession 实例
        → 返回 session_id
    """
    session_id = create_default_session()
    return {"session_id": session_id}


# ============================================================================
# 接口4: POST /api/chat  — 核心对话接口（SSE流式）
# ============================================================================

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    整个项目最核心的接口：接收用户消息，流式返回AI回复。

    请求体示例：
      {"message": "你好", "session_id": "a1b2c3d4"}

    流式返回格式（SSE = Server-Sent Events）：
      每条事件一行，格式为 "data: {JSON}\n\n"

    事件类型：
      - {"type": "token", "content": "你"}     ← LLM逐字输出的token
      - {"type": "tool_start", "name": "..."}  ← 开始调用工具
      - {"type": "tool_result", "result": "..."} ← 工具返回结果
      - {"type": "final_output", "content": "..."} ← 最终完整回复
      - {"type": "error", "content": "..."}     ← 错误信息
      - {"type": "end"}                         ← 流结束标记

    处理流程：
      1. 校验 message 不为空
      2. session_id 为空则自动创建新会话
      3. stream_chat() 逐条生成事件（Generator模式）
      4. 每条事件序列化为 JSON，用 SSE 格式发送给前端
      5. 前端 EventSource.onmessage 逐条接收并渲染
    """
    # 校验：消息不能为空
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    # session_id为空则创建新会话，保证每个请求都有上下文
    if not req.session_id:
        session_id = create_default_session()
    else:
        session_id = req.session_id

    def generate():
        """
        SSE 事件生成器。
        这是一个Python Generator函数，yield 每条SSE消息。
        FastAPI 的 StreamingResponse 会逐条发送到客户端，
        实现"边生成边发送"的流式效果。
        """
        try:
            # stream_chat 是核心编排函数，返回 Generator[dict, None, None]
            # 逐条产出 token/工具调用/最终结果/结束 等事件
            for event in stream_chat(req.message, session_id):
                # 每条事件包装成 SSE 格式：data: {JSON}\n\n
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[chat] {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': '服务处理异常，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",  # SSE 标准 MIME 类型
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲，保证实时推送
        }
    )


# ============================================================================
# 接口5: GET /api/history/{session_id}  — 查询会话历史
# ============================================================================

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """
    查询指定会话的完整对话历史。

    URL示例：
      GET /api/history/a1b2c3d4

    返回值示例：
      {
        "session_id": "a1b2c3d4",
        "history": [
          {"role": "user", "content": "你好"},
          {"role": "assistant", "content": "你好！有什么可以帮你？"},
        ],
        "count": 2
      }

    调用链路：
      → SessionManager.get_session(session_id) 获取 ChatSession 实例
      → session.get_chat_history() 从 ConversationBufferMemory 提取消息
        → 遍历 memory.chat_memory.messages
        → 每条消息转成 {"role": "user/assistant", "content": "..."}
    """
    try:
        session = SessionManager().get_session(session_id)
        history = session.get_chat_history()
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"[history] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="查询会话失败，请稍后重试")


# ============================================================================
# 接口6: DELETE /api/history/{session_id}  — 删除会话
# ============================================================================

@app.delete("/api/history/{session_id}")
async def delete_history(session_id: str):
    """
    删除指定会话及其所有对话历史。

    URL示例：
      DELETE /api/history/a1b2c3d4

    返回值示例：
      {"deleted": "a1b2c3d4"}

    调用链路：
      → SessionManager.delete_session(session_id)
        → 从 _sessions 字典中移除该 session_id
        → ChatSession 实例被Python垃圾回收，ConversationBufferMemory随之销毁
    """
    try:
        SessionManager().delete_session(session_id)
        return {"deleted": session_id}
    except Exception as e:
        logger.error(f"[delete] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="删除会话失败，请稍后重试")


# ============================================================================
# 接口7: GET /api/sessions  — 列出所有活跃会话
# ============================================================================

@app.get("/api/sessions")
async def list_sessions():
    """
    列出当前服务端内存中所有活跃的会话。

    返回值示例：
      {
        "sessions": [
          {"session_id": "a1b2c3d4", "history_count": 5},
          {"session_id": "e5f6g7h8", "history_count": 2}
        ]
      }

    注意：
      - 会话存储在内存中（SessionManager._sessions字典），服务重启后全部丢失
      - history_count 是当前会话中已有的对话轮次数
    """
    return {"sessions": SessionManager().list_sessions()}


# ============================================================================
# 接口8: POST /api/upload  — 知识库文件上传（5步入库流水线）
# ============================================================================

@app.post("/api/upload")
async def upload_file(
        # --- 请求参数 ---
        file: UploadFile = File(...),
        # 知识库ID，默认1。同一个知识库内的文档可以被一起检索。
        kb_id: int = Form(default=1),
        # 文档ID，默认以时间戳作为唯一标识。同一个文档的多个分块共享此ID。
        doc_id: int = Form(default=0),
):
    """
    上传文件到知识库，完成"格式转换→分块→MySQL写入→Qdrant向量化"全流程。

    请求方式：multipart/form-data
    参数：
      - file: 上传的文件（必填），支持 .txt/.md/.pdf/.docx/.csv
      - kb_id: 知识库ID（可选，默认1）
      - doc_id: 文档ID（可选，默认自动生成时间戳ID）

    处理流程（6步）：
      步骤1 - 读文件：读取上传的二进制内容，记录文件名和大小
      步骤2 - 格式转换：调用 file_converter.convert_to_markdown()
               PDF→pypdf逐页提取 / DOCX→python-docx解析
               TXT/MD/CSV→直接读取，统一转为Markdown文本
      步骤3 - 文本分块：调用 DocumentProcessor.process_document()
               用 LangChain RecursiveCharacterTextSplitter 按500字/块切开，
               相邻块重叠50字，确保信息不丢失
      步骤4 - MySQL存储：调用 ChunkStore.save_chunks()
               将分块写入 document_chunks 表（含FULLTEXT全文索引）
               先删旧记录（按doc_id），再插入新记录（幂等覆盖）
      步骤5 - Qdrant向量化：调用 VectorStore.add_chunks()
               每个分块→DashScope嵌入→1024维向量→Qdrant upsert
               先删旧向量，再插入新向量
      步骤6 - 倒排索引构建：调用 InvertedIndexBuilder.add_chunks()
               jieba分词 → 写入 document_terms + doc_stats
               先删旧倒排（按doc_id），再插入新记录

    返回值示例：
      {
        "success": true,
        "filename": "Python教程.pdf",
        "file_size": 123456,
        "doc_id": 1712345678,
        "kb_id": 1,
        "chunk_count": 15,
        "mysql_saved": 15,
        "qdrant_saved": 15,
        "elapsed": 2.35,
        "steps": [
          {"step": 1, "name": "read_file", "detail": "Python教程.pdf (123,456 bytes)"},
          {"step": 2, "name": "convert", "detail": "Markdown 8234 chars"},
          {"step": 3, "name": "chunk", "detail": "15 chunks"},
          {"step": 4, "name": "MySQL", "detail": "15/15 rows"},
          {"step": 5, "name": "Qdrant", "detail": "15/15 vectors"}
        ]
      }

    容错设计：
      - MySQL或Qdrant任一写入失败，不影响另一个（独立try-catch）
      - 任一成功即返回 success: true
      - steps 数组记录每个步骤的执行情况，方便排查问题
    """
    # --- 参数校验 ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file selected")

    # 延迟导入：避免模块加载时初始化数据库连接
    from src.knowledge_ingest.file_converter import convert_to_markdown
    from src.knowledge_ingest.document_processor import DocumentProcessor
    from src.knowledge_ingest.chunk_store import ChunkStore
    from src.knowledge_ingest.vector_store import VectorStore
    from src.knowledge_ingest.bm25_indexer import InvertedIndexBuilder

    start_time = time.time()
    filename = file.filename
    ext = Path(filename).suffix.lower()  # 取文件扩展名，如 ".pdf"

    # doc_id 未指定则用当前时间戳作为唯一标识
    if doc_id <= 0:
        doc_id = int(time.time())

    # ============================
    # 步骤1: 读取文件内容
    # ============================
    try:
        content = await file.read()  # await: FastAPI的UploadFile.read()是异步的
        file_size = len(content)
    except Exception:
        content = b""
        file_size = 0

    if not content:
        raise HTTPException(status_code=400, detail="file content is empty")

    steps = []
    steps.append({
        "step": 1,
        "name": "read_file",
        "detail": f"{filename} ({file_size:,} bytes)"
    })

    # --- 纯文本格式直接解码为字符串 ---
    if ext in (".txt", ".md", ".csv"):
        text_content = content.decode("utf-8", errors="replace")
    else:
        # PDF/DOCX等二进制格式需要先写入临时文件，
        # 因为 pypdf/python-docx 需要文件路径而不是内存字节
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            # 二进制格式先读字节，convert_to_markdown内部处理
            with open(tmp_path, "rb") as f:
                text_content = f.read()
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ============================
    # 步骤2: 格式转换 → 统一Markdown
    # ============================
    try:
        # convert_to_markdown 入口函数，根据文件扩展名自动分发到对应转换器：
        #   .txt  → _convert_txt()
        #   .md   → _convert_md()
        #   .pdf  → _convert_pdf()  使用 pypdf.PdfReader
        #   .docx → _convert_docx() 使用 python-docx
        #   .csv  → _convert_csv()  使用 csv.reader
        markdown_text = convert_to_markdown(text_content, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"format conversion failed: {str(e)}")

    # 检查转换结果是否包含错误前缀
    if markdown_text.startswith("[error]") or markdown_text.startswith("[Error]"):
        raise HTTPException(status_code=500, detail=markdown_text)

    steps.append({
        "step": 2,
        "name": "convert",
        "detail": f"Markdown {len(markdown_text)} chars"
    })

    # ============================
    # 步骤3: 文本分块
    # ============================
    processor = DocumentProcessor()
    # process_document 内部流程：
    #   1. LangChainTextSplitter.split_text() 按500字/块拆分
    #   2. 为每个块生成 MD5(doc_id + chunk_index) 作为 chunk_id
    #   3. 返回 List[DocumentChunk]，每个chunk包含 content + metadata
    chunks = processor.process_document(
        markdown_text=markdown_text,
        doc_id=doc_id,
        kb_id=kb_id,
        filename=filename,
        file_type=ext.lstrip(".")  # 去掉点号，".pdf" → "pdf"
    )

    steps.append({
        "step": 3,
        "name": "chunk",
        "detail": f"{len(chunks)} chunks"
    })

    # ============================
    # 步骤4: MySQL存储
    # ============================
    mysql_saved = 0
    mysql_error = None
    try:
        chunk_store = ChunkStore()
        # 先删旧记录：确保同一doc_id重复上传时是覆盖而非追加
        chunk_store.delete_chunks_by_document(doc_id)
        # 批量插入分块：每条 INSERT ... ON DUPLICATE KEY UPDATE（幂等写入）
        mysql_saved = chunk_store.save_chunks(chunks)
        steps.append({
            "step": 4,
            "name": "MySQL",
            "detail": f"{mysql_saved}/{len(chunks)} rows"
        })
    except Exception as e:
        mysql_error = str(e)
        steps.append({
            "step": 4,
            "name": "MySQL",
            "detail": f"failed: {mysql_error}"
        })

    # ============================
    # 步骤5: Qdrant向量化存储
    # ============================
    qdrant_saved = 0
    qdrant_error = None
    try:
        vector_store = VectorStore()
        # 先删旧向量：确保同一doc_id的旧向量被清除
        vector_store.delete_by_document(doc_id)
        # 向量化并存入Qdrant：
        #   遍历每个chunk → EmbeddingService.embed(content)
        #   → DashScope text-embedding-v4 生成1024维向量
        #   → Qdrant.upsert() 存入 knowledge_vectors 集合
        qdrant_saved = vector_store.add_chunks(chunks)
        steps.append({
            "step": 5,
            "name": "Qdrant",
            "detail": f"{qdrant_saved}/{len(chunks)} vectors"
        })
    except Exception as e:
        qdrant_error = str(e)
        steps.append({
            "step": 5,
            "name": "Qdrant",
            "detail": f"failed: {qdrant_error}"
        })

    elapsed = round(time.time() - start_time, 2)

    # ============================
    # 步骤6: 倒排索引构建（BM25）
    # ============================
    inverted_saved = 0
    inverted_error = None
    try:
        builder = InvertedIndexBuilder()
        # 先删旧倒排（幂等覆盖，同doc_id重复上传时清理旧记录）
        builder.remove_document(doc_id)
        # 写入新倒排
        builder.add_chunks(chunks)
        inverted_saved = len(chunks)
        steps.append({
            "step": 6,
            "name": "倒排索引",
            "detail": f"{inverted_saved}/{len(chunks)} terms indexed"
        })
    except Exception as e:
        inverted_error = str(e)
        steps.append({
            "step": 6,
            "name": "倒排索引",
            "detail": f"failed: {inverted_error}"
        })

    return {
        # success判断：MySQL、Qdrant、倒排索引任一成功即为成功
        "success": True if mysql_saved > 0 or qdrant_saved > 0 or inverted_saved > 0 else False,
        "filename": filename,
        "file_size": file_size,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "chunk_count": len(chunks),
        "mysql_saved": mysql_saved,
        "qdrant_saved": qdrant_saved,
        "inverted_saved": inverted_saved,
        "elapsed": elapsed,
        "steps": steps,  # 每步执行详情，前端可展示进度条
    }


# ============================================================================
# 知识库管理接口
# ============================================================================

@app.get("/api/kb/list")
async def list_knowledge_bases():
    """获取知识库列表"""
    try:
        from src.knowledge_ingest.chunk_store import ChunkStore
        chunk_store = ChunkStore()
        conn = chunk_store._get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT kb_id, COUNT(*) as doc_count, COUNT(DISTINCT doc_id) as docs
                FROM document_chunks
                GROUP BY kb_id
                ORDER BY kb_id
            """)
            rows = cursor.fetchall()
        conn.close()
        bases = []
        for r in rows:
            bases.append({
                "kb_id": r["kb_id"],
                "doc_count": r["doc_count"],
                "docs": r["docs"]
            })
        return {"success": True, "knowledge_bases": bases}
    except Exception as e:
        logger.error(f"[kb_list] {str(e)}", exc_info=True)
        return {"success": False, "error": "获取知识库列表失败", "knowledge_bases": []}


@app.post("/api/kb/create")
async def create_knowledge_base(kb_name: str = Form(...)):
    """创建知识库（目前只是逻辑概念，通过kb_id区分）"""
    import time
    kb_id = int(time.time())
    return {"success": True, "kb_id": kb_id, "name": kb_name}


@app.get("/api/kb/{kb_id}/docs")
async def list_documents(kb_id: int):
    """获取知识库中的文档列表"""
    try:
        from src.knowledge_ingest.chunk_store import ChunkStore
        chunk_store = ChunkStore()
        conn = chunk_store._get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT doc_id, filename, file_type, COUNT(*) as chunk_count, MAX(created_at) as updated
                FROM document_chunks
                WHERE kb_id = %s
                GROUP BY doc_id, filename, file_type
                ORDER BY updated DESC
            """, (kb_id,))
            rows = cursor.fetchall()
        conn.close()
        docs = []
        for r in rows:
            docs.append({
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "file_type": r["file_type"],
                "chunk_count": r["chunk_count"],
                "updated": str(r["updated"]) if r["updated"] else ""
            })
        return {"success": True, "kb_id": kb_id, "documents": docs}
    except Exception as e:
        logger.error(f"[kb_docs] {str(e)}", exc_info=True)
        return {"success": False, "error": "获取文档列表失败", "documents": []}


@app.delete("/api/kb/{kb_id}/doc/{doc_id}")
async def delete_document(kb_id: int, doc_id: int):
    """删除知识库中的指定文档（同时清理倒排索引和向量）"""
    try:
        from src.knowledge_ingest.chunk_store import ChunkStore
        from src.knowledge_ingest.vector_store import VectorStore
        from src.knowledge_ingest.bm25_indexer import InvertedIndexBuilder
        chunk_store = ChunkStore()
        vector_store = VectorStore()
        deleted_chunks = chunk_store.delete_chunks_by_document(doc_id)
        deleted_vectors = vector_store.delete_by_document(doc_id)
        # 清理倒排索引
        builder = InvertedIndexBuilder()
        builder.remove_document(doc_id)
        return {
            "success": True,
            "deleted_chunks": deleted_chunks,
            "deleted_vectors": deleted_vectors,
            "inverted_cleaned": True
        }
    except Exception as e:
        logger.error(f"[kb_delete] {str(e)}", exc_info=True)
        return {"success": False, "error": "删除文档失败"}


# ============================================================================
# 启动入口：python src/server/api_server.py 或 uvicorn src.server.api_server:app
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
