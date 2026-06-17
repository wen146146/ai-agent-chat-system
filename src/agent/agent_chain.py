# -*- coding: utf-8 -*-
# ============================================================================
# Agent 编排核心（原生 Function Calling 版）
# ============================================================================
# 作用：
#   1. 通过 ChatOpenAI + bind_tools() 实现原生的 LLM 工具调用
#   2. LLM 自动根据工具的 args_schema 决定调用哪个工具、传什么参数
#   3. Agent 循环：调用 LLM → 检查 tool_calls → 执行工具 → 回传结果 → 最终回复
#   4. 会话管理（SessionManager + ChatSession）
#
# 核心改进：
#   - 不再需要 response_format JSON 强制、build_system_prompt 手写工具列表
#   - 不再需要 arguments_desc（LLM 从 BaseModel Schema 自动获取参数结构）
#   - CalculatorInput 等 Schema 真正发挥作用：既校验参数，又告诉 LLM 参数格式
# ============================================================================

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import sys
import uuid
import json
from pathlib import Path
from typing import Dict, Generator, Any, Optional, List
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferMemory

# 日志模块
from src.utils.logger import logger
from src.utils.audit import log_tool_call
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 对话历史 MySQL 持久化
from src.server.chat_store import ChatStore, ensure_table
ensure_table()

# ============================================================================
# 配置
# ============================================================================
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "12"))
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
ITERATION_TIMEOUT = int(os.getenv("ITERATION_TIMEOUT", "60"))

# LangChain ChatOpenAI 客户端（原生支持 function calling）
_model = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    streaming=True,  # 启用流式输出
    timeout=LLM_TIMEOUT,  # LLM 请求超时（秒）
)


# ============================================================================
# 原生工具注册：直接从 Registry 拿 @tool 实例
# ============================================================================
from src.tools.tool_registry import ToolRegistry

_registry = ToolRegistry.get_instance()
NATIVE_TOOLS = _registry.collect_native_tools()

# 工具名 → 原生工具实例 快速查找
TOOL_MAP: Dict[str, BaseTool] = {t.name: t for t in NATIVE_TOOLS}


# ============================================================================
# System Prompt（简化版，不列工具，模型从 bind_tools 自动获取）
# ============================================================================

SYSTEM_PROMPT = (
    "你是一个万事通智能助手，名叫【面试知识助手】。你无所不知，但更擅长利用工具和知识库来给出准确答案。\n\n"
    "【角色定位】\n"
    "你是一个万事通——面试知识、技术原理、编程问题、系统操作，样样精通。\n"
    "但你不靠猜测回答，而是通过工具查找、知识库检索和记忆回顾来确保答案准确。\n\n"
    "【核心能力 — 你拥有以下工具】\n"
    "📚 知识库检索（rag_retrieve）\n"
    "  - 技术概念、面试题、原理分析 → 先查知识库，给出有来源的答案\n"
    "🌐 联网搜索（web_search / web_fetch）\n"
    "  - 最新资讯、实时数据、知识库查不到的内容 → 联网获取\n"
    "🧮 数学计算（calculator）\n"
    "  - 数字运算 → 调用计算器而非自己算\n"
    "🧠 记忆功能（episodic_memory / semantic_memory）\n"
    "  - 保存和回顾对话记录、用户偏好、知识总结\n"
    "📁 文件操作（read_file / write_file / search_files / list_directory）\n"
    "  - 读取、写入、搜索文件和目录\n"
    "💻 系统操作（run_command / get_system_info / get_process_list）\n"
    "  - 执行命令、查看系统信息、进程列表\n"
    "🖥️ 应用控制（open_application / list_applications）\n"
    "  - 打开本地应用程序\n\n"
    "【回答原则】\n"
    "1. 专业、简洁、有引用来源\n"
    "2. 需要工具时主动调用，不需要时不啰嗦\n"
    "3. 工具返回结果要整合到回答中，不要重复原文\n"
    "4. 不知道就说不知道，不要编造\n"
    "5. 多轮对话中注意上下文连贯性\n\n"
    "【安全规范】\n"
    "1. 永远不要执行删除文件、格式化磁盘、关机等危险操作\n"
    "2. 写入文件仅限于项目目录（./data, ./output, ./static）\n"
    "3. 执行命令只允许白名单中的命令（dir/git/npm/ipconfig 等）\n"
    "4. 不要安装或卸载任何软件包（pip install/uninstall 被拦截）\n"
    "5. 不要执行 python -c 任意代码\n"
    "6. 保护用户隐私，不记录敏感信息（密码、token、Key 等）\n"
    "7. 如果用户要求做危险操作，礼貌拒绝并解释原因"
)


# ============================================================================
# ChatSession — 单个会话的全部状态
# ============================================================================

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        # 从 MySQL 加载历史对话到 memory（页面刷新后恢复）
        self._load_history_from_db()

    def _load_history_from_db(self):
        """从 MySQL 加载该会话的历史消息到 ConversationBufferMemory。"""
        try:
            store = ChatStore()
            history = store.get_history(self.session_id)
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    self.memory.chat_memory.add_user_message(content)
                elif role == "assistant":
                    self.memory.chat_memory.add_ai_message(content)
        except Exception:
            pass  # MySQL 不可用时静默跳过

    def _build_messages(self, user_message: str) -> list:
        """
        构建发给 LLM 的完整消息列表（滑动窗口）。
        system prompt + 第一轮 + 最近 N-1 轮 + 当前用户消息。
        MAX_HISTORY_ROUNDS 控制保留的对话轮数，防止上下文无限增长。
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        all_msgs = self.memory.chat_memory.messages
        max_msg_count = MAX_HISTORY_ROUNDS * 2  # 每轮 user + assistant

        if len(all_msgs) > max_msg_count:
            # 保留第一轮（2 条）+ 最近 N-1 轮
            recent_msgs = all_msgs[:2] + all_msgs[-(max_msg_count - 2):]
        else:
            recent_msgs = all_msgs

        for msg in recent_msgs:
            messages.append(msg)
        messages.append(HumanMessage(content=user_message))
        return messages

    def run_agent(self, user_message: str, callback) -> str:
        """
        Agent 主循环（原生 function calling）。
        流程：
          1. 构建消息列表
          2. 绑定工具 → model.bind_tools(NATIVE_TOOLS)
          3. 流式调用 LLM，收集完整 AIMessage
          4. 检查是否有 tool_calls：
             - 有 → 执行工具 → 追加结果到消息列表 → 回到步骤2
             - 无 → 返回 content
        最多循环 {MAX_AGENT_ITERATIONS} 轮，防止死循环。
        """
        messages = self._build_messages(user_message)

        for _ in range(MAX_AGENT_ITERATIONS):
            # 绑定工具到模型（每次调用都绑，确保一致性）
            model_with_tools = _model.bind_tools(NATIVE_TOOLS)

            # 流式调用，逐 chunk 收集并通知前端（带超时）
            full_message: Optional[AIMessage] = None

            def _run_llm_stream():
                """在子线程中执行一轮 LLM 流式调用。"""
                nonlocal full_message
                for chunk in model_with_tools.stream(messages):
                    if full_message is None:
                        full_message = chunk
                    else:
                        full_message += chunk
                    if chunk.content:
                        callback.on_llm_new_token(chunk.content)
                return full_message

            try:
                with ThreadPoolExecutor() as pool:
                    future = pool.submit(_run_llm_stream)
                    full_message = future.result(timeout=ITERATION_TIMEOUT)
            except TimeoutError:
                callback.on_tool_end(f"[超时] LLM 响应超过 {ITERATION_TIMEOUT}s")
                logger.warning(f"[agent] 会话 {self.session_id} 迭代超时 ({ITERATION_TIMEOUT}s)")
                return f"处理超时，请简化问题后重试（超过 {ITERATION_TIMEOUT} 秒）"

            # 检查原生 tool_calls
            if full_message and full_message.tool_calls:
                # 把 AIMessage（含 tool_calls）追加到对话
                messages.append(full_message)

                for tc in full_message.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]

                    # 通知前端
                    callback.on_tool_start(
                        {"name": tool_name},
                        json.dumps(tool_args, ensure_ascii=False)
                    )

                    # 执行原生工具（带计时和审计）
                    import time as _time
                    _t0 = _time.time()
                    tool = TOOL_MAP.get(tool_name)
                    if tool:
                        try:
                            result = str(tool.invoke(tool_args))
                            log_tool_call(tool_name, tool_args, "success", int((_time.time() - _t0) * 1000))
                        except Exception as e:
                            result = f"[工具执行错误] {e}"
                            log_tool_call(tool_name, tool_args, "error", int((_time.time() - _t0) * 1000))
                    else:
                        result = f"[错误] 工具 {tool_name} 不存在"
                        log_tool_call(tool_name, tool_args, "error", 0)

                    callback.on_tool_end(result[:500])

                    # 追加 ToolMessage 到对话
                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tc["id"],
                    ))
                # 继续循环，让 LLM 基于工具结果生成回复
                continue
            else:
                # 无工具调用 → 最终回复
                return full_message.content if full_message else ""

        return "抱歉，工具调用次数过多，请简化问题后重试。"

    def clear_history(self):
        self.memory.clear()

    def get_chat_history(self) -> list:
        messages = self.memory.chat_memory.messages
        result = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = "user" if msg.type == "human" else "assistant"
                result.append({"role": role, "content": msg.content})
        return result


# ============================================================================
# SessionManager — 全局会话管理（单例）
# ============================================================================

class SessionManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
        return cls._instance

    def get_session(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> list:
        return [
            {"session_id": sid, "history_count": len(sess.get_chat_history())}
            for sid, sess in self._sessions.items()
        ]


# ============================================================================
# StreamCallback — 前端流式事件收集
# ============================================================================

import queue

class StreamCallback:
    """收集工具调用和 token 事件，供前端实时展示（线程安全，使用 queue.Queue）"""
    def __init__(self, q: queue.Queue):
        self.queue = q

    def on_llm_new_token(self, token: str):
        self.queue.put({"type": "token", "content": token})

    def on_tool_start(self, serialized: dict, input_str: str):
        name = serialized.get("name", "unknown") if serialized else "unknown"
        self.queue.put({"type": "tool_start", "name": name, "input": str(input_str)[:200]})

    def on_tool_end(self, output: str):
        self.queue.put({"type": "tool_result", "result": str(output)[:500]})


def drain_queue(q: queue.Queue) -> list:
    """安全地排出 queue.Queue 中所有事件，返回 list。"""
    events = []
    while not q.empty():
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


# ============================================================================
# stream_chat — 对外暴露的流式对话入口
# ============================================================================

def stream_chat(message: str, session_id: str) -> Generator[dict, None, None]:
    """
    流式对话入口函数。
    被 api_server.py 的 /api/chat 接口调用。

    参数：
      - message: 用户输入的消息文本
      - session_id: 会话ID，用于保持多轮对话上下文

    产出（Generator）：
      - {"type": "token", "content": "..."}     LLM 逐字输出
      - {"type": "tool_start", "name": "..."}    开始调用工具
      - {"type": "tool_result", "result": "..."} 工具返回结果
      - {"type": "final_output", "content": "..."} 最终回复
      - {"type": "error", "content": "..."}      错误信息
      - {"type": "end"}                          流结束
    """
    session = SessionManager().get_session(session_id)
    event_queue: queue.Queue = queue.Queue()
    callback = StreamCallback(event_queue)

    try:
        output_text = session.run_agent(message, callback)

        # 先发送工具调用等中间事件（从线程安全队列取出）
        for event in drain_queue(event_queue):
            yield event

        # 保存本轮对话到 memory
        if output_text:
            session.memory.save_context(
                {"input": message},
                {"output": output_text}
            )
            # 持久化到 MySQL（页面刷新后可以恢复）
            try:
                store = ChatStore()
                store.save_message(session_id, "user", message)
                store.save_message(session_id, "assistant", output_text)
            except Exception:
                pass  # MySQL 写入失败不影响主流程

        yield {"type": "final_output", "content": output_text}
        yield {"type": "end"}

    except Exception as e:
        logger.error(str(e), exc_info=True)
        yield {"type": "error", "content": "服务处理异常，请稍后重试"}
        yield {"type": "end"}


def create_default_session() -> str:
    """创建新会话，返回8位随机 session_id"""
    session_id = str(uuid.uuid4())[:8]
    SessionManager().get_session(session_id)
    return session_id
