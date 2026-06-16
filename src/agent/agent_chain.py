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

# ============================================================================
# 配置
# ============================================================================
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "12"))

# LangChain ChatOpenAI 客户端（原生支持 function calling）
_model = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    streaming=True,  # 启用流式输出
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
    "你是一个智能面试知识助手。\n\n"
    "核心能力：\n"
    "1. 知识问答 - 从知识库检索技术面试相关问题\n"
    "2. 工具调用 - 需要时可计算、搜索、操作文件\n"
    "3. 记忆功能 - 自动记住对话中的关键信息\n\n"
    "回答原则：\n"
    "- 专业、简洁、有引用来源\n"
    "- 需要工具时主动调用，不需要时不啰嗦\n"
    "- 如果工具返回结果，把结果整合到回答中"
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

    def _build_messages(self, user_message: str) -> list:
        """
        构建发给 LLM 的完整消息列表（使用 LangChain 原生消息类型）。
        system prompt + 历史对话（含工具调用） + 当前用户消息
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        # 直接使用 memory 中保存的 LangChain 消息对象
        for msg in self.memory.chat_memory.messages:
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

            # 流式调用，逐 chunk 收集并通知前端
            full_message: Optional[AIMessage] = None
            for chunk in model_with_tools.stream(messages):
                if full_message is None:
                    full_message = chunk
                else:
                    full_message += chunk
                # 逐字推送给前端
                if chunk.content:
                    callback.on_llm_new_token(chunk.content)

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

class StreamCallback:
    """收集工具调用和 token 事件，供前端实时展示"""
    def __init__(self, queue: list):
        self.queue = queue

    def on_llm_new_token(self, token: str):
        self.queue.append({"type": "token", "content": token})

    def on_tool_start(self, serialized: dict, input_str: str):
        name = serialized.get("name", "unknown") if serialized else "unknown"
        self.queue.append({"type": "tool_start", "name": name, "input": str(input_str)[:200]})

    def on_tool_end(self, output: str):
        self.queue.append({"type": "tool_result", "result": str(output)[:500]})


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
    event_queue: List[dict] = []
    callback = StreamCallback(event_queue)

    try:
        output_text = session.run_agent(message, callback)

        # 先发送工具调用等中间事件
        for event in event_queue:
            yield event

        # 保存本轮对话到 memory
        if output_text:
            session.memory.save_context(
                {"input": message},
                {"output": output_text}
            )

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
