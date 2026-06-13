# -*- coding: utf-8 -*-
# 捕获 run_agent 内部每次调用 LLM 时实际发送的消息列表
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from src.agent.agent_chain import stream_chat, create_default_session, _model

# 拦截 model.stream，记录每次调用的 messages
_original_stream = _model.__class__.stream
stream_calls = []

def _intercept_stream(self, *args, **kwargs):
    msgs = args[0] if args else kwargs.get("input", [])
    stream_calls.append([
        {
            "role": type(m).__name__,
            "content": getattr(m, "content", ""),
            **({"tool_calls": getattr(m, "tool_calls", None)} if hasattr(m, "tool_calls") and getattr(m, "tool_calls", None) else {}),
            **({"tool_call_id": getattr(m, "tool_call_id", None)} if hasattr(m, "tool_call_id") else {}),
        }
        for m in msgs
    ])
    for chunk in _original_stream(self, *args, **kwargs):
        yield chunk

_model.__class__.stream = _intercept_stream

# 模拟对话
session_id = create_default_session()
list(stream_chat("1+1等于几？用计算器工具算一下", session_id))

# 获取 memory 中的持久化数据
from src.agent.agent_chain import SessionManager
session = SessionManager().get_session(session_id)
memory_msgs = []
for msg in session.memory.chat_memory.messages:
    item = {"role": type(msg).__name__, "content": getattr(msg, "content", "")}
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        item["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id"):
        item["tool_call_id"] = msg.tool_call_id
    memory_msgs.append(item)

output = {
    "一_重要区别": {
        "memory存的消息": "存到磁盘/内存中，下次对话还能读到（持久化）",
        "run_agent临时消息": "只在本轮对话中存在，用完即丢（不持久化）",
        "结论": "工具调用过程（tool_calls + ToolMessage）只存在于临时消息中，不会存到 memory。下次对话 AI 不知道上轮调过什么工具。",
    },
    "二_每次调用LLM时实际发送的消息": [
        {"第1次调用LLM": stream_calls[0] if len(stream_calls) > 0 else []},
        {"第2次调用LLM_含工具结果": stream_calls[1] if len(stream_calls) > 1 else []},
    ],
    "三_memory中持久化的消息_下次对话还能看到": memory_msgs,
    "四_消息角色说明": {
        "SystemMessage": "系统提示词，告诉 AI 它是谁（run_agent 动态添加，不存 memory）",
        "HumanMessage": "用户的消息",
        "AIMessage": "AI 的完整回复（text 部分），含 tool_calls 表示要调工具",
        "AIMessageChunk": "AI 流式输出的片段（多 chunk 合并成 AIMessage）",
        "ToolMessage": "工具执行结果，通过 tool_call_id 匹配对应的 tool_call",
    },
}

output_path = Path(__file__).parent.parent / "logs" / "full_messages.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"保存到 logs/full_messages.json")
print(f"LLM 调用次数: {len(stream_calls)}")
print(f"Memory 消息数: {len(memory_msgs)}")
