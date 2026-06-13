# -*- coding: utf-8 -*-
# 模拟一段含工具调用的对话，导出 memory 中存储的历史消息
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from src.agent.agent_chain import stream_chat, create_default_session

session_id = create_default_session()

# 模拟对话：会触发计算器工具
events = list(stream_chat("1+1等于几？用计算器工具算一下", session_id))

# 获取 memory 中存的历史消息
from src.agent.agent_chain import SessionManager
session = SessionManager().get_session(session_id)

history = []
for msg in session.memory.chat_memory.messages:
    item = {"role": type(msg).__name__}
    item["content"] = getattr(msg, "content", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        item["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id"):
        item["tool_call_id"] = msg.tool_call_id
    history.append(item)

output = {
    "说明": "ConversationBufferMemory.chat_memory.messages 中存储的所有消息",
    "消息角色说明": {
        "SystemMessage":    "系统提示词（run_agent 里临时拼的，不存 memory）",
        "HumanMessage":     "用户说的话",
        "AIMessage":        "AI 的回复（含 tool_calls 时也会存成 AIMessage）",
        "ToolMessage":      "工具执行结果（通过 tool_call_id 和 AI 的 tool_calls 关联）",
    },
    "memory存储的消息列表": history,
}

output_path = Path(__file__).parent.parent / "logs" / "memory_history.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"已保存到 logs/memory_history.json，共 {len(history)} 条消息")
