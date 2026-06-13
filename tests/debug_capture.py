﻿# -*- coding: utf-8 -*-
import os, sys, json, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from src.agent.agent_chain import stream_chat, create_default_session, _model

# 拦截 stream 调用
_original_stream = _model.__class__.stream
captured_llm_messages = []
captured_llm_chunks = []

def _intercept_stream(self, *args, **kwargs):
    messages = args[0] if args else kwargs.get("input", [])
    if isinstance(messages, dict):
        messages = [messages]
    captured_llm_messages.append({
        "call_index": len(captured_llm_messages) + 1,
        "messages": [
            {
                "role": type(msg).__name__,
                "content": getattr(msg, "content", str(msg)),
                **({"tool_calls": getattr(msg, "tool_calls", None)} if hasattr(msg, "tool_calls") and msg.tool_calls else {}),
                **({"tool_call_id": getattr(msg, "tool_call_id", None)} if hasattr(msg, "tool_call_id") else {}),
            }
            for msg in messages
        ] if isinstance(messages, list) else [],
    })
    for chunk in _original_stream(self, *args, **kwargs):
        tc_list = getattr(chunk, "tool_calls", []) or []
        captured_llm_chunks.append({
            "call_index": len(captured_llm_messages),
            "content": getattr(chunk, "content", ""),
            "tool_calls": [{"name": tc.get("name",""), "args": tc.get("args",{}), "id": tc.get("id","")} for tc in tc_list],
        })
        yield chunk

_model.__class__.stream = _intercept_stream

# 模拟对话
session_id = create_default_session()
events = list(stream_chat("1+1等于几？用计算器工具算一下", session_id))

# 保存到文件
output = {"sse_events": events, "llm_requests": captured_llm_messages, "llm_chunks": captured_llm_chunks}
output_path = Path(__file__).parent.parent / "logs" / "captured_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"OK - saved {len(events)} events, {len(captured_llm_messages)} LLM calls to {output_path}")
