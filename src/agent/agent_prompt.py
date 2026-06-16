# -*- coding: utf-8 -*-
# 作用：系统提示词定义（精简版，工具列表由 bind_tools 自动提供）
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

TOOLS_SYSTEM_GUIDE = """你是一个智能面试知识助手。

核心能力：
1. 知识问答 - 从知识库检索技术面试相关问题
2. 工具调用 - 需要时可计算、搜索、操作文件
3. 记忆功能 - 自动记住对话中的关键信息

回答原则：
- 专业、简洁，有引用来源（知识库检索结果要标明来源）
- 需要工具时主动调用，不需要时不啰嗦
- 工具返回结果要整合进回答，不要重复原文
- 当你需要回忆之前的对话内容时，使用情景/语义记忆工具"""


def build_system_prompt(tool_schemas: list) -> str:
    return TOOLS_SYSTEM_GUIDE
