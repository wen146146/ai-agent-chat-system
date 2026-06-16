# -*- coding: utf-8 -*-
"""
配置文件加载器 — 集中管理所有环境变量读取。

职责：
  1. 统一 API Key 读取入口（兼容新旧命名，一处修改全项目生效）
  2. 提供路径安全校验工具函数（供 file_tools / shell_tools 复用）
  3. 提供启动时 Key 有效性检查

使用方式：
  from src.utils.config_loader import get_llm_key, validate_path

  api_key = get_llm_key()
  safe_path = validate_path("../some/path")
"""

import os
import logging

logger = logging.getLogger("agent.config")


# ============================================================================
# API Key 读取（兼容新旧命名，不改现有模块引用）
# ============================================================================
# 各模块现状：
#   agent_chain.py          → os.getenv("OPENAI_API_KEY")
#   rag_retriever_tool.py   → os.getenv("OPENAI_API_KEY")
#   vector_store.py         → os.getenv("DASHSCOPE_API_KEY")
#   semantic_memory_tool.py → os.getenv("DASHSCOPE_API_KEY")
#
# 如果未来需要改 Key 名，只需修改这里，无需改 4 个模块。

def get_llm_key() -> str:
    """
    获取 LLM API Key。
    优先读 OPENAI_API_KEY（向后兼容），其次读 LLM_API_KEY。
    """
    return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""


def get_llm_base_url() -> str:
    """获取 LLM API Base URL。"""
    return os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1"


def get_llm_model() -> str:
    """获取 LLM 模型名。"""
    return os.getenv("LLM_MODEL") or "deepseek-v4-flash"


def get_embedding_key() -> str:
    """获取嵌入模型 API Key（DashScope）。"""
    return os.getenv("DASHSCOPE_API_KEY") or ""


def get_embedding_model() -> str:
    """获取嵌入模型名。"""
    return os.getenv("DASHSCOPE_EMBEDDING_MODEL") or "text-embedding-v3"


# ============================================================================
# 启动时 Key 有效性检查
# ============================================================================

REQUIRED_KEYS = {
    "OPENAI_API_KEY": "LLM 对话（DeepSeek）",
    "DASHSCOPE_API_KEY": "文本嵌入（DashScope）",
}

OPTIONAL_KEYS = {
    "QDRANT_HOST": "向量数据库",
    "MYSQL_HOST": "MySQL 数据库",
}


def validate_keys_at_startup() -> list:
    """
    启动时检查环境变量，返回警告信息列表。
    不抛出异常，仅打印警告——允许在 Key 缺失时启动（部分功能降级）。
    """
    warnings = []

    for key, label in REQUIRED_KEYS.items():
        value = os.getenv(key)
        if not value:
            warnings.append(f"[WARN] 环境变量 {key} 未设置 —— {label} 功能不可用")
        elif value.startswith("sk-") and len(value) < 20:
            warnings.append(f"[WARN] {key} 格式异常（长度过短）—— {label} 可能无法正常工作")
        elif value.startswith("sk-") and len(value) > 60:
            warnings.append(f"[WARN] {key} 长度异常（过长）—— {label} 可能配置错误")

    for key, label in OPTIONAL_KEYS.items():
        if not os.getenv(key):
            warnings.append(f"[INFO] 环境变量 {key} 未设置 —— 将使用默认值（{label}）")

    return warnings


# ============================================================================
# 路径安全校验（供 file_tools / shell_tools 复用）
# ============================================================================

# 文件写入白名单目录
ALLOWED_WRITE_DIRS = [
    os.path.abspath("."),            # 项目根目录
    os.path.abspath("./data"),        # 数据目录
    os.path.abspath("./output"),      # 输出目录
    os.path.abspath("./static"),      # 静态文件目录
]


def resolve_safe_path(path: str) -> str:
    """
    规范化路径 + 防路径穿越。

    处理流程：
      1. os.path.realpath() 解析所有 .. 和符号链接
      2. 检查是否在项目根目录内
      3. 越权则抛出 PermissionError

    参数:
      path: 用户传入的路径字符串

    返回:
      规范化后的绝对路径

    异常:
      PermissionError: 路径越权时抛出

    示例:
      >>> resolve_safe_path("../etc/passwd")
      PermissionError: 路径越权: C:\etc\passwd

      >>> resolve_safe_path("./data/file.txt")
      "C:\\project\\data\\file.txt"
    """
    base = os.path.realpath(os.path.abspath("."))
    target = os.path.realpath(os.path.abspath(path))

    if not target.startswith(base):
        raise PermissionError(f"路径越权: {path} → {target}，不允许超出项目根目录: {base}")

    return target


def validate_write_path(filepath: str) -> str:
    """
    写入路径安全校验（第二道防线，即使 user_confirm 也拦截）。

    比 resolve_safe_path 更严格：
      1. 先做路径规范化 + 防穿越
      2. 额外检查是否在白名单子目录内

    参数:
      filepath: 用户传入的文件路径

    返回:
      规范化后的绝对路径

    异常:
      PermissionError: 路径不在白名单内或越权

    示例:
      >>> validate_write_path("../../Windows/system32/drivers/etc/hosts")
      PermissionError: 写入路径不在白名单内

      >>> validate_write_path("./data/notes.txt")
      "C:\\project\\data\\notes.txt"
    """
    abs_path = resolve_safe_path(filepath)

    for allowed in ALLOWED_WRITE_DIRS:
        if abs_path.startswith(allowed):
            return abs_path

    raise PermissionError(f"写入路径不在白名单内: {abs_path}（允许: {ALLOWED_WRITE_DIRS}）")
