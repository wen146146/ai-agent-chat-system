# -*- coding: utf-8 -*-
"""
工具调用审计日志模块。

职责：
  - 记录每次工具调用的时间、工具名、参数（脱敏）、耗时、结果状态
  - 写入 agent_audit.log 文件（按天轮转）

使用方式：
  from src.utils.audit import log_tool_call

  log_tool_call("read_file", {"path": "./data.txt"}, "success", 120)
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler

# ============================================================================
# 审计日志配置
# ============================================================================

AUDIT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
AUDIT_LOG_FILE = os.path.join(AUDIT_LOG_DIR, "agent_audit.log")

os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

# 创建独立的审计日志 logger
audit_logger = logging.getLogger("agent_audit")
audit_logger.setLevel(logging.INFO)

# 避免重复添加 handler
if not audit_logger.handlers:
    _handler = RotatingFileHandler(
        AUDIT_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s"
    ))
    audit_logger.addHandler(_handler)


# ============================================================================
# 参数脱敏
# ============================================================================

SENSITIVE_KEYWORDS = [
    "password", "passwd", "pwd",
    "token", "api_key", "api-key", "apikey",
    "secret", "secret_key", "secret-key",
    "authorization", "auth", "credential",
    "private_key", "private-key",
    "access_key", "access-key",
]


def _sanitize_args(args: dict) -> dict:
    """
    工具参数脱敏：对包含敏感词的字段自动打码。

    使用模糊匹配而非精确匹配，覆盖 snake_case / kebab-case / 大小写变体：
      - "ApiKey" → ***
      - "API-KEY" → ***
      - "my_secret_password" → ***
    """
    safe = {}
    for k, v in args.items():
        k_normalized = k.lower().replace("-", "_").replace(" ", "_")
        is_sensitive = any(s in k_normalized for s in SENSITIVE_KEYWORDS)
        safe[k] = "***" if is_sensitive else v
    return safe


# ============================================================================
# 日志记录
# ============================================================================

def log_tool_call(tool_name: str, args: dict, status: str, duration_ms: int):
    """
    记录一次工具调用到审计日志。

    参数:
      tool_name: 工具名，如 "read_file"
      args: 工具参数（自动脱敏）
      status: 结果状态 "success" / "error" / "blocked"
      duration_ms: 执行耗时（毫秒）
    """
    safe_args = _sanitize_args(args)
    record = json.dumps({
        "tool": tool_name,
        "args": safe_args,
        "status": status,
        "duration_ms": duration_ms,
    }, ensure_ascii=False)
    audit_logger.info(record)
