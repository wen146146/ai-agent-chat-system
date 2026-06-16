# -*- coding: utf-8 -*-
"""
应用控制工具集：打开应用程序、列出可打开的应用。

安全设计：
  - 只允许打开预定义映射表中的应用程序
  - 不允许运行任意路径的可执行文件
  - 每个应用映射包含名称和路径，LLM 只传名称
"""

import os
import subprocess
from typing import Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============================================================================
# 内置应用映射表
# ============================================================================
# 键：LLM 可用的应用名（不区分大小写）
# 值：(显示名, 可执行文件路径, 启动参数前缀)

APP_MAP: Dict[str, tuple] = {
    "notepad":    ("记事本", "notepad.exe", ""),
    "calc":       ("计算器", "calc.exe", ""),
    "calculator": ("计算器", "calc.exe", ""),
    "chrome":     ("Google Chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe", ""),
    "edge":       ("Microsoft Edge", "msedge.exe", ""),
    "explorer":   ("文件资源管理器", "explorer.exe", ""),
    "cmd":        ("命令提示符", "cmd.exe", ""),
    "terminal":   ("Windows Terminal", "wt.exe", ""),
    "code":       ("VS Code", "code.cmd", ""),
}


# ============================================================================
# 参数 Schema
# ============================================================================

class OpenAppInput(BaseModel):
    """打开应用程序"""
    name: str = Field(description="应用名称（不区分大小写），如 'notepad', 'calc', 'chrome'")
    args: str = Field(default="", description="启动参数，如文件路径、URL 等（可选）")


class ListAppsInput(BaseModel):
    """列出可打开的应用程序"""
    pass


# ============================================================================
# 工具函数
# ============================================================================

@tool(args_schema=OpenAppInput)
def open_application(name: str, args: str = "") -> str:
    """打开本地应用程序。当用户说"打开记事本"、"启动计算器"、"打开Chrome"时调用。内置应用映射表，支持传参（如打开特定文件或网址）。目前支持：notepad、calc、chrome、edge、explorer、cmd、terminal、code。"""
    try:
        name_lower = name.strip().lower()

        if name_lower not in APP_MAP:
            available = ", ".join(sorted(APP_MAP.keys()))
            return f"[错误] 不支持的应用: `{name}`\n支持的应用: {available}"

        display_name, executable, prefix = APP_MAP[name_lower]
        full_args = executable

        if args:
            full_args = f"{full_args} {args}"
        if prefix:
            full_args = f"{prefix} {full_args}"

        # 使用 subprocess.Popen 异步启动（不阻塞工具返回）
        subprocess.Popen(
            full_args,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return f"✅ 已启动 {display_name}"

    except FileNotFoundError:
        return f"[错误] 应用 `{name}` 的可执行文件未找到，可能未安装"
    except Exception as e:
        return f"[错误] 启动失败: {e}"


@tool(args_schema=ListAppsInput)
def list_applications() -> str:
    """列出所有可以通过 open_application 打开的应用程序。当用户问"你能打开什么应用"、"有哪些应用可以启动"时调用。返回应用名称和对应的可执行文件路径。"""
    lines = ["📋 可打开的应用程序："]
    for name in sorted(APP_MAP.keys()):
        display_name, executable, _ = APP_MAP[name]
        lines.append(f"  - `{name}` → {display_name} ({executable})")

    # 去重：同名应用只显示一次
    seen = set()
    unique = []
    for line in lines:
        key = line.split("→")[-1].strip() if "→" in line else line
        if key not in seen:
            seen.add(key)
            unique.append(line)

    return "\n".join(unique)
