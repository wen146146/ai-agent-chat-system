# -*- coding: utf-8 -*-
"""
文件系统工具集：读/写/搜索/列目录。

安全设计：
  - read_file:   路径规范化 + 大小限制（100KB，可配）
  - write_file:  路径白名单 + 路径穿越防护（validate_write_path）
  - search_files: glob 模式搜索，只搜文件名
  - list_directory: 列表目录内容（非递归）
"""

import os
import glob as glob_module
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.utils.config_loader import resolve_safe_path, validate_write_path


# ============================================================================
# 参数 Schema
# ============================================================================

class ReadFileInput(BaseModel):
    """读取文件内容"""
    path: str = Field(description="文件路径（相对或绝对路径）")
    encoding: str = Field(default="utf-8", description="文件编码，默认 utf-8")
    max_bytes: int = Field(default=102400, description="最大读取字节数，默认 100KB，最大 1MB")


class WriteFileInput(BaseModel):
    """写入/追加文件内容"""
    path: str = Field(description="文件路径（必须在项目白名单目录内）")
    content: str = Field(description="要写入的文件内容")
    mode: Literal["overwrite", "append"] = Field(default="overwrite", description="写入模式：overwrite（覆盖）/ append（追加）")


class SearchFilesInput(BaseModel):
    """按模式搜索文件"""
    pattern: str = Field(description="搜索模式，如 *.py 匹配所有Python文件，**/*.md 递归匹配Markdown")
    root_dir: str = Field(default=".", description="搜索根目录，默认当前目录")


class ListDirInput(BaseModel):
    """列出目录内容"""
    path: str = Field(default=".", description="目录路径，默认当前目录")
    show_hidden: bool = Field(default=False, description="是否显示隐藏文件（以 . 开头的文件）")


# ============================================================================
# 工具函数
# ============================================================================

@tool(args_schema=ReadFileInput)
def read_file(path: str, encoding: str = "utf-8", max_bytes: int = 102400) -> str:
    """读取文件内容。当用户说"帮我看看这个文件"、"读取某某文件"、"打开文件"时调用。支持所有文本文件格式，自动限制大小（默认100KB），路径自动规范化防穿越。"""
    try:
        safe_path = resolve_safe_path(path)

        if not os.path.isfile(safe_path):
            return f"[错误] 文件不存在或不是普通文件: {path}"

        file_size = os.path.getsize(safe_path)
        if file_size > max_bytes:
            return f"[错误] 文件过大 ({file_size:,} 字节)，最大允许读取 {max_bytes:,} 字节"

        with open(safe_path, "r", encoding=encoding) as f:
            content = f.read(max_bytes)

        return f"📄 {os.path.basename(safe_path)} ({file_size:,} 字节)\n\n{content}"

    except PermissionError as e:
        return f"[安全拦截] {e}"
    except UnicodeDecodeError:
        return f"[错误] 文件编码不是 {encoding}，请尝试其他编码（如 gbk / utf-16）"
    except Exception as e:
        return f"[错误] 读取失败: {e}"


@tool(args_schema=WriteFileInput)
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """写入或追加文件内容。当用户说"帮我写个文件"、"保存内容到"、"创建文件"时调用。路径必须在项目白名单目录内（./data, ./output, ./static），防止覆盖系统文件。"""
    try:
        safe_path = validate_write_path(path)

        # 创建父目录（如果不存在）
        parent_dir = os.path.dirname(safe_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        write_mode = "w" if mode == "overwrite" else "a"
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)

        action = "覆盖写入" if mode == "overwrite" else "追加写入"
        file_size = len(content.encode("utf-8"))
        return f"✅ {action}成功: {safe_path} ({file_size:,} 字节)"

    except PermissionError as e:
        return f"[安全拦截] {e}"
    except Exception as e:
        return f"[错误] 写入失败: {e}"


@tool(args_schema=SearchFilesInput)
def search_files(pattern: str, root_dir: str = ".") -> str:
    """按文件名模式搜索文件。当用户说"找一下某某文件"、"搜索文件"、"哪些.py文件"时调用。支持glob语法如 *.py、**/*.md，返回文件名、大小和修改时间。"""
    try:
        safe_root = resolve_safe_path(root_dir)

        if not os.path.isdir(safe_root):
            return f"[错误] 目录不存在: {root_dir}"

        matches = []
        for filepath in glob_module.glob(pattern, root_dir=safe_root, recursive=True):
            full_path = os.path.join(safe_root, filepath)
            if os.path.isfile(full_path):
                file_size = os.path.getsize(full_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M")
                matches.append(f"{filepath} ({file_size:,} 字节, {mtime})")

        if not matches:
            return f"未找到匹配 `{pattern}` 的文件（搜索目录: {safe_root}）"

        # 限制显示条数，避免刷屏
        MAX_DISPLAY = 50
        truncated = len(matches) > MAX_DISPLAY
        display = matches[:MAX_DISPLAY]

        result = f"🔍 找到 {len(matches)} 个匹配 `{pattern}` 的文件:\n"
        for i, line in enumerate(display, 1):
            result += f"\n{i}. {line}"
        if truncated:
            result += f"\n\n... 还有 {len(matches) - MAX_DISPLAY} 个文件未显示"
        return result

    except PermissionError as e:
        return f"[安全拦截] {e}"
    except Exception as e:
        return f"[错误] 搜索失败: {e}"


@tool(args_schema=ListDirInput)
def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """列出目录内容。当用户说"看看这个目录有什么"、"列出文件夹"、"目录结构"时调用。非递归，显示文件夹/文件、大小、修改时间。"""
    try:
        safe_path = resolve_safe_path(path)

        if not os.path.isdir(safe_path):
            return f"[错误] 目录不存在: {path}"

        entries = []
        dir_count = 0
        file_count = 0

        for name in sorted(os.listdir(safe_path)):
            # 隐藏文件过滤
            if not show_hidden and name.startswith("."):
                continue

            full = os.path.join(safe_path, name)
            is_dir = os.path.isdir(full)
            file_size = os.path.getsize(full) if not is_dir else 0
            mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
            icon = "📁" if is_dir else "📄"
            size_str = f"{file_size:,} 字节" if not is_dir else ""
            entries.append(f"{icon}  {name}    {size_str}    {mtime}")

            if is_dir:
                dir_count += 1
            else:
                file_count += 1

        if not entries:
            return f"📂 `{safe_path}` 为空目录"

        result = f"📂 `{safe_path}` （{dir_count} 个文件夹, {file_count} 个文件）:\n"
        for line in entries:
            result += f"\n{line}"
        return result

    except PermissionError as e:
        return f"[安全拦截] {e}"
    except Exception as e:
        return f"[错误] 列目录失败: {e}"
