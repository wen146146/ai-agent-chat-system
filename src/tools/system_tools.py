# -*- coding: utf-8 -*-
"""
系统信息工具集：CPU/内存/磁盘/OS 信息、进程列表。

依赖：
  - psutil（已添加到 requirements.txt）

安全设计：
  - 所有操作均为只读
  - 无需用户确认（auto 权限级别）
  - 不修改任何系统状态
"""

import platform
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SystemInfoInput(BaseModel):
    """获取系统信息"""
    pass


class ProcessListInput(BaseModel):
    """获取进程列表"""
    top_n: int = Field(default=10, ge=1, le=50, description="返回进程数量，默认 10 条，最多 50 条")


@tool(args_schema=SystemInfoInput)
def get_system_info() -> str:
    """
    获取系统信息：CPU 使用率、内存占用、磁盘空间、操作系统版本。
    只读操作，无需确认。
    """
    try:
        import psutil
    except ImportError:
        return ("[错误] 缺少 psutil 库，请运行: pip install psutil\n"
                "本工具依赖 psutil 获取系统信息。")

    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)

        # 内存
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / 1024**3
        mem_used_gb = mem.used / 1024**3
        mem_percent = mem.percent

        # 磁盘
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / 1024**3
        disk_used_gb = disk.used / 1024**3
        disk_percent = disk.percent

        # OS
        os_name = platform.system()
        os_version = platform.version()
        os_release = platform.release()
        node_name = platform.node()

        lines = [
            "🖥️ **系统信息**",
            f"  主机名: {node_name}",
            f"  操作系统: {os_name} {os_release} (build {os_version})",
            f"  Python: {platform.python_version()}",
            "",
            "🟦 **CPU**",
            f"  使用率: {cpu_percent}%",
            f"  逻辑核心: {cpu_count} | 物理核心: {cpu_physical}",
            "",
            "🧠 **内存**",
            f"  已用: {mem_used_gb:.1f} GB / {mem_total_gb:.1f} GB ({mem_percent}%)",
            "",
            "💾 **磁盘 (C:\\)**",
            f"  已用: {disk_used_gb:.1f} GB / {disk_total_gb:.1f} GB ({disk_percent}%)",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"[错误] 获取系统信息失败: {e}"


@tool(args_schema=ProcessListInput)
def get_process_list(top_n: int = 10) -> str:
    """
    获取当前进程列表，按 CPU 占用率降序排列。
    只读操作，无需确认。
    """
    try:
        import psutil
    except ImportError:
        return "[错误] 缺少 psutil 库，请运行: pip install psutil"

    try:
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = proc.info
                processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 按 CPU 占用排序
        processes.sort(key=lambda p: p.get("cpu_percent", 0) or 0, reverse=True)
        top = processes[:top_n]

        lines = [f"📋 进程列表 (Top {top_n}，按 CPU 占用排序):"]
        lines.append(f"{'PID':>7} {'CPU%':>5} {'MEM%':>6} {'状态':>8}   {'名称'}")
        lines.append("-" * 60)

        for p in top:
            pid = p.get("pid", "?")
            cpu = p.get("cpu_percent") or 0
            mem = p.get("memory_percent") or 0
            status = (p.get("status") or "?").upper()[:6]
            name = (p.get("name") or "?").strip()[:30]
            lines.append(f"{pid:>7} {cpu:>5.1f} {mem:>5.1f}% {status:>8}   {name}")

        return "\n".join(lines)

    except Exception as e:
        return f"[错误] 获取进程列表失败: {e}"
