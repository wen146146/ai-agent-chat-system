# -*- coding: utf-8 -*-
"""
Shell 命令执行工具 — 三层安全防护。

安全设计（多层防护）：
  第一层：命令白名单（SAFE_COMMANDS）— 只允许执行白名单中的命令
  第二层：危险命令黑名单（BLOCKED_COMMANDS）— 路径匹配到黑名单也拦截
  第三层：python/pip 特殊处理 — python -c 和 pip install 被拦截

Windows 专用：
  - 使用 shell=True 执行（cmd 原生支持 dir, type 等内部命令）
  - 通过 cwd 参数限定工作目录
  - 默认超时 15 秒防挂起
"""

import os
import subprocess
import shlex
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.utils.config_loader import resolve_safe_path


# ============================================================================
# 第一层：命令白名单
# ============================================================================

SAFE_COMMANDS: set = {
    # --- 信息查询（只读，无副操作） ---
    "dir", "type", "find", "echo",
    "where", "which",
    "whoami", "systeminfo", "netstat", "tasklist",
    "ipconfig", "ping", "tracert", "route",
    # --- 开发工具 ---
    "npm", "git",
    # --- 网络工具（只读用途） ---
    "curl", "wget",
}

# python3 是 python 的别名，同等对待（走 _check_python_invocation 安全校验）
# 注意：ssh / scp / rsync 不在白名单中——它们是数据外泄通道


# ============================================================================
# 第二层：危险命令黑名单
# ============================================================================

BLOCKED_COMMANDS: set = {
    # 删除/格式化
    "del", "rd", "rm", "rmdir", "rmtree", "format",
    # 系统修改
    "chkdsk", "attrib", "takeown", "icacls", "cacls",
    "shutdown", "reboot", "restart", "stop",
    # 进程/服务管理
    "taskkill", "kill", "stop-process", "killall",
    # 注册表/系统配置
    "reg", "regedit", "regedt32", "sc", "wmic",
    "gpedit", "secpol",
    # 权限提升
    "runas", "sudo",
}


# ============================================================================
# 第三层：python/pip 特殊处理
# ============================================================================

def _check_python_invocation(args: List[str]) -> bool:
    """
    python / python3 的调用安全检查。

    允许：
      - python --version / -V
      - python script.py          （脚本文件必须在项目目录内）
      - python -m module_name     （标准模块）

    拦截：
      - python -c "恶意代码"       （任意代码执行）
      - python script.py 不在项目内

    返回 True=安全，False=拦截
    """
    if not args:
        return False

    # python --version / -V 等无参数选项
    if args[0].startswith("--"):
        return True

    # python -c "..." → 拦截（任意代码执行）
    if args[0] == "-c":
        return False

    # python -m module → 允许（标准模块调用）
    if args[0] == "-m":
        return True

    # python script.py → 检查脚本路径是否在项目内
    try:
        script_path = resolve_safe_path(args[0])
        return True
    except PermissionError:
        return False


def _check_pip_invocation(args: List[str]) -> bool:
    """
    pip 的调用安全检查。

    允许：
      - pip list
      - pip freeze
      - pip show <package>

    拦截：
      - pip install <package>     （安装任意包，可能含恶意代码）
      - pip uninstall <package>   （卸载系统包）
      - pip --help 等不涉及安装/卸载的操作

    返回 True=安全，False=拦截
    """
    if not args:
        return True  # 纯 pip，显示帮助

    action = args[0].lstrip("-").lower()

    # 安装和卸载 → 拦截
    if action in ("install", "uninstall", "remove", "delete"):
        return False

    # 其余操作（list, freeze, show, check, config, cache 等）→ 允许
    return True


# ============================================================================
# 安全检查总入口
# ============================================================================

def _validate_command(command_str: str) -> str:
    """
    对整个命令字符串做三层安全检查。

    返回：
      - 校验通过 → 返回原始命令字符串
      - 校验失败 → 抛出 ValueError

    处理流程：
      1. 取命令基名
      2. 检查黑名单（BLOCKED_COMMANDS）
      3. 检查白名单（SAFE_COMMANDS）
      4. python/pip 特殊安全检查
    """
    # 拆分命令：取第一个词作为基名
    parts = shlex.split(command_str)
    if not parts:
        raise ValueError("命令为空")

    base_name = os.path.basename(parts[0]).lower().replace(".exe", "")

    # --- 检查黑名单 ---
    if base_name in BLOCKED_COMMANDS:
        raise ValueError(f"命令 `{base_name}` 在黑名单中，已拦截")

    # --- 检查白名单 ---
    if base_name in SAFE_COMMANDS:
        return command_str

    # --- python / python3 特殊处理 ---
    if base_name in ("python", "python3"):
        if _check_python_invocation(parts[1:]):
            return command_str
        raise ValueError(
            f"python 调用被安全策略拦截: `{command_str}`\n"
            f"  - python -c '...' 不允许（防止任意代码执行）\n"
            f"  - python 脚本必须在项目目录内"
        )

    # --- pip 特殊处理 ---
    if base_name == "pip":
        if _check_pip_invocation(parts[1:]):
            return command_str
        raise ValueError(
            f"pip install/uninstall 被安全策略拦截\n"
            f"  - 不允许通过工具安装或卸载包（防止供应链攻击）\n"
            f"  - pip list / pip freeze 等只读操作允许"
        )

    # --- 未在白名单中 ---
    raise ValueError(
        f"命令 `{base_name}` 不在白名单中\n"
        f"  允许的命令: {', '.join(sorted(SAFE_COMMANDS))}\n"
        f"  如需添加请修改 SAFE_COMMANDS"
    )


# ============================================================================
# 参数 Schema
# ============================================================================

class RunCommandInput(BaseModel):
    """执行 Shell 命令"""
    command: str = Field(description="要执行的命令，如 'dir src/' 或 'git status'")
    cwd: str = Field(default=".", description="工作目录，默认项目根目录")
    timeout: int = Field(default=15, description="超时秒数，默认 15 秒，最大 60 秒")


# ============================================================================
# 工具函数
# ============================================================================

@tool(args_schema=RunCommandInput)
def run_command(command: str, cwd: str = ".", timeout: int = 15) -> str:
    """执行 Shell 命令。当用户说"运行个命令"、"帮我执行"、"查一下系统配置"时调用。只允许白名单命令（dir/git/npm/ipconfig等），危险命令和任意代码执行被拦截。适合：查看目录、运行git、查网络状态等。"""
    try:
        # --- 安全检查 ---
        _validate_command(command)

        # --- 限定工作目录 ---
        safe_cwd = resolve_safe_path(cwd)
        timeout = min(timeout, 60)

        # --- 执行（Windows 用 shell=True） ---
        result = subprocess.run(
            command,
            shell=True,
            cwd=safe_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # --- 处理输出 ---
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout.strip()[:3000])
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.strip()[:1000]}")

        output = "\n\n".join(output_parts) if output_parts else "(无输出)"

        exit_code = result.returncode
        status = "✅" if exit_code == 0 else "⚠️"
        return f"{status} 命令执行完成 (exit code: {exit_code})\n\n{output}"

    except PermissionError as e:
        return f"[安全拦截] {e}"
    except ValueError as e:
        return f"[安全拦截] {e}"
    except subprocess.TimeoutExpired:
        return f"[错误] 命令执行超时（{timeout} 秒）"
    except subprocess.SubprocessError as e:
        return f"[错误] 命令执行失败: {e}"
    except Exception as e:
        return f"[错误] 执行异常: {e}"
