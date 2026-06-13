# -*- coding: utf-8 -*-
# ============================================================================
# 情景记忆工具（Episodic Memory Tool）
# ============================================================================
# 作用：
#   记录用户与AI的每次交互事件（谁说了什么、什么时候说的），
#   支持后续按用户ID和关键词查询历史对话。
#
# 类比：
#   语义记忆 = 长期知识（"文浩喜欢吃辣"）
#   情景记忆 = 对话流水账（"2024-01-01 14:30 文浩问了天气 "）
#
# 注册的工具（3个）：
#   1. episodic_memory_save   — 保存一条交互记录
#   2. episodic_memory_search — 按用户+关键词搜索历史记录
#   3. episodic_memory_delete — 删除指定记录
#
# 存储方式：
#   MySQL 单表 episodic_memories，每次保存/搜索/删除都直接操作MySQL。
#   不使用向量数据库（情景记忆只做关键词匹配，不需要语义搜索）。
#
# 调用链路（以保存为例）：
#   LLM输出 [TOOL:episodic_memory_save]{"content":"user_message 你好"}
#       → agent_chain.py 解析到工具名
#       → 调用 episodic_memory_save.func(user_id="default", event_type="user_message", content="你好")
#       → _save_to_db() → INSERT INTO episodic_memories
#       → 返回 "[情景记忆] 已保存 #123 ..."
# ============================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中，方便跨模块导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# ============================================================================
# 数据库配置（从 .env 读取，有默认值兜底）
# ============================================================================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "ai_agent_db")


# ============================================================================
# 数据库底层操作（私有函数，不对外暴露）
# ============================================================================

def _get_connection():
    """
    获取MySQL数据库连接。
    每次调用都创建新连接（不用连接池，简单场景够用）。
    返回 DictCursor，查询结果直接是字典格式，方便取字段。
    """
    import pymysql
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor  # 查询结果返回 {列名: 值} 格式
    )


def _ensure_table():
    """
    确保 episodic_memories 表存在，不存在则自动创建。
    表结构说明：
      - id: 自增主键，每条记忆的唯一数字ID
      - user_id: 用户标识（如 "default"），支持多用户隔离
      - event_type: 事件类型枚举（user_message/ai_response/system_event/tool_call/conversation_summary）
      - content: 事件内容文本
      - metadata: 扩展字段（JSON格式），可存任意附加信息
      - created_at: 记录创建时间，自动填充
    索引设计：
      - idx_user: 按用户查询时加速
      - idx_event_type: 按事件类型筛选时加速
      - idx_created: 按时间排序时加速
      - FULLTEXT idx_content: 全文索引，支持中文关键词模糊搜索（LIKE %关键词%）
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(128) NOT NULL,
                    event_type VARCHAR(64) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSON DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id),
                    INDEX idx_event_type (event_type),
                    INDEX idx_created (created_at),
                    FULLTEXT idx_content (content)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


# 模块加载时自动建表，MySQL不可用则静默跳过（工具调用时再重试）
try:
    _ensure_table()
except Exception:
    pass


def _save_to_db(user_id: str, event_type: str, content: str, metadata: Dict[str, Any] = None) -> int:
    """
    保存一条情景记忆到MySQL。
    参数：
      - user_id: 用户标识（目前固定 "default"）
      - event_type: 事件类型，如 "user_message"
      - content: 事件内容文本
      - metadata: 扩展字段（字典格式），可存任意附加信息
    返回：
      - 成功返回自增ID，失败抛出异常
    """
    if metadata is None:
        metadata = {}
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO episodic_memories (user_id, event_type, content, metadata)
                   VALUES (%s, %s, %s, %s)""",
                (user_id, event_type, content, json.dumps(metadata, ensure_ascii=False))
            )
        conn.commit()
        return cur.lastrowid  # 返回新插入记录的自增ID
    finally:
        conn.close()


def _search_from_db(user_id: str, keyword: str = "", limit: int = 10) -> List[Dict]:
    """
    从MySQL搜索情景记忆。
    参数：
      - user_id: 必填，只查该用户的记录
      - keyword: 可选，对content字段做LIKE模糊匹配
      - limit: 返回条数上限，默认10
    返回：
      - 记录列表，每条包含 id/user_id/event_type/content/metadata/created_at
      - created_at 自动转为 ISO 格式字符串，方便JSON序列化
    排序：按创建时间倒序（最新的在前）
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if keyword:
                # 有关键词：按用户+内容模糊匹配，按时间倒序
                cur.execute(
                    """SELECT id, user_id, event_type, content, metadata, created_at
                       FROM episodic_memories
                       WHERE user_id = %s AND content LIKE %s
                       ORDER BY created_at DESC LIMIT %s""",
                    (user_id, f"%{keyword}%", limit)
                )
            else:
                # 无关键词：返回该用户最近N条记录
                cur.execute(
                    """SELECT id, user_id, event_type, content, metadata, created_at
                       FROM episodic_memories
                       WHERE user_id = %s
                       ORDER BY created_at DESC LIMIT %s""",
                    (user_id, limit)
                )
            rows = cur.fetchall()
            # datetime对象转字符串，方便前端JSON展示
            for row in rows:
                if isinstance(row.get("created_at"), datetime):
                    row["created_at"] = row["created_at"].isoformat()
            return rows
    finally:
        conn.close()


def _delete_from_db(memory_id: int) -> bool:
    """
    按主键ID删除一条情景记忆。
    返回 True=删除成功，False=ID不存在。
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM episodic_memories WHERE id = %s", (memory_id,))
        conn.commit()
        return cur.rowcount > 0  # rowcount>0 说明确实删掉了记录
    finally:
        conn.close()


# ============================================================================
# 工具1: episodic_memory_save — 保存情景记忆
# ============================================================================

class EpisodicSaveInput(BaseModel):
    """
    保存工具的参数Schema。
    LangChain用这个Pydantic模型校验LLM传过来的参数，
    字段不匹配或类型错误会直接拒绝调用。
    """
    user_id: str = Field(description="用户唯一标识")
    event_type: Literal[
        "user_message",         # 用户说的话
        "ai_response",          # AI的回复
        "system_event",         # 系统事件（如会话创建）
        "tool_call",            # 工具调用记录
        "conversation_summary"  # 对话摘要
    ] = Field(description="事件类型")
    content: str = Field(description="要存储的内容文本")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据，键值对格式")


@tool(args_schema=EpisodicSaveInput)
def episodic_memory_save(user_id: str, event_type: str, content: str, metadata: Dict[str, Any] = None) -> str:
    """
    工具1：保存情景记忆。
    把用户与AI的一次交互事件写入MySQL episodic_memories表。
    后续可通过 episodic_memory_search 按用户和关键词检索。

    使用场景举例：
      - 用户说"你好" → event_type="user_message", content="你好"
      - AI回复了 → event_type="ai_response", content="AI的回复内容"
      - 用户要求记住某事 → event_type="user_message", content="用户要记住的内容"
    """
    try:
        mem_id = _save_to_db(user_id, event_type, content, metadata)
        return f"[情景记忆] 已保存 #{mem_id} 类型:{event_type} 内容:{content[:50]}..."
    except Exception as e:
        return f"[情景记忆] 保存失败: {str(e)}"


# ============================================================================
# 工具2: episodic_memory_search — 搜索情景记忆
# ============================================================================

class EpisodicSearchInput(BaseModel):
    """搜索工具的参数Schema"""
    user_id: str = Field(description="用户唯一标识")
    keyword: str = Field(default="", description="搜索关键词，留空则返回最近记录")
    limit: int = Field(default=10, description="返回数量上限，默认10")


@tool(args_schema=EpisodicSearchInput)
def episodic_memory_search(user_id: str, keyword: str = "", limit: int = 10) -> str:
    """
    工具2：搜索情景记忆。
    按用户ID和可选关键词搜索历史交互记录。
    keyword为空时返回该用户最近10条记录；
    填了关键词则对content字段做 LIKE %关键词% 模糊匹配。

    使用场景举例：
      - "我之前问过什么？" → keyword="" 返回最近记录
      - "我上次问天气是什么时候？" → keyword="天气"
    """
    results = _search_from_db(user_id, keyword, limit)
    if not results:
        return "暂无匹配的情景记忆记录"
    # 格式化输出：省略长内容，只展示前80字
    lines = [f"[情景记忆] ({len(results)}条):"]
    for item in results:
        etype = item["event_type"]
        content = item["content"][:80]
        lines.append(f"  #{item['id']} [{etype}] {content}...")
    return "\n".join(lines)


# ============================================================================
# 工具3: episodic_memory_delete — 删除情景记忆
# ============================================================================

class EpisodicDeleteInput(BaseModel):
    """删除工具的参数Schema"""
    memory_id: int = Field(description="要删除的记忆记录ID")


@tool(args_schema=EpisodicDeleteInput)
def episodic_memory_delete(memory_id: int) -> str:
    """
    工具3：删除情景记忆。
    按主键ID从MySQL删除一条记录。
    返回删除结果（成功/记录不存在）。
    """
    ok = _delete_from_db(memory_id)
    return f"[情景记忆] #{memory_id} {'已删除' if ok else '删除失败(不存在)'}"
