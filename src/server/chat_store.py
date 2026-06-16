# -*- coding: utf-8 -*-
"""
对话历史 MySQL 存储。

复用 MYSQL_* 配置（与 episodic_memories 同库 ai_agent_db）。

表结构：
  conversation_history
    id          AUTO_INCREMENT 主键（按此排序即消息顺序）
    session_id  会话 ID（8 位 UUID）
    role        'user' / 'assistant'
    content     消息内容
    created_at  创建时间

使用方式：
  from src.server.chat_store import ChatStore
  store = ChatStore()
  store.save_message("abc123", "user", "你好")
  history = store.get_history("abc123")
  sessions = store.list_sessions()
  store.delete_session("abc123")
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ============================================================================
# 数据库配置（复用 MYSQL_*，与 episodic_memory 同库）
# ============================================================================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "ai_agent_db")


def _get_connection():
    import pymysql
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


TABLE_NAME = "conversation_history"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(16) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def ensure_table():
    """确保 conversation_history 表存在（幂等）。"""
    conn = _get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
    except Exception as e:
        print(f"[WARN] conversation_history 建表失败: {e}")
    finally:
        conn.close()


class ChatStore:
    """对话历史存储。"""

    def __init__(self):
        self._table = TABLE_NAME

    def _ensure(self):
        """确保连接和表就绪。"""
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def save_message(self, session_id: str, role: str, content: str):
        """
        保存一条对话消息。
        在 stream_chat 的 final_output 后调用。
        """
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {self._table} (session_id, role, content) VALUES (%s, %s, %s)",
                    (session_id, role, content),
                )
            conn.commit()
        except Exception as e:
            print(f"[WARN] 保存对话历史失败: {e}")
        finally:
            conn.close()

    def get_history(self, session_id: str) -> List[Dict]:
        """
        获取指定会话的完整对话历史。
        按 id ASC 保证消息顺序。
        """
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT id, session_id, role, content, created_at FROM {self._table} "
                    f"WHERE session_id = %s ORDER BY id ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    if isinstance(row.get("created_at"), datetime):
                        row["created_at"] = row["created_at"].isoformat()
                return rows
        except Exception:
            return []
        finally:
            conn.close()

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """
        列出所有历史会话。
        按最后消息时间倒序。

        返回: [{"session_id": "...", "message_count": N, "last_time": "...", "preview": "..."}, ...]
        """
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        session_id,
                        COUNT(*) as message_count,
                        MAX(created_at) as last_time,
                        SUBSTRING_INDEX(GROUP_CONCAT(content ORDER BY id DESC SEPARATOR '\n'), '\n', 1) as preview
                    FROM {self._table}
                    GROUP BY session_id
                    ORDER BY last_time DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    if isinstance(row.get("last_time"), datetime):
                        row["last_time"] = row["last_time"].isoformat()
                    # 截断 preview
                    if row.get("preview"):
                        row["preview"] = row["preview"][:80]
                return rows
        except Exception:
            return []
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话的全部消息。
        返回 True 表示有记录被删除。
        """
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._table} WHERE session_id = %s",
                    (session_id,),
                )
                deleted = cursor.rowcount
            conn.commit()
            return deleted > 0
        except Exception:
            return False
        finally:
            conn.close()

    def count_messages(self, session_id: str) -> int:
        """统计指定会话的消息数。"""
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) as cnt FROM {self._table} WHERE session_id = %s",
                    (session_id,),
                )
                return cursor.fetchone()["cnt"]
        except Exception:
            return 0
        finally:
            conn.close()
