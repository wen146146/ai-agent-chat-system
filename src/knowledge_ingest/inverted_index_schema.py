# -*- coding: utf-8 -*-
"""
倒排索引数据库表结构定义与创建。

两张表：
  1. document_terms — 倒排索引表：term → {chunk_id → term_freq}
  2. doc_stats      — 文档统计表：chunk_id → total_terms

使用方式：
  from src.knowledge_ingest.inverted_index_schema import ensure_inverted_index_tables
  ensure_inverted_index_tables()
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ============================================================================
# 数据库配置（复用 chunk_store 的同源配置）
# ============================================================================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "fojiao_db")


# ============================================================================
# DDL
# ============================================================================

INVERTED_INDEX_DDL = """
CREATE TABLE IF NOT EXISTS document_terms (
    term VARCHAR(128) NOT NULL COLLATE utf8mb4_bin,
    doc_id INT NOT NULL,
    chunk_id VARCHAR(32) NOT NULL,
    term_freq INT NOT NULL DEFAULT 1,
    kb_id INT DEFAULT NULL,
    PRIMARY KEY (term, chunk_id),
    INDEX idx_doc_id (doc_id),
    INDEX idx_kb_id (kb_id),
    INDEX idx_term_prefix (term(8))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
"""

DOC_STATS_DDL = """
CREATE TABLE IF NOT EXISTS doc_stats (
    chunk_id VARCHAR(32) PRIMARY KEY,
    doc_id INT NOT NULL,
    kb_id INT DEFAULT NULL,
    total_terms INT NOT NULL DEFAULT 0,
    INDEX idx_doc_id (doc_id),
    INDEX idx_kb_id (kb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def _get_connection():
    """获取 MySQL 连接（与 chunk_store 风格一致）。"""
    import pymysql
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_inverted_index_tables():
    """
    确保倒排索引相关表存在（幂等，重复调用安全）。
    应在服务启动时调用一次。
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(INVERTED_INDEX_DDL)
            cursor.execute(DOC_STATS_DDL)
        conn.commit()
        print("[INFO] 倒排索引表已就绪: document_terms, doc_stats")
    except Exception as e:
        print(f"[WARN] 倒排索引建表失败（不影响启动）: {e}")
    finally:
        conn.close()
