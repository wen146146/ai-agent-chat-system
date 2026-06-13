import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "fojiao_db")


@dataclass
class MySQLChunk:
    chunk_id: str
    doc_id: int
    kb_id: Optional[int]
    chunk_index: int
    content: str
    filename: str = ""
    file_type: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "kb_id": self.kb_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "filename": self.filename,
            "file_type": self.file_type,
        }


class ChunkStore:
    """MySQL 分块存储管理，支持关键词检索"""

    def __init__(self):
        self._connection = None
        self._table = "document_chunks"

    def _get_connection(self):
        if self._connection is None or not self._connection.open:
            import pymysql
            self._connection = pymysql.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
        return self._connection

    def _ensure_table(self):
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{self._table}` (
                    `chunk_id` VARCHAR(32) NOT NULL,
                    `doc_id` INT NOT NULL,
                    `kb_id` INT DEFAULT NULL,
                    `chunk_index` INT NOT NULL DEFAULT 0,
                    `content` LONGTEXT NOT NULL,
                    `filename` VARCHAR(512) DEFAULT '',
                    `file_type` VARCHAR(64) DEFAULT '',
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (`chunk_id`),
                    INDEX `idx_doc_id` (`doc_id`),
                    INDEX `idx_kb_id` (`kb_id`),
                    FULLTEXT INDEX `idx_content` (`content`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()

    def save_chunks(self, chunks: List[Any], verbose: bool = False) -> int:
        self._ensure_table()
        conn = self._get_connection()
        saved = 0
        errors = []
        with conn.cursor() as cursor:
            for chunk in chunks:
                try:
                    if hasattr(chunk, "to_dict"):
                        data = chunk.to_dict()
                    elif isinstance(chunk, dict):
                        data = chunk
                    else:
                        continue
                    cursor.execute(f"""
                        INSERT INTO `{self._table}`
                        (chunk_id, doc_id, kb_id, chunk_index, content, filename, file_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            content = VALUES(content),
                            kb_id = VALUES(kb_id),
                            chunk_index = VALUES(chunk_index)
                    """, (
                        data.get("chunk_id", ""),
                        data.get("doc_id", 0),
                        data.get("kb_id"),
                        data.get("chunk_index", 0),
                        data.get("content", ""),
                        data.get("filename", ""),
                        data.get("file_type", ""),
                    ))
                    saved += 1
                except Exception as e:
                    errors.append(f"chunk_id={data.get('chunk_id','?')}: {e}")
        conn.commit()
        if verbose and errors:
            print(f"      MySQL写入警告: {'; '.join(errors[:3])}")
        return saved

    def delete_chunks_by_document(self, doc_id: int) -> int:
        self._ensure_table()
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM `{self._table}` WHERE doc_id = %s", (doc_id,))
            deleted = cursor.rowcount
        conn.commit()
        return deleted

    def delete_chunks_by_knowledge_base(self, kb_id: int) -> int:
        self._ensure_table()
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM `{self._table}` WHERE kb_id = %s", (kb_id,))
            deleted = cursor.rowcount
        conn.commit()
        return deleted

    def keyword_search(
            self,
            keyword: str,
            kb_ids: Optional[List[int]] = None,
            limit: int = 10
    ) -> List[dict]:
        """LIKE 模糊匹配检索"""
        self._ensure_table()
        conn = self._get_connection()
        with conn.cursor() as cursor:
            sql = f"SELECT * FROM `{self._table}` WHERE content LIKE %s"
            params = [f"%{keyword}%"]
            if kb_ids:
                placeholders = ",".join(["%s"] * len(kb_ids))
                sql += f" AND kb_id IN ({placeholders})"
                params.extend(kb_ids)
            sql += " ORDER BY chunk_index LIMIT %s"
            params.append(limit)
            cursor.execute(sql, params)
            return cursor.fetchall()

    def exact_match_search(
            self,
            keywords: List[str],
            synonyms: Optional[Dict[str, List[str]]] = None,
            kb_ids: Optional[List[int]] = None,
            limit: int = 10
    ) -> List[dict]:
        """带同义词的精确匹配检索"""
        self._ensure_table()
        conn = self._get_connection()
        all_results = []
        seen_ids = set()

        search_terms = list(keywords)
        if synonyms:
            for kw, syns in synonyms.items():
                search_terms.extend(syns)

        with conn.cursor() as cursor:
            for term in search_terms:
                sql = f"SELECT * FROM `{self._table}` WHERE content LIKE %s"
                params = [f"%{term}%"]
                if kb_ids:
                    placeholders = ",".join(["%s"] * len(kb_ids))
                    sql += f" AND kb_id IN ({placeholders})"
                    params.extend(kb_ids)
                sql += " LIMIT %s"
                params.append(max(limit // len(search_terms), 1))
                try:
                    cursor.execute(sql, params)
                    for row in cursor.fetchall():
                        chunk_id = row.get("chunk_id")
                        if chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            row["score"] = 0.9
                            row["source"] = "mysql"
                            all_results.append(row)
                except Exception:
                    pass

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:limit]

    def count_chunks(self, kb_id: Optional[int] = None) -> int:
        self._ensure_table()
        conn = self._get_connection()
        with conn.cursor() as cursor:
            if kb_id is not None:
                cursor.execute(
                    f"SELECT COUNT(*) as cnt FROM `{self._table}` WHERE kb_id = %s",
                    (kb_id,)
                )
            else:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM `{self._table}`")
            result = cursor.fetchone()
            return result["cnt"] if result else 0
