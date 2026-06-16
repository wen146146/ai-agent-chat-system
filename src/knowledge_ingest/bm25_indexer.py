# -*- coding: utf-8 -*-
"""
倒排索引构建器（Inverted Index Builder）。

职责：
  1. jieba 分词 + 清洗 + 去停用词
  2. 对 chunk 内容生成 term → term_freq 映射
  3. 写入 MySQL document_terms 倒排表 + doc_stats 统计表

使用方式：
  from src.knowledge_ingest.bm25_indexer import InvertedIndexBuilder

  builder = InvertedIndexBuilder()

  # 全量重建所有文档的索引
  builder.rebuild_all()

  # 增量更新（上传文档后）
  builder.add_chunks(chunks)

  # 删除文档时
  builder.remove_document(doc_id)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.knowledge_ingest.inverted_index_schema import _get_connection


# ============================================================================
# 停用词表（与 BM25 引擎保持一致，确保分词一致性）
# ============================================================================

STOP_WORDS: set = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "些", "什么", "怎么", "怎样", "哪", "哪里",
    "吗", "呢", "吧", "啊", "哦", "与", "或", "及", "等", "对",
    "把", "被", "让", "给", "为", "所", "以", "之", "其", "该", "此",
    "每个", "所有", "一些", "可以", "能够", "需要", "应该",
    "一个", "这个", "那个", "这些", "那些", "它们",
    "已经", "还是", "因为", "所以", "如果", "虽然", "但是",
}


class InvertedIndexBuilder:
    """
    倒排索引构建器。

    与 BM25Index 共享相同的 tokenize 逻辑（分词 + 停用词 + 最小长度过滤），
    确保建索引和检索时查询词的拆分方式一致。
    """

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        对文本做中文分词 + 清洗 + 去停用词。

        处理流程：
          1. 保留中文 + 英文 + 数字，去掉标点/特殊字符
          2. jieba 分词
          3. 小写化、去停用词、过滤长度 < 2 的词
        """
        import jieba

        cleaned = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text.lower()).strip()
        if not cleaned:
            return []

        words = jieba.lcut(cleaned)
        return [w for w in words if len(w) >= 2 and w not in STOP_WORDS]

    @staticmethod
    def build_terms(content: str) -> Counter:
        """对内容分词，返回 term → term_freq 的 Counter。"""
        tokens = InvertedIndexBuilder.tokenize(content)
        return Counter(tokens)

    # ----------------------------------------------------------------
    # 写入 MySQL
    # ----------------------------------------------------------------

    def add_chunks(self, chunks: List[Any], conn=None):
        """
        增量添加 chunks 到倒排索引。
        幂等操作（同 chunk_id 的 term 会被覆盖）。

        参数:
          chunks: List[DocumentChunk] 或 List[dict]（需有 chunk_id, content, doc_id, kb_id）
          conn: 可选，外部传入的数据库连接（用于事物内批量操作）
        """
        should_close = False
        if conn is None:
            conn = _get_connection()
            should_close = True

        try:
            with conn.cursor() as cursor:
                for chunk in chunks:
                    data = chunk.to_dict() if hasattr(chunk, "to_dict") else chunk
                    chunk_id = data.get("chunk_id", "")
                    content = data.get("content", "")
                    doc_id = data.get("doc_id", 0)
                    kb_id = data.get("kb_id")

                    if not chunk_id or not content:
                        continue

                    # --- 写入倒排表 ---
                    tf_counter = self.build_terms(content)
                    if not tf_counter:
                        continue

                    for term, freq in tf_counter.items():
                        cursor.execute(
                            """INSERT INTO document_terms (term, doc_id, chunk_id, term_freq, kb_id)
                               VALUES (%s, %s, %s, %s, %s)
                               ON DUPLICATE KEY UPDATE term_freq = VALUES(term_freq)""",
                            (term, doc_id, chunk_id, freq, kb_id),
                        )

                    # --- 写入文档统计 ---
                    total_terms = len(tf_counter)
                    cursor.execute(
                        """INSERT INTO doc_stats (chunk_id, doc_id, kb_id, total_terms)
                           VALUES (%s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE total_terms = VALUES(total_terms)""",
                        (chunk_id, doc_id, kb_id, total_terms),
                    )

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            if should_close:
                conn.close()

    def remove_document(self, doc_id: int):
        """
        删除文档时同步清理倒排索引。
        需在删除 document_chunks 后调用。
        """
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM document_terms WHERE doc_id = %s", (doc_id,))
                cursor.execute("DELETE FROM doc_stats WHERE doc_id = %s", (doc_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def rebuild_all(self):
        """
        从 document_chunks 全量重建倒排索引（幂等）。

        流程：
          1. TRUNCATE 两张索引表
          2. 读取所有 chunk
          3. 逐条分词写入
          4. 服务启动时或数据迁移后调用
        """
        from src.knowledge_ingest.chunk_store import ChunkStore

        conn = _get_connection()
        try:
            # 清空旧数据
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE document_terms")
                cursor.execute("TRUNCATE TABLE doc_stats")
            conn.commit()

            # 读取所有 chunk（使用 ChunkStore）
            chunk_store = ChunkStore()
            # ChunkStore 没有 "read all" 方法，直接读 MySQL
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT chunk_id, doc_id, kb_id, content FROM document_chunks ORDER BY doc_id, chunk_index"
                )
                rows = cursor.fetchall()

            # 分批写入倒排索引
            # 将 rows 转换为 dict 格式（ChunkStore 返回的已经是 dict）
            chunk_dicts = [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "kb_id": r["kb_id"],
                    "content": r["content"],
                }
                for r in rows
                if r.get("content")
            ]

            if not chunk_dicts:
                print("[INFO] document_chunks 为空，跳过倒排索引重建")
                return

            # 分批处理（每批 50 条，避免单次事务过大）
            BATCH_SIZE = 50
            for i in range(0, len(chunk_dicts), BATCH_SIZE):
                batch = chunk_dicts[i : i + BATCH_SIZE]
                self.add_chunks(batch, conn=conn)

            print(f"[INFO] 倒排索引重建完成: {len(chunk_dicts)} chunks 已索引")

        except Exception as e:
            print(f"[ERROR] 倒排索引重建失败: {e}")
            raise e
        finally:
            conn.close()
