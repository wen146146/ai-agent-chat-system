# -*- coding: utf-8 -*-
"""
BM25 稀疏检索引擎。

数据流：
  1. load_from_mysql() — 从 MySQL document_terms + doc_stats 加载到内存
  2. search(query) — 对查询分词 → 查倒排 → BM25Okapi 打分 → 排序返回

BM25Okapi 公式：
  score(D, Q) = Σ IDF(qi) · f(qi, D) · (k1+1) / (f(qi, D) + k1·(1-b+b·|D|/avgdl))

  其中:
    IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)    ← 平滑 IDF
    k1 = 1.5    (词频饱和参数)
    b  = 0.75   (文档长度归一化参数)

使用方式：
  from src.knowledge_ingest.bm25_index import BM25Index

  bm25 = BM25Index()
  bm25.load_from_mysql()

  results = bm25.search("Python 闭包原理")
  for r in results:
      print(r["chunk_id"], r["bm25_score"], r["content"][:50])
"""

import os
import re
import math
import sys
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.knowledge_ingest.inverted_index_schema import _get_connection


# ============================================================================
# 停用词表（与 bm25_indexer 保持一致！）
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


# ============================================================================
# BM25 引擎
# ============================================================================

class BM25Index:
    """
    BM25 稀疏检索引擎（内存缓存）。

    核心数据结构（load_from_mysql() 后构建）：
      - self.df:        Dict[str, int]          term → 包含该 term 的文档数
      - self.total_docs: int                    总文档数
      - self.avgdl:     float                   平均文档长度
      - self.doc_len:   Dict[str, int]          chunk_id → 该 chunk 的总词数
      - self.term_freqs: Dict[str, Dict[str, int]]  term → {chunk_id → tf}
      - self.doc_meta:  Dict[str, dict]         chunk_id → {doc_id, kb_id, content, filename}
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.df: Dict[str, int] = {}
        self.total_docs: int = 0
        self.avgdl: float = 0.0
        self.doc_len: Dict[str, int] = {}
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_meta: Dict[str, dict] = {}
        self._loaded: bool = False

    # ----------------------------------------------------------------
    # 索引加载
    # ----------------------------------------------------------------

    def load_from_mysql(self, kb_ids: Optional[List[int]] = None):
        """
        从 MySQL 倒排索引表加载到内存。

        参数:
          kb_ids: 可选，只加载指定知识库的文档，不传则加载全部

        复杂度: O(terms + docs) ≈ 5 万条记录 ~200ms
        """
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                # --- 加载文档统计 ---
                if kb_ids:
                    placeholders = ",".join(["%s"] * len(kb_ids))
                    cursor.execute(
                        f"SELECT chunk_id, doc_id, kb_id, total_terms FROM doc_stats WHERE kb_id IN ({placeholders})",
                        kb_ids,
                    )
                else:
                    cursor.execute("SELECT chunk_id, doc_id, kb_id, total_terms FROM doc_stats")

                stats_rows = cursor.fetchall()
                self.doc_len = {r["chunk_id"]: r["total_terms"] for r in stats_rows}
                self.total_docs = len(stats_rows)
                self.avgdl = (
                    sum(r["total_terms"] for r in stats_rows) / max(self.total_docs, 1)
                )

                # --- 加载倒排索引 ---
                if kb_ids:
                    cursor.execute(
                        f"SELECT term, chunk_id, term_freq, doc_id, kb_id FROM document_terms WHERE kb_id IN ({placeholders})",
                        kb_ids,
                    )
                else:
                    cursor.execute(
                        "SELECT term, chunk_id, term_freq, doc_id, kb_id FROM document_terms"
                    )

                rows = cursor.fetchall()
                self.term_freqs = {}
                term_docs: Dict[str, Set[str]] = {}

                for r in rows:
                    term = r["term"]
                    chunk_id = r["chunk_id"]
                    tf = r["term_freq"]

                    if term not in self.term_freqs:
                        self.term_freqs[term] = {}
                        term_docs[term] = set()
                    self.term_freqs[term][chunk_id] = tf
                    term_docs[term].add(chunk_id)

                # 计算 df（document frequency）
                self.df = {term: len(docs) for term, docs in term_docs.items()}

                # --- 加载文档元数据（从 document_chunks，只取需要的字段） ---
                if kb_ids:
                    cursor.execute(
                        f"SELECT chunk_id, doc_id, kb_id, content, filename FROM document_chunks WHERE kb_id IN ({placeholders})",
                        kb_ids,
                    )
                else:
                    cursor.execute(
                        "SELECT chunk_id, doc_id, kb_id, content, filename FROM document_chunks"
                    )

                for r in cursor.fetchall():
                    cid = r["chunk_id"]
                    if cid in self.doc_len:  # 只缓存有倒排统计的文档
                        self.doc_meta[cid] = {
                            "doc_id": r["doc_id"],
                            "kb_id": r["kb_id"],
                            "content": r.get("content", ""),
                            "filename": r.get("filename", ""),
                        }

            self._loaded = True

        except Exception as e:
            print(f"[WARN] BM25 索引加载失败（将降级为纯 Qdrant 检索）: {e}")
            self._loaded = False
        finally:
            conn.close()

    @property
    def is_loaded(self) -> bool:
        """索引是否已成功加载到内存。"""
        return self._loaded and self.total_docs > 0

    # ----------------------------------------------------------------
    # 分词（与 bm25_indexer 保持一致）
    # ----------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """对查询做中文分词 + 清洗 + 去停用词。"""
        import jieba

        cleaned = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text.lower()).strip()
        if not cleaned:
            return []
        words = jieba.lcut(cleaned)
        return [w for w in words if len(w) >= 2 and w not in STOP_WORDS]

    # ----------------------------------------------------------------
    # BM25 打分
    # ----------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """
        BM25 IDF（平滑版本，避免除零）。

        IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
        """
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score_for_doc(self, term: str, chunk_id: str, idf: float) -> float:
        """
        计算单个 term 对单个文档的 BM25 贡献分。

        score = IDF · tf · (k1+1) / (tf + k1·(1-b+b·|D|/avgdl))
        """
        tf = self.term_freqs.get(term, {}).get(chunk_id, 0)
        if tf == 0:
            return 0.0

        doc_len = self.doc_len.get(chunk_id, 1)
        doc_norm = 1 - self.b + self.b * doc_len / max(self.avgdl, 1)
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * doc_norm)

    # ----------------------------------------------------------------
    # 检索入口
    # ----------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """
        BM25 检索。

        流程：
          1. 对 query 分词（与建索引时一致）
          2. 对每个 term 查倒排表拿到候选 chunk
          3. BM25Okapi 公式打分
          4. 按分数降序返回 Top-K

        参数:
          query: 用户查询文本
          top_k: 返回结果数量（默认 20 条）

        返回:
          List[dict]，每项包含：
            - chunk_id, content, bm25_score, doc_id, kb_id, filename, source='bm25'
        """
        if not self.is_loaded:
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        # 收集候选文档
        candidates: Dict[str, float] = {}
        for term in terms:
            if term not in self.term_freqs:
                continue
            idf = self._idf(term)
            if idf == 0:
                continue
            for chunk_id in self.term_freqs[term]:
                score = self._score_for_doc(term, chunk_id, idf)
                if score > 0:
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + score

        if not candidates:
            return []

        # 按 BM25 分数降序排列
        ranked = sorted(candidates.items(), key=lambda x: -x[1])

        # 组装结果
        results = []
        for chunk_id, score in ranked[:top_k]:
            meta = self.doc_meta.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "content": meta.get("content", ""),
                "bm25_score": round(score, 4),
                "doc_id": meta.get("doc_id"),
                "kb_id": meta.get("kb_id"),
                "filename": meta.get("filename", ""),
                "source": "bm25",
            })

        return results

    def search_with_detail(self, query: str, top_k: int = 20) -> dict:
        """
        BM25 检索（含调试信息）。

        返回:
          {
            "results": [...],
            "terms": ["Python", "闭包"],
            "matched_terms": {"Python": 15, "闭包": 8},  # 每个 term 命中文档数
            "total_candidates": 23,
            "elapsed_ms": 12.5,
          }
        """
        import time
        t0 = time.time()

        terms = self._tokenize(query)
        matched_terms = {}
        for t in terms:
            if t in self.df:
                matched_terms[t] = self.df[t]

        results = self.search(query, top_k)

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "results": results,
            "terms": terms,
            "matched_terms": matched_terms,
            "total_candidates": len(set().union(*[
                set(self.term_freqs.get(t, {}).keys())
                for t in terms if t in self.term_freqs
            ])) if terms else 0,
            "elapsed_ms": elapsed_ms,
        }
