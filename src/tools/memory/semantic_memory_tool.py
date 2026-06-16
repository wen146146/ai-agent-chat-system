# -*- coding: utf-8 -*-
# 作用：语义记忆工具，@tool装饰器实现，MySQL+Qdrant双写，支持向量/关键词/混合三种检索的CRUD+统计
import os

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "ai_agent_db")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "semantic_memory")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")


def _get_mysql_conn():
    import pymysql
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )


def _ensure_mysql_table():
    conn = _get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(128) NOT NULL,
                    title VARCHAR(256) NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(64) DEFAULT '',
                    keywords VARCHAR(512) DEFAULT '',
                    metadata JSON DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id),
                    INDEX idx_category (category),
                    INDEX idx_created (created_at),
                    FULLTEXT idx_content (content)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


try:
    _ensure_mysql_table()
except Exception:
    pass  # MySQL不可用时跳过，工具调用时再重试


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _ensure_qdrant_collection():
    """按需初始化 Qdrant collection"""
    try:
        client = _get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
    except Exception:
        pass  # Qdrant 不可用时静默跳过，不影响服务器启动


_ensure_qdrant_collection()


def _get_embedding(text: str) -> List[float]:
    import dashscope
    dashscope.api_key = DASHSCOPE_API_KEY
    resp = dashscope.TextEmbedding.call(
        model=DASHSCOPE_EMBEDDING_MODEL,
        input=text
    )
    if resp.status_code == 200:
        return resp.output["embeddings"][0]["embedding"]
    return [0.0] * VECTOR_DIM


class _SemanticMemoryStore:
    def save(self, user_id: str, title: str, content: str,
             category: str = "", metadata: Dict[str, Any] = None) -> int:
        if metadata is None:
            metadata = {}

        conn = _get_mysql_conn()
        try:
            with conn.cursor() as cur:
                keywords = self._extract_keywords(content)
                keywords_str = ",".join(keywords[:10])
                cur.execute(
                    """INSERT INTO semantic_memories
                       (user_id, title, content, category, keywords, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (user_id, title, content, category, keywords_str,
                     json.dumps(metadata, ensure_ascii=False))
                )
            conn.commit()
            mem_id = cur.lastrowid
        finally:
            conn.close()

        try:
            vector = _get_embedding(content[:1000])
            client = _get_qdrant_client()
            from qdrant_client.models import PointStruct
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[PointStruct(
                    id=mem_id,
                    vector=vector,
                    payload={
                        "user_id": user_id, "title": title,
                        "category": category, "mysql_id": mem_id
                    }
                )]
            )
        except Exception:
            pass

        return mem_id

    def _extract_keywords(self, text: str) -> List[str]:
        try:
            import jieba
            stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                          "什么", "怎么", "怎样", "哪", "哪里", "哪些", "如何"}
            cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text).strip()
            if not cleaned:
                return []
            words = jieba.lcut(cleaned)
            seen = set()
            result = []
            for w in words:
                w = w.strip()
                if len(w) >= 2 and w not in stop_words and w not in seen:
                    seen.add(w)
                    result.append(w)
            return result[:10]
        except Exception:
            return []

    def search(self, user_id: str, query: str, search_type: str = "hybrid",
               category: str = "", top_k: int = 5) -> List[Dict]:

        vector_results = []
        keyword_results = []

        if search_type in ("vector", "hybrid"):
            try:
                query_vector = _get_embedding(query)
                client = _get_qdrant_client()
                qdrant_result = client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_vector,
                    query_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
                    limit=top_k * 2,
                    score_threshold=0.35
                )
                for hit in qdrant_result:
                    mysql_id = hit.payload.get("mysql_id", hit.id)
                    vector_results.append({
                        "id": mysql_id, "title": hit.payload.get("title", ""),
                        "score": hit.score, "source": "vector"
                    })
            except Exception:
                pass

        if search_type in ("keyword", "hybrid"):
            keywords = self._extract_keywords(query)
            if keywords:
                conn = _get_mysql_conn()
                try:
                    with conn.cursor() as cur:
                        conditions = []
                        params = [user_id]
                        for kw in keywords[:5]:
                            conditions.append("(title LIKE %s OR content LIKE %s OR keywords LIKE %s)")
                            params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])
                        if category:
                            conditions.append("category = %s")
                            params.append(category)
                        where = " AND ".join(conditions)
                        cur.execute(
                            f"SELECT id, title, content, category, created_at FROM semantic_memories "
                            f"WHERE user_id = %s AND ({where}) ORDER BY created_at DESC LIMIT %s",
                            params[:1] + params[1:] + [top_k * 2]
                        )
                        for row in cur.fetchall():
                            row["score"] = 0.7
                            row["source"] = "keyword"
                            keyword_results.append(row)
                finally:
                    conn.close()

        if search_type == "vector":
            return vector_results[:top_k]
        elif search_type == "keyword":
            return keyword_results[:top_k]

        merged = {}
        for r in vector_results:
            merged[r["id"]] = r
        for r in keyword_results:
            kid = r["id"]
            if kid in merged:
                merged[kid]["score"] = max(merged[kid]["score"], r.get("score", 0))
            else:
                merged[kid] = r

        return sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    def delete(self, memory_id: int) -> bool:
        conn = _get_mysql_conn()
        ok = False
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM semantic_memories WHERE id = %s", (memory_id,))
            conn.commit()
            ok = cur.rowcount > 0
        finally:
            conn.close()
        try:
            client = _get_qdrant_client()
            from qdrant_client.models import PointIdsSelector
            client.delete(collection_name=QDRANT_COLLECTION, points_selector=PointIdsSelector(points=[memory_id]))
        except Exception:
            pass
        return ok

    def count(self, user_id: str = "") -> int:
        conn = _get_mysql_conn()
        try:
            with conn.cursor() as cur:
                if user_id:
                    cur.execute(
                        "SELECT COUNT(*) AS cnt FROM semantic_memories WHERE user_id = %s",
                        (user_id,)
                    )
                else:
                    cur.execute("SELECT COUNT(*) AS cnt FROM semantic_memories")
                return cur.fetchone()["cnt"]
        finally:
            conn.close()


_store = _SemanticMemoryStore()


class SemanticSaveInput(BaseModel):
    user_id: str = Field(description="用户唯一标识")
    title: str = Field(description="记忆标题，简明概括内容主旨")
    content: str = Field(description="要保存的知识/偏好/总结文本")
    category: str = Field(default="", description="分类标签，如 preference/knowledge/summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据，键值对格式")


@tool(args_schema=SemanticSaveInput)
def semantic_memory_save(user_id: str, title: str, content: str,
                         category: str = "", metadata: Dict[str, Any] = None) -> str:
    """
    保存语义记忆到长期知识库。当用户说"记住我喜欢..."、"我的偏好是..."、"总结一下刚才的内容"时调用。
    语义记忆 = 长期知识（"用户喜欢吃辣"、"用户是Java后端工程师"），支持关键词和语义两种搜索方式。
    同时写入MySQL和Qdrant，区别于情景记忆（对话流水账）。
    """
    if metadata is None:
        metadata = {}
    try:
        mem_id = _store.save(user_id, title, content, category, metadata)
        return f"[语义记忆] 已保存 #{mem_id} 标题:{title}"
    except Exception as e:
        return f"[语义记忆] 保存失败: {str(e)}"


class SemanticSearchInput(BaseModel):
    user_id: str = Field(description="用户唯一标识")
    query: str = Field(description="查询内容，用于匹配语义记忆")
    search_type: Literal["vector", "keyword", "hybrid"] = Field(
        default="hybrid", description="检索类型: vector(向量语义) keyword(关键词) hybrid(混合)"
    )
    top_k: int = Field(default=5, description="返回结果数量，默认5")
    category: str = Field(default="", description="按分类过滤，留空不过滤")


@tool(args_schema=SemanticSearchInput)
def semantic_memory_search(user_id: str, query: str, search_type: str = "hybrid",
                           top_k: int = 5, category: str = "") -> str:
    """
    搜索已保存的语义记忆。当需要回顾之前用户说过的偏好、知识要点或总结时调用。
    支持三种模式：vector（语义搜索）、keyword（关键词精确）、hybrid（混合，推荐）。
    适合场景："我之前说过什么偏好"、"帮我找找关于XXX的笔记"、"查一下我存过的知识点"。
    """
    results = _store.search(user_id, query, search_type, category, top_k)
    if not results:
        return "暂无匹配的语义记忆"
    lines = [f"[语义记忆] ({len(results)}条 模式:{search_type}):"]
    for item in results:
        title = item.get("title", "无标题")
        score = item.get("score", 0)
        lines.append(f"  #{item['id']} [{item.get('category','')}] {title}")
        lines.append(f"    相关度:{score:.4f}")
    return "\n".join(lines)


class SemanticDeleteInput(BaseModel):
    memory_id: int = Field(description="要删除的语义记忆ID")


@tool(args_schema=SemanticDeleteInput)
def semantic_memory_delete(memory_id: int) -> str:
    """
    按ID删除一条语义记忆。同时从MySQL和Qdrant中移除。需要先通过 semantic_memory_search 查到记忆的ID。
    """
    ok = _store.delete(memory_id)
    return f"[语义记忆] #{memory_id} {'已删除' if ok else '删除失败(不存在)'}"


class SemanticCountInput(BaseModel):
    user_id: str = Field(default="", description="用户ID，留空则统计全部")


@tool(args_schema=SemanticCountInput)
def semantic_memory_count(user_id: str = "") -> str:
    """
    统计语义记忆的总条数。返回指定用户或全部用户的记忆数量，用于了解知识库的积累情况。
    """
    cnt = _store.count(user_id)
    scope = f"用户 {user_id}" if user_id else "全部"
    return f"[语义记忆统计] {scope}: {cnt} 条记忆"
