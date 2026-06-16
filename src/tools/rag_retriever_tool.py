import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import sys
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.knowledge_ingest.vector_store import VectorStore, EmbeddingService
from src.knowledge_ingest.chunk_store import ChunkStore
from src.knowledge_ingest.bm25_index import BM25Index

QUERY_PREPROCESS_ENABLED = os.getenv("QUERY_PREPROCESS_ENABLED", "true").lower() == "true"
QUERY_EXPAND_ENABLED = os.getenv("QUERY_EXPAND_ENABLED", "true").lower() == "true"
QUERY_EXPAND_COUNT = int(os.getenv("QUERY_EXPAND_COUNT", "3"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.6"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RRF_K = int(os.getenv("RRF_K", "60"))

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")


CN_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "怎样", "哪", "哪里", "吗", "呢", "吧", "啊", "哦",
}


class SimpleLLM:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from langchain_community.chat_models import ChatOpenAI
            self._client = ChatOpenAI(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._client

    def invoke(self, prompt: str) -> str:
        try:
            client = self._get_client()
            response = client.invoke(prompt)
            return response.content
        except Exception:
            return ""


class RagRetrieveInput(BaseModel):
    query: str = Field(description="用户问题")
    kb_ids: List[int] = Field(default_factory=list, description="知识库ID列表，传空数组则搜索全部")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量，默认5条，最多20条")


class RagRetrieveTool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._vector_store = None
        self._chunk_store = None
        self._embedding = None
        self._llm = None
        self._bm25 = None
        self._initialized = True

    def initialize(self):
        if self._vector_store is not None:
            return
        try:
            self._vector_store = VectorStore()
            self._chunk_store = ChunkStore()
            self._embedding = EmbeddingService()
            self._llm = SimpleLLM()
        except Exception:
            pass

        # 加载 BM25 索引（独立 try-catch，失败不阻塞其他检索）
        try:
            bm25 = BM25Index()
            bm25.load_from_mysql()
            if bm25.is_loaded:
                self._bm25 = bm25
        except Exception:
            pass

    def query_preprocess(self, query: str) -> str:
        if not QUERY_PREPROCESS_ENABLED:
            return query
        prompt = f"""请将以下问题进行标准化、润色和纠错：

问题：{query}

请直接输出处理后的问题，不要解释："""
        try:
            result = self._llm.invoke(prompt)
            if result:
                return result.strip()
        except Exception:
            pass
        return query

    def query_expand(self, query: str) -> List[str]:
        if not QUERY_EXPAND_ENABLED:
            return [query]
        prompt = f"""请为以下问题生成{QUERY_EXPAND_COUNT}条语义相似的不同表达方式：

问题：{query}

请逐行输出，每行一条，不要编号："""
        try:
            result = self._llm.invoke(prompt)
            lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
            expanded = [query] + lines[:QUERY_EXPAND_COUNT]
            return expanded
        except Exception:
            return [query]

    def extract_keywords_local(self, text: str) -> List[str]:
        try:
            import jieba
            cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text).strip()
            if not cleaned:
                return []
            words = jieba.lcut(cleaned)
            keywords = []
            seen = set()
            for w in words:
                w = w.strip()
                if len(w) < 2 or w in CN_STOP_WORDS or w in seen:
                    continue
                seen.add(w)
                keywords.append(w)
            return keywords[:10]
        except Exception:
            return []

    def extract_keywords(self, query: str) -> Tuple[List[str], Dict[str, List[str]]]:
        prompt = f"""请从以下问题中提取关键词和同义词：

问题：{query}

请以JSON格式输出：
{{"keywords": ["关键词1", "关键词2"], "synonyms": {{"关键词1": ["同义词1", "同义词2"]}}}}"""
        try:
            result = self._llm.invoke(prompt)
            data = json.loads(result)
            keywords = data.get("keywords", [])
            synonyms = data.get("synonyms", {})
            return keywords, synonyms
        except Exception:
            keywords = self.extract_keywords_local(query)
            return keywords, {}

    def vector_search(self, queries: List[str], kb_ids: List[int] = None, top_k: int = 5) -> List[dict]:
        if self._vector_store is None:
            return []
        if kb_ids is None:
            kb_ids = []

        all_results = []
        seen_contents = set()

        for query in queries:
            try:
                results = self._vector_store.search(
                    query_text=query,
                    kb_ids=kb_ids or None,
                    top_k=top_k,
                    score_threshold=RETRIEVER_SCORE_THRESHOLD
                )
                for item in results:
                    content_key = item.get("content", "")[:100]
                    if content_key not in seen_contents:
                        seen_contents.add(content_key)
                        item["source"] = "vector"
                        all_results.append(item)
            except Exception:
                pass

        return all_results

    def keyword_search_mysql(
            self, keywords: List[str], synonyms: Dict[str, List[str]] = None,
            kb_ids: List[int] = None, top_k: int = 5
    ) -> List[dict]:
        if not keywords or self._chunk_store is None:
            return []
        try:
            results = self._chunk_store.exact_match_search(
                keywords=keywords[:5],
                synonyms=synonyms,
                kb_ids=kb_ids or None,
                limit=top_k * 2
            )
            return results
        except Exception:
            return []

    @staticmethod
    def parse_rerank_response(text: str, doc_count: int) -> list:
        """
        从 LLM 回复中提取排序结果。
        尝试顺序：```json 代码块 → 裸 JSON → 纯数字 → 回退原始顺序
        """
        import re, json
        # 1) 匹配 ```json 代码块
        match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\n?\s*```', text, re.DOTALL)
        # 2) 匹配裸 JSON
        if not match:
            match = re.search(r'\{.*?"ranked_indices".*?\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1) if match.lastindex else match.group())
                indices = data.get("ranked_indices", list(range(doc_count)))
                return [i for i in indices if 0 <= i < doc_count]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        # 3) 容错：提取数字
        numbers = [int(n) - 1 for n in re.findall(r'\d+', text) if n.isdigit()]
        if numbers:
            return [n for n in numbers if 0 <= n < doc_count]
        # 4) 回退
        return list(range(doc_count))

    def rerank(self, query: str, results: List[dict]) -> List[dict]:
        if not results or not RERANK_ENABLED:
            return results
        try:
            docs_content = "\n\n".join([
                f"[{i+1}] {item.get('content', '')[:300]}"
                for i, item in enumerate(results)
            ])
            prompt = f"""你是一个文档相关性评估专家。

用户查询：{query}

候选文档：
{docs_content}

请从以下维度评估：
1. 与查询的主题相关度
2. 信息完整度
3. 时效性

输出 JSON：{{"ranked_indices": [3, 1, 5, ...], "scores": [0.95, 0.82, ...], "reason": "简要说明"}}"""
            result = self._llm.invoke(prompt)
            order = self.parse_rerank_response(result, len(results))

            reranked = []
            seen = set()
            for idx in order:
                if idx not in seen:
                    seen.add(idx)
                    reranked.append(results[idx])
            for i, item in enumerate(results):
                if i not in seen:
                    reranked.append(item)
            return reranked
        except Exception:
            return results

    def merge_results(self, vector_results: List[dict], mysql_results: List[dict], top_k: int) -> List[dict]:
        """旧版双路融合（保留向后兼容，但 retrieve() 已改用 RRF）。"""
        return self.fuse_results(
            vector_results=vector_results,
            bm25_results=[],
            keyword_results=mysql_results,
            top_k=top_k,
        )

    # ----------------------------------------------------------------
    # BM25 稀疏检索
    # ----------------------------------------------------------------

    def bm25_search(self, query: str, kb_ids: List[int] = None, top_k: int = 20) -> List[dict]:
        """BM25 稀疏检索入口（封装 BM25Index.search）。"""
        if self._bm25 is None:
            return []
        return self._bm25.search(query, top_k)

    # ----------------------------------------------------------------
    # RRF 三路融合
    # ----------------------------------------------------------------

    @staticmethod
    def fuse_results(
        vector_results: List[dict],
        bm25_results: List[dict],
        keyword_results: List[dict],
        top_k: int = 5,
        k: int = None,
    ) -> List[dict]:
        """
        RRF（Reciprocal Rank Fusion）三路结果融合。

        score(doc) = 1/(k + rank_vector) + 1/(k + rank_bm25) + 1/(k + rank_keyword)

        优点：
          - 排名比分数更稳定，消除跨模型分数量纲差
          - Qdrant cosine (0~1) 和 BM25 (0~open) 的分数直接不可比
          - 排第 1 名贡献 1/(k+1)，排第 10 名贡献 1/(k+10)，差距平滑
        """
        if k is None:
            k = RRF_K

        def _rank(items: List[dict], sort_key: str) -> dict:
            """将结果列表按 sort_key 降序排列，返回 {dedup_key: rank}。"""
            ranked = {}
            for idx, item in enumerate(
                sorted(items, key=lambda x: abs(x.get(sort_key, 0)), reverse=True), 1
            ):
                key = item.get("chunk_id") or item.get("content", "")[:100]
                if key not in ranked:
                    ranked[key] = idx
            return ranked

        vector_ranked = _rank(vector_results, "score") if vector_results else {}
        bm25_ranked = _rank(bm25_results, "bm25_score") if bm25_results else {}
        keyword_ranked = _rank(keyword_results, "score") if keyword_results else {}

        # 所有候选文档的 dedup key
        all_keys = set(vector_ranked) | set(bm25_ranked) | set(keyword_ranked)

        # 计算 RRF 分数
        rrf_scores = {}
        for key in all_keys:
            score = 0.0
            score += 1.0 / (k + vector_ranked.get(key, 999))
            score += 1.0 / (k + bm25_ranked.get(key, 999))
            score += 1.0 / (k + keyword_ranked.get(key, 999))
            rrf_scores[key] = round(score, 6)

        # 从原始结果中提取 metadata
        meta_cache: Dict[str, dict] = {}
        for item in vector_results + bm25_results + keyword_results:
            key = item.get("chunk_id") or item.get("content", "")[:100]
            if key not in meta_cache:
                meta_cache[key] = {
                    "content": item.get("content", ""),
                    "filename": item.get("filename", ""),
                    "doc_id": item.get("doc_id"),
                    "kb_id": item.get("kb_id"),
                    "source_types": set(),
                    "best_score": 0.0,
                }
            # 记录来自哪一路
            src = item.get("source", "unknown")
            meta_cache[key]["source_types"].add(src)
            meta_cache[key]["best_score"] = max(
                meta_cache[key]["best_score"],
                abs(item.get("score", 0)) or abs(item.get("bm25_score", 0)),
            )

        # 按 RRF 分数降序取 Top-K
        ranked_keys = sorted(rrf_scores.keys(), key=lambda k: -rrf_scores[k])

        results = []
        for key in ranked_keys[:top_k]:
            meta = meta_cache.get(key, {})
            source_types = meta.get("source_types", set())
            # source 标签：优先显示主要来源
            source_label = "fusion"
            if source_types == {"vector"}:
                source_label = "vector"
            elif source_types == {"bm25"}:
                source_label = "bm25"
            elif source_types == {"keyword"} or source_types == {"mysql"}:
                source_label = "keyword"

            results.append({
                "content": meta.get("content", ""),
                "rrf_score": rrf_scores[key],
                "score": meta.get("best_score", 0),
                "filename": meta.get("filename", ""),
                "doc_id": meta.get("doc_id"),
                "kb_id": meta.get("kb_id"),
                "source": source_label,
                "source_detail": "+".join(sorted(source_types)),
            })

        return results

    def retrieve(self, query: str, kb_ids: List[int] = None, top_k: int = 5) -> List[dict]:
        """
        三路检索 + RRF 融合 + LLM 重排 完整管线。

        检索路数：
          ① Qdrant 语义向量检索
          ② BM25 稀疏检索（倒排索引）
          ③ MySQL LIKE 关键词检索（保留向下兼容）

        融合策略：RRF Reciprocal Rank Fusion
        """
        if kb_ids is None:
            kb_ids = []

        # 取更多候选让融合 + 重排有足够素材
        retrieve_k = max(top_k * 3, 15)

        # 1. Query 预处理 + 扩写（现有不变）
        processed_query = self.query_preprocess(query)
        expanded_queries = self.query_expand(processed_query)

        # 2. 三路并行检索
        vector_results = self.vector_search(expanded_queries, kb_ids, retrieve_k)

        keywords, synonyms = self.extract_keywords(query)
        mysql_results = self.keyword_search_mysql(keywords, synonyms, kb_ids, retrieve_k)

        bm25_results = self.bm25_search(query, kb_ids, retrieve_k)

        # 3. RRF 三路融合
        fused_results = self.fuse_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            keyword_results=mysql_results,
            top_k=top_k * 2 if RERANK_ENABLED else top_k,
        )

        # 4. LLM 重排（可选）
        if RERANK_ENABLED:
            fused_results = self.rerank(query, fused_results)

        return fused_results[:top_k]

    def format_results(self, results: List[dict]) -> str:
        if not results:
            return "暂无相关参考资料"
        lines = ["📖 **参考资料：**"]
        for i, item in enumerate(results, 1):
            filename = item.get("filename", "未知文件")
            score = item.get("rrf_score") or item.get("score", 0)
            content = item.get("content", "")[:200].replace("\n", " ")
            source_label = {
                "vector": "🟦[向量]",
                "bm25": "🟩[BM25]",
                "keyword": "🟨[关键词]",
                "fusion": "🔀[融合]",
            }.get(item.get("source", ""), "")
            detail = item.get("source_detail", "")
            tag = f"{source_label}" if not detail else f"{source_label}({detail})"
            lines.append(f"{i}. {tag}【{filename}】相关度:{score:.4f}\n   {content}...")
        return "\n\n".join(lines)


_rag_tool_instance = RagRetrieveTool()


def _init_rag_retriever():
    """初始化 RAG 检索器（懒加载）：确保数据库表就绪 + 加载索引。"""
    from src.knowledge_ingest.inverted_index_schema import ensure_inverted_index_tables
    ensure_inverted_index_tables()
    _rag_tool_instance.initialize()


@tool(args_schema=RagRetrieveInput)
def rag_retrieve(query: str, kb_ids: List[int] = None, top_k: int = 5) -> str:
    """
    RAG知识库检索工具，输入用户问题，从知识库中检索相关文档内容。
    三路检索 + RRF 融合 + LLM 重排完整管线：
    1. 查询预处理 - LLM标准化+润色+纠错
    2. 查询扩写 - LLM生成多条语义相似表达
    3. 向量检索(Qdrant) - 语义搜索
    4. BM25稀疏检索(倒排索引) - 词频统计排序
    5. 关键词检索(MySQL) - 同义词精确匹配 (保留)
    6. RRF三路融合 - Reciprocal Rank Fusion
    7. Rerank重排(可选) - LLM精排
    """
    _init_rag_retriever()

    try:
        results = _rag_tool_instance.retrieve(query, kb_ids, top_k)
        return _rag_tool_instance.format_results(results)
    except Exception as e:
        return f"[错误] 检索异常: {str(e)}"
