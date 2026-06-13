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

QUERY_PREPROCESS_ENABLED = os.getenv("QUERY_PREPROCESS_ENABLED", "true").lower() == "true"
QUERY_EXPAND_ENABLED = os.getenv("QUERY_EXPAND_ENABLED", "true").lower() == "true"
QUERY_EXPAND_COUNT = int(os.getenv("QUERY_EXPAND_COUNT", "3"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.6"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"

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

    def rerank(self, query: str, results: List[dict]) -> List[dict]:
        if not results or not RERANK_ENABLED:
            return results
        try:
            docs_content = "\n\n".join([
                f"[文档{i+1}]\n{item.get('content', '')[:500]}"
                for i, item in enumerate(results)
            ])
            prompt = f"""请根据以下问题，对文档进行相关性排序：

问题：{query}

{docs_content}

请按相关性从高到低输出文档编号，用逗号分隔（如：3,1,2）："""
            result = self._llm.invoke(prompt)
            order_str = result.strip().replace("[", "").replace("]", "")
            order = []
            for num_str in order_str.split(","):
                try:
                    order.append(int(num_str.strip()) - 1)
                except Exception:
                    pass
            if order:
                reranked = []
                for idx in order:
                    if 0 <= idx < len(results):
                        reranked.append(results[idx])
                for item in results:
                    if item not in reranked:
                        reranked.append(item)
                return reranked
        except Exception:
            pass
        return results

    def merge_results(self, vector_results: List[dict], mysql_results: List[dict], top_k: int) -> List[dict]:
        seen_contents = set()
        merged = []

        for item in sorted(vector_results, key=lambda x: x.get("score", 0), reverse=True):
            content_key = item.get("content", "")[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(item)

        for item in sorted(mysql_results, key=lambda x: x.get("score", 0), reverse=True):
            content_key = item.get("content", "")[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(item)

        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:top_k]

    def retrieve(self, query: str, kb_ids: List[int] = None, top_k: int = 5) -> List[dict]:
        if kb_ids is None:
            kb_ids = []

        processed_query = self.query_preprocess(query)
        expanded_queries = self.query_expand(processed_query)

        vector_results = self.vector_search(expanded_queries, kb_ids, top_k)

        keywords, synonyms = self.extract_keywords(query)

        mysql_results = self.keyword_search_mysql(keywords, synonyms, kb_ids, top_k)

        merged_results = self.merge_results(vector_results, mysql_results, top_k)

        if RERANK_ENABLED:
            merged_results = self.rerank(query, merged_results)

        return merged_results

    def format_results(self, results: List[dict]) -> str:
        if not results:
            return "暂无相关参考资料"
        lines = ["📖 **参考资料：**"]
        for i, item in enumerate(results, 1):
            filename = item.get("filename", "未知文件")
            score = item.get("score", 0)
            content = item.get("content", "")[:200].replace("\n", " ")
            source_tag = {"vector": "[向量]", "mysql": "[关键词]"}.get(item.get("source", ""), "")
            lines.append(f"{i}. {source_tag}【{filename}】相关度:{score:.2f}\n   {content}...")
        return "\n\n".join(lines)


_rag_tool_instance = RagRetrieveTool()


def _init_rag_retriever():
    _rag_tool_instance.initialize()


@tool(args_schema=RagRetrieveInput)
def rag_retrieve(query: str, kb_ids: List[int] = None, top_k: int = 5) -> str:
    """
    RAG知识库检索工具，输入用户问题，从知识库中检索相关文档内容。
    多阶段检索流程：
    1. 查询预处理 - LLM标准化+润色+纠错
    2. 查询扩写 - LLM生成多条语义相似表达
    3. 向量检索(Qdrant) - 批量语义搜索+去重
    4. 关键词检索(MySQL) - 关键词+同义词精确匹配
    5. 结果合并 - 双路去重+排序
    6. Rerank重排(预留) - RERANK_ENABLED=true时启用
    """
    _init_rag_retriever()

    try:
        results = _rag_tool_instance.retrieve(query, kb_ids, top_k)
        return _rag_tool_instance.format_results(results)
    except Exception as e:
        return f"[错误] 检索异常: {str(e)}"
