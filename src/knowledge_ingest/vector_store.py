import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
COLLECTION_NAME = "knowledge_vectors"


class EmbeddingService:
    """DashScope 文本嵌入服务"""

    def __init__(self, api_key: str = None, model: str = "text-embedding-v4"):
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.model = model
        self.dimension = VECTOR_SIZE

    def embed(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            import dashscope
            from dashscope import TextEmbedding
            resp = TextEmbedding.call(
                model=self.model,
                input=texts,
                api_key=self.api_key,
            )
            if resp.status_code == 200:
                embeddings = resp.output.get("embeddings", [])
                return [emb.get("embedding", []) for emb in embeddings]
            return [[0.0] * self.dimension for _ in texts]
        except Exception:
            return [[0.0] * self.dimension for _ in texts]


class VectorStore:
    """Qdrant 向量数据库存储，支持语义相似度搜索"""

    @staticmethod
    def _hex_to_uuid(hex_str: str) -> str:
        hex_clean = hex_str[:32]
        padded = hex_clean.zfill(32)
        return str(uuid.UUID(padded))

    def __init__(
            self,
            host: str = QDRANT_HOST,
            port: int = QDRANT_PORT,
            collection: str = COLLECTION_NAME,
            vector_size: int = VECTOR_SIZE
    ):
        self.host = host
        self.port = port
        self.collection = collection
        self.vector_size = vector_size
        self._client = None
        self._embedding = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                https=False,
                timeout=30,
            )
            collections = [
                c.name for c in self._client.get_collections().collections
            ]
            if self.collection not in collections:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        return self._client

    def _get_embedding(self) -> EmbeddingService:
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    def add_chunks(self, chunks: List[Any]) -> int:
        """将文档块向量化后存入 Qdrant"""
        from qdrant_client.models import PointStruct
        if not chunks:
            return 0

        embedding = self._get_embedding()
        points = []

        for chunk in chunks:
            if hasattr(chunk, "to_dict"):
                data = chunk.to_dict()
            elif isinstance(chunk, dict):
                data = chunk
            else:
                continue

            content = data.get("content", "")
            if not content.strip():
                continue

            chunk_id = data.get("chunk_id", "")
            try:
                vector = embedding.embed(content)
                points.append(PointStruct(
                    id=self._hex_to_uuid(chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "doc_id": data.get("doc_id"),
                        "kb_id": data.get("kb_id"),
                        "chunk_index": data.get("chunk_index", 0),
                        "content": content,
                        "filename": data.get("filename", ""),
                        "file_type": data.get("file_type", ""),
                    }
                ))
            except Exception:
                pass

        if points:
            client = self._get_client()
            client.upsert(collection_name=self.collection, points=points)

        return len(points)

    def search(
            self,
            query_text: str,
            kb_ids: Optional[List[int]] = None,
            top_k: int = 5,
            score_threshold: float = 0.6
    ) -> List[dict]:
        """向量相似度搜索"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        embedding = self._get_embedding()
        try:
            query_vector = embedding.embed(query_text)
        except Exception:
            return []

        query_filter = None
        if kb_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(key="kb_id", match=MatchValue(value=kbid))
                    for kbid in kb_ids
                ]
            )

        client = self._get_client()
        try:
            results = client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            return [
                {
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "score": hit.score,
                    "kb_id": hit.payload.get("kb_id"),
                    "doc_id": hit.payload.get("doc_id"),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "filename": hit.payload.get("filename"),
                    "file_type": hit.payload.get("file_type"),
                    "source": "vector",
                }
                for hit in results
            ]
        except Exception:
            return []

    def delete_by_document(self, doc_id: int) -> int:
        """删除指定文档的所有向量"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = self._get_client()
        try:
            result = client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            return result.status.value if hasattr(result, "status") else 0
        except Exception:
            return 0

    def delete_by_knowledge_base(self, kb_id: int) -> int:
        """删除指定知识库的所有向量"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = self._get_client()
        try:
            result = client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
                ),
            )
            return result.status.value if hasattr(result, "status") else 0
        except Exception:
            return 0

    def count_vectors(self, kb_id: Optional[int] = None) -> int:
        client = self._get_client()
        try:
            if kb_id is not None:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                result = client.count(
                    collection_name=self.collection,
                    count_filter=Filter(
                        must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
                    ),
                )
            else:
                result = client.count(collection_name=self.collection)
            return result.count
        except Exception:
            return 0
