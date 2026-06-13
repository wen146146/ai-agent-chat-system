import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import re
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    "  ",
]


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_index": self.metadata.get("chunk_index", 0),
            "kb_id": self.metadata.get("kb_id"),
            "doc_id": self.metadata.get("doc_id"),
            "filename": self.metadata.get("filename", ""),
            "file_type": self.metadata.get("file_type", ""),
        }


class LangChainTextSplitter:
    """基于 LangChain RecursiveCharacterTextSplitter 的分块器"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(cleaned) <= self.chunk_size:
            return [cleaned] if cleaned else []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=SEPARATORS,
                keep_separator=True,
            )
            return splitter.split_text(cleaned)
        except ImportError:
            return self._fallback_split(cleaned)

    def _fallback_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            for sep in SEPARATORS:
                if end < len(text):
                    last_sep = chunk.rfind(sep)
                    if last_sep > 0:
                        end = start + last_sep + len(sep)
                        chunk = text[start:end]
                        break
            chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks


class DocumentProcessor:
    """文档处理器：将 Markdown 文本切分成文本块"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = LangChainTextSplitter(chunk_size, chunk_overlap)

    def _generate_chunk_id(self, doc_id: int, chunk_index: int) -> str:
        raw = f"{doc_id}_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def process_document(
            self,
            markdown_text: str,
            doc_id: int,
            kb_id: Optional[int] = None,
            filename: str = "",
            file_type: str = ""
    ) -> List[DocumentChunk]:
        text_chunks = self._splitter.split_text(markdown_text)
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(doc_id, i)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "chunk_index": i,
                    "filename": filename,
                    "file_type": file_type,
                }
            ))
        return chunks

    def get_stats(self, markdown_text: str) -> dict:
        text_chunks = self._splitter.split_text(markdown_text)
        return {
            "total_chars": len(markdown_text),
            "chunk_count": len(text_chunks),
            "avg_chunk_size": sum(len(c) for c in text_chunks) / max(len(text_chunks), 1),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
