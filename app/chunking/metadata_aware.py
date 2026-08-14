"""Strategy 3: Metadata-aware Chunking with rich contextual headers & structured metadata."""

import uuid
from typing import List, Optional, Dict, Any
from app.models.schemas import DocumentChunk, ChunkMetadata, ChunkingStrategyEnum


class MetadataAwareChunker:
    """Enriches chunk text with contextual header prefixes (title, section, lang) to boost dense retrieval."""

    def __init__(self, chunk_size: int = 200, include_header_in_text: bool = True):
        self.chunk_size = chunk_size
        self.include_header_in_text = include_header_in_text

    def chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        language: str = "hi",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Creates metadata-enriched chunks with contextual document headers."""
        text = text.strip()
        if not text:
            return []

        extra_metadata = extra_metadata or {}
        title = extra_metadata.get("title", "")
        category = extra_metadata.get("category", "")
        query_id = extra_metadata.get("query_id", "")
        passage_id = extra_metadata.get("passage_id", "")

        words = text.split()
        chunks: List[DocumentChunk] = []

        # Build contextual prefix
        header_parts = []
        if title:
            header_parts.append(f"शीर्षक: {title}" if language == "hi" else f"Title: {title}")
        if category:
            header_parts.append(f"श्रेणी: {category}" if language == "hi" else f"Category: {category}")
        if language:
            header_parts.append(f"भाषा: {language}")

        header_prefix = " | ".join(header_parts)
        if header_prefix:
            header_prefix = f"[{header_prefix}]\n"

        # Split into blocks of roughly chunk_size words
        idx = 0
        chunk_idx = 0
        step = max(50, self.chunk_size - 40)

        while idx < len(words):
            window_words = words[idx : idx + self.chunk_size]
            body_text = " ".join(window_words)

            if self.include_header_in_text and header_prefix:
                full_text = f"{header_prefix}{body_text}"
            else:
                full_text = body_text

            chunk_id = f"meta_{doc_id or uuid.uuid4().hex[:8]}_{chunk_idx}"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                language=language,
                strategy=ChunkingStrategyEnum.METADATA_AWARE,
                start_char=0,
                end_char=len(full_text),
                extra={
                    "title": title,
                    "category": category,
                    "query_id": query_id,
                    "passage_id": passage_id,
                    "word_count": len(window_words),
                    "chunk_index": chunk_idx,
                    **extra_metadata
                }
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=full_text, metadata=meta))

            idx += step
            chunk_idx += 1
            if idx >= len(words):
                break

        return chunks
