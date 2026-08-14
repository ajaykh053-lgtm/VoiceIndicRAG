"""Strategy 1: Fixed-size Chunking with configurable window and token/word overlap."""

import uuid
from typing import List, Optional
from app.models.schemas import DocumentChunk, ChunkMetadata, ChunkingStrategyEnum


class FixedSizeChunker:
    """Chunks text into fixed token/word windows with configurable sliding overlap."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 64):
        """
        Args:
            chunk_size: approximate number of words/tokens per chunk
            chunk_overlap: number of overlapping words/tokens between consecutive chunks
        """
        self.chunk_size = max(10, chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size - 1)
        self.step_size = max(1, self.chunk_size - self.chunk_overlap)

    def chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        language: str = "hi",
        extra_metadata: Optional[dict] = None
    ) -> List[DocumentChunk]:
        """Split text into overlapping fixed-size chunks."""
        text = text.strip()
        if not text:
            return []

        # Split words while preserving Indic Unicode glyphs cleanly
        words = text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        extra_metadata = extra_metadata or {}

        # If text is smaller than chunk_size, return 1 chunk
        if len(words) <= self.chunk_size:
            chunk_id = f"fixed_{doc_id or uuid.uuid4().hex[:8]}_0"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                language=language,
                strategy=ChunkingStrategyEnum.FIXED_SIZE,
                start_char=0,
                end_char=len(text),
                extra={"word_count": len(words), "chunk_index": 0, **extra_metadata}
            )
            return [DocumentChunk(chunk_id=chunk_id, text=text, metadata=meta)]

        # Sliding window
        idx = 0
        chunk_idx = 0
        while idx < len(words):
            window_words = words[idx : idx + self.chunk_size]
            chunk_text = " ".join(window_words)
            chunk_id = f"fixed_{doc_id or uuid.uuid4().hex[:8]}_{chunk_idx}"

            meta = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                language=language,
                strategy=ChunkingStrategyEnum.FIXED_SIZE,
                start_char=0,
                end_char=len(chunk_text),
                extra={
                    "word_count": len(window_words),
                    "chunk_index": chunk_idx,
                    "window_start": idx,
                    "window_end": min(len(words), idx + self.chunk_size),
                    **extra_metadata
                }
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))

            idx += self.step_size
            chunk_idx += 1

            # Prevent trailing tiny fragment if already covered
            if idx >= len(words):
                break

        return chunks
