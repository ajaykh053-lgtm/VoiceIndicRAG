"""Strategy 2: Sentence-level Semantic Chunking supporting Indic sentence boundary markers."""

import re
import uuid
from typing import List, Optional
from app.models.schemas import DocumentChunk, ChunkMetadata, ChunkingStrategyEnum


class SentenceSemanticChunker:
    """Chunks text at natural sentence boundaries, grouping sentences up to target length."""

    def __init__(self, target_sentences: int = 3, max_words: int = 200):
        self.target_sentences = max(1, target_sentences)
        self.max_words = max_words
        # Regex matching English (. ! ?) and Indic sentence delimiters (। ॥ ? !)
        self.sentence_regex = re.compile(r'(?<=[.!?।॥\n])\s+')

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling both English and Indic script punctuations."""
        sentences = self.sentence_regex.split(text.strip())
        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned if cleaned else [text.strip()]

    def chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        language: str = "hi",
        extra_metadata: Optional[dict] = None
    ) -> List[DocumentChunk]:
        """Group semantic sentences respecting boundaries and max word targets."""
        text = text.strip()
        if not text:
            return []

        sentences = self.split_into_sentences(text)
        chunks: List[DocumentChunk] = []
        extra_metadata = extra_metadata or {}

        current_group: List[str] = []
        current_word_count = 0
        chunk_idx = 0

        for s in sentences:
            s_words = len(s.split())
            if (len(current_group) >= self.target_sentences or 
                (current_word_count + s_words > self.max_words and current_group)):
                # Flush group
                chunk_text = " ".join(current_group)
                chunk_id = f"sent_{doc_id or uuid.uuid4().hex[:8]}_{chunk_idx}"
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    language=language,
                    strategy=ChunkingStrategyEnum.SENTENCE_SEMANTIC,
                    start_char=0,
                    end_char=len(chunk_text),
                    extra={
                        "sentence_count": len(current_group),
                        "word_count": current_word_count,
                        "chunk_index": chunk_idx,
                        **extra_metadata
                    }
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))
                current_group = []
                current_word_count = 0
                chunk_idx += 1

            current_group.append(s)
            current_word_count += s_words

        if current_group:
            chunk_text = " ".join(current_group)
            chunk_id = f"sent_{doc_id or uuid.uuid4().hex[:8]}_{chunk_idx}"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                language=language,
                strategy=ChunkingStrategyEnum.SENTENCE_SEMANTIC,
                start_char=0,
                end_char=len(chunk_text),
                extra={
                    "sentence_count": len(current_group),
                    "word_count": current_word_count,
                    "chunk_index": chunk_idx,
                    **extra_metadata
                }
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))

        return chunks
