"""Strategy 4: Hierarchical Parent-Child Chunking.
Indexes fine-grained child chunks (sentences) for maximum retrieval precision,
while storing and linking the parent passage context for full LLM answering.
"""

import uuid
from typing import List, Optional, Dict, Any
from app.models.schemas import DocumentChunk, ChunkMetadata, ChunkingStrategyEnum
from app.chunking.sentence_semantic import SentenceSemanticChunker


class HierarchicalChunker:
    """Creates parent chunks (full passage) and associated child chunks (sentences)."""

    def __init__(self, parent_max_words: int = 400, child_sentence_group: int = 1):
        self.parent_max_words = parent_max_words
        self.sentence_chunker = SentenceSemanticChunker(target_sentences=child_sentence_group)

    def chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        language: str = "hi",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Returns both parent chunk and child chunks linked via parent_id."""
        text = text.strip()
        if not text:
            return []

        extra_metadata = extra_metadata or {}
        parent_id = f"parent_{doc_id or uuid.uuid4().hex[:8]}"

        # 1. Create Parent Chunk
        parent_meta = ChunkMetadata(
            chunk_id=parent_id,
            doc_id=doc_id,
            language=language,
            strategy=ChunkingStrategyEnum.HIERARCHICAL,
            parent_id=None,
            start_char=0,
            end_char=len(text),
            extra={"is_parent": True, "parent_text": text, **extra_metadata}
        )
        parent_chunk = DocumentChunk(chunk_id=parent_id, text=text, metadata=parent_meta)

        # 2. Create Child Chunks (indexed for dense vector matching)
        child_sentences = self.sentence_chunker.split_into_sentences(text)
        all_chunks: List[DocumentChunk] = [parent_chunk]

        for c_idx, s in enumerate(child_sentences):
            child_id = f"child_{parent_id}_{c_idx}"
            child_meta = ChunkMetadata(
                chunk_id=child_id,
                doc_id=doc_id,
                language=language,
                strategy=ChunkingStrategyEnum.HIERARCHICAL,
                parent_id=parent_id,
                start_char=0,
                end_char=len(s),
                extra={
                    "is_parent": False,
                    "parent_id": parent_id,
                    "parent_text": text,
                    "child_index": c_idx,
                    **extra_metadata
                }
            )
            all_chunks.append(DocumentChunk(chunk_id=child_id, text=s, metadata=child_meta))

        return all_chunks
