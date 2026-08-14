"""Chunking strategies module providing unified factory for 4 chunking strategies."""

from typing import List, Optional, Dict, Any
from app.models.schemas import ChunkingStrategyEnum, DocumentChunk
from .fixed_size import FixedSizeChunker
from .sentence_semantic import SentenceSemanticChunker
from .metadata_aware import MetadataAwareChunker
from .hierarchical import HierarchicalChunker


def get_chunker(strategy: ChunkingStrategyEnum):
    """Factory returning the requested chunker instance."""
    if strategy == ChunkingStrategyEnum.FIXED_SIZE:
        return FixedSizeChunker(chunk_size=256, chunk_overlap=64)
    elif strategy == ChunkingStrategyEnum.SENTENCE_SEMANTIC:
        return SentenceSemanticChunker(target_sentences=3, max_words=200)
    elif strategy == ChunkingStrategyEnum.METADATA_AWARE:
        return MetadataAwareChunker(chunk_size=200, include_header_in_text=True)
    elif strategy == ChunkingStrategyEnum.HIERARCHICAL:
        return HierarchicalChunker(parent_max_words=400, child_sentence_group=1)
    else:
        return MetadataAwareChunker(chunk_size=200, include_header_in_text=True)


def chunk_document(
    text: str,
    strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.METADATA_AWARE,
    doc_id: Optional[str] = None,
    language: str = "hi",
    extra_metadata: Optional[Dict[str, Any]] = None
) -> List[DocumentChunk]:
    """Helper to chunk text using the specified strategy."""
    chunker = get_chunker(strategy)
    return chunker.chunk(text=text, doc_id=doc_id, language=language, extra_metadata=extra_metadata)


__all__ = [
    "FixedSizeChunker",
    "SentenceSemanticChunker",
    "MetadataAwareChunker",
    "HierarchicalChunker",
    "get_chunker",
    "chunk_document",
]
