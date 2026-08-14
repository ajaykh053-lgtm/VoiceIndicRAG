"""FAISS Vector Store with metadata persistence and strategy filtering."""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.models.schemas import DocumentChunk, ChunkMetadata, SearchResultItem, ChunkingStrategyEnum
from app.services.embedding_service import embedding_service

logger = logging.getLogger("vector_store")


class FaissVectorStore:
    def __init__(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.METADATA_PATH
        self.dim = settings.EMBEDDING_DIM
        self.index = None
        self.metadata_list: List[Dict[str, Any]] = []
        self._faiss_module = None
        self._load_faiss()
        self.load()

    def _load_faiss(self):
        try:
            import faiss
            self._faiss_module = faiss
        except ImportError:
            logger.warning("FAISS not installed or failed to import. In-memory numpy fallback will be used.")
            self._faiss_module = None

    def _init_new_index(self):
        if self._faiss_module:
            # Inner Product index on normalized embeddings equals Cosine Similarity
            self.index = self._faiss_module.IndexFlatIP(self.dim)
        else:
            self.index = None
        self.metadata_list = []

    def load(self) -> bool:
        """Loads FAISS index and metadata JSON from disk if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                if self._faiss_module:
                    self.index = self._faiss_module.read_index(str(self.index_path))
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata_list = json.load(f)
                logger.info(f"Loaded FAISS index with {len(self.metadata_list)} items from {self.index_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading index: {e}. Initializing clean index.")
        self._init_new_index()
        return False

    def save(self):
        """Persists FAISS index and metadata JSON to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        if self._faiss_module and self.index is not None:
            self._faiss_module.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index with {len(self.metadata_list)} items.")

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Embeds and indexes a list of DocumentChunks."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_texts(texts)

        if self._faiss_module and self.index is None:
            self._init_new_index()

        if self._faiss_module and self.index is not None:
            self.index.add(embeddings)
        
        # Store serialized metadata
        for c in chunks:
            self.metadata_list.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata.model_dump()
            })

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy_filter: Optional[ChunkingStrategyEnum] = None,
        language_filter: Optional[str] = None
    ) -> List[SearchResultItem]:
        """Performs vector search, with optional strategy/language filtering and parent context resolving."""
        if not self.metadata_list:
            return []

        query_vec = embedding_service.embed_query(query).reshape(1, -1)
        k_fetch = min(max(top_k * 4, 10), len(self.metadata_list))

        results: List[SearchResultItem] = []

        if self._faiss_module and self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query_vec, k_fetch)
            candidate_pairs = zip(indices[0], scores[0])
        else:
            # Fallback cosine calculation
            all_texts = [m["text"] for m in self.metadata_list]
            all_embs = embedding_service.embed_texts(all_texts)
            sims = np.dot(all_embs, query_vec.T).squeeze()
            top_indices = np.argsort(-sims)[:k_fetch]
            candidate_pairs = [(idx, float(sims[idx])) for idx in top_indices]

        # Build lookup for parent passages if hierarchical
        parent_map = {}
        for m in self.metadata_list:
            if m["metadata"].get("strategy") == ChunkingStrategyEnum.HIERARCHICAL.value:
                if m["metadata"].get("extra", {}).get("is_parent"):
                    parent_map[m["chunk_id"]] = m["text"]

        seen_chunks = set()
        for idx, score in candidate_pairs:
            if idx < 0 or idx >= len(self.metadata_list):
                continue
            item = self.metadata_list[idx]
            chunk_id = item["chunk_id"]
            if chunk_id in seen_chunks:
                continue

            meta_dict = item["metadata"]
            item_strategy = meta_dict.get("strategy")
            item_lang = meta_dict.get("language")

            # Filters
            if strategy_filter and item_strategy != strategy_filter.value:
                continue
            if language_filter and item_lang and item_lang != language_filter:
                # Allow cross-lingual retrieval unless strictly filtered
                pass

            display_text = item["text"]
            # If child chunk in hierarchical strategy, provide parent context for rich answer generation
            if item_strategy == ChunkingStrategyEnum.HIERARCHICAL.value:
                parent_id = meta_dict.get("parent_id")
                if parent_id and parent_id in parent_map:
                    display_text = f"[Parent Passage Context: {parent_map[parent_id]}]\nMatch sentence: {item['text']}"

            meta_obj = ChunkMetadata(**meta_dict)
            results.append(
                SearchResultItem(
                    chunk_id=chunk_id,
                    text=display_text,
                    score=float(score),
                    metadata=meta_obj
                )
            )
            seen_chunks.add(chunk_id)
            if len(results) >= top_k:
                break

        return results

    @property
    def total_count(self) -> int:
        return len(self.metadata_list)


vector_store = FaissVectorStore()
