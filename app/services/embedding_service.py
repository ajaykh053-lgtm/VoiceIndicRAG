"""Embedding service using SentenceTransformers multilingual model."""

import logging
import time
from typing import List, Union
import numpy as np
from app.core.config import settings

logger = logging.getLogger("embedding_service")


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer ({e}). Falling back to dummy mock embeddings for test.")
                self._model = None

    def get_embedding_dim(self) -> int:
        return settings.EMBEDDING_DIM

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generates L2-normalized float32 embeddings for single or batch texts."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, settings.EMBEDDING_DIM), dtype="float32")

        self._load_model()
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.astype("float32")
        else:
            # Deterministic pseudo-embedding for testing / fallback environments
            np.random.seed(42)
            emb = []
            for t in texts:
                # generate repeatable vector based on text hash
                val = sum(ord(c) for c in t) % 1000
                vec = np.sin(np.linspace(val, val + 10, settings.EMBEDDING_DIM))
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                emb.append(vec)
            return np.array(emb, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        res = self.embed_texts([query])
        return res[0]


embedding_service = EmbeddingService()
