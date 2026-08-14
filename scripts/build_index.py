"""Script to build and persist FAISS index from documents using the 4 chunking strategies."""

import json
import time
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.core.config import settings
from app.chunking import chunk_document
from app.models.schemas import ChunkingStrategyEnum
from app.services.vector_store import vector_store
from app.main import load_initial_samples_if_empty


def build_index(samples_file: Path = None):
    samples_file = samples_file or settings.SAMPLES_DATA_PATH
    if not samples_file.exists():
        print(f"Sample file {samples_file} not found. Creating default sample dataset...")
        load_initial_samples_if_empty()
        return

    with open(samples_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents. Chunking across all 4 strategies...")
    start_t = time.time()
    all_chunks = []

    strategies = [
        ChunkingStrategyEnum.FIXED_SIZE,
        ChunkingStrategyEnum.SENTENCE_SEMANTIC,
        ChunkingStrategyEnum.METADATA_AWARE,
        ChunkingStrategyEnum.HIERARCHICAL
    ]

    for doc in docs:
        text = doc.get("passage_text") or doc.get("text", "")
        doc_id = doc.get("doc_id") or "doc"
        lang = doc.get("language", "hi")
        extra = {"title": doc.get("title", "")}

        for strat in strategies:
            chunks = chunk_document(
                text=text,
                strategy=strat,
                doc_id=doc_id,
                language=lang,
                extra_metadata=extra
            )
            all_chunks.extend(chunks)

    print(f"Generated {len(all_chunks)} total chunks. Embedding and adding to FAISS...")
    vector_store.add_chunks(all_chunks)
    vector_store.save()

    elapsed = time.time() - start_t
    print(f"[SUCCESS] FAISS index successfully built with {vector_store.total_count} chunks in {elapsed:.2f}s.")


if __name__ == "__main__":
    build_index()
