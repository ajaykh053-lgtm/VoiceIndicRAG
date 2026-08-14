"""Ingest live batches directly from Hugging Face ai4bharat/MSMARCO-XI dataset into FAISS index."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.chunking import chunk_document
from app.models.schemas import ChunkingStrategyEnum
from app.services.vector_store import vector_store


def ingest_from_huggingface(language: str = "hi", max_samples: int = 100, strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.METADATA_AWARE):
    print(f"=== Streaming from Hugging Face: ai4bharat/MSMARCO-XI (lang={language}, max={max_samples}) ===")
    try:
        from datasets import load_dataset
        dataset = load_dataset("ai4bharat/MSMARCO-XI", language, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading Hugging Face dataset: {e}")
        return

    all_chunks = []
    count = 0
    start_t = time.time()

    for item in dataset:
        passages = item.get("passages", [])
        query_id = item.get("query_id", f"q_{count}")
        query_text = item.get("query", "")

        # Extract passages
        for p_idx, p in enumerate(passages):
            p_text = p.get("passage_text", "") if isinstance(p, dict) else str(p)
            if not p_text.strip():
                continue

            chunks = chunk_document(
                text=p_text,
                strategy=strategy,
                doc_id=f"msmarco_{query_id}_{p_idx}",
                language=language,
                extra_metadata={
                    "query_id": query_id,
                    "query_text": query_text,
                    "is_selected": p.get("is_selected", 0) if isinstance(p, dict) else 0
                }
            )
            all_chunks.extend(chunks)

        count += 1
        if count % 20 == 0:
            print(f"  Streamed {count}/{max_samples} examples ({len(all_chunks)} chunks generated)...")

        if count >= max_samples:
            break

    print(f"Embedding and indexing {len(all_chunks)} chunks into FAISS vector store...")
    vector_store.add_chunks(all_chunks)
    vector_store.save()
    elapsed = time.time() - start_t
    print(f"[SUCCESS] Ingested {count} MSMARCO-XI items ({len(all_chunks)} chunks). Total FAISS Index Size: {vector_store.total_count} (took {elapsed:.2f}s)")


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "hi"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ingest_from_huggingface(language=lang, max_samples=limit)
