"""Script to benchmark end-to-end and stage-by-stage RAG latency across 50 test queries (P50/P70/P90/P100)."""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.models.schemas import RAGRequest, ChunkingStrategyEnum
from app.core.harness import rag_harness
from app.services.vector_store import vector_store
from scripts.build_index import build_index


def run_benchmark_suite(num_queries: int = 50):
    print("=== Starting Voice/Text RAG Latency Benchmark ===")
    if vector_store.total_count == 0:
        print("Vector store is empty, building initial index...")
        build_index()

    test_queries = [
        "भारत की राजधानी क्या है?",
        "ताजमहल किस शहर में स्थित है और इसे किसने बनवाया?",
        "सौर ऊर्जा के प्रमुख लाभ क्या हैं?",
        "कंप्यूटर मेमोरी में RAM और ROM में क्या अंतर है?",
        "वर्षा जल संचयन और जल संरक्षण क्यों आवश्यक है?",
        "भारतीय संविधान कब लागू हुआ था और इसके मुख्य निर्माता कौन थे?",
        "What is Retrieval-Augmented Generation (RAG)?",
        "How do solar panels convert sunlight to electricity?"
    ]

    total_e2e_latencies = []
    guard_pre_latencies = []
    retrieval_latencies = []
    llm_latencies = []
    guard_post_latencies = []

    print(f"Warming up model & FAISS index with 1 sample query...")
    rag_harness.execute_rag_pipeline(RAGRequest(query="warmup", strategy=ChunkingStrategyEnum.METADATA_AWARE, top_k=1))

    print(f"Running {num_queries} queries across 4 chunking strategies...")
    strategies = list(ChunkingStrategyEnum)

    start_suite = time.time()
    for i in range(num_queries):
        q = test_queries[i % len(test_queries)]
        strat = strategies[i % len(strategies)]

        req = RAGRequest(
            query=q,
            strategy=strat,
            top_k=3
        )
        res = rag_harness.execute_rag_pipeline(req)

        total_e2e_latencies.append(res.latency.total_e2e_ms)
        guard_pre_latencies.append(res.latency.guardrails_pre_ms)
        retrieval_latencies.append(res.latency.retrieval_ms)
        llm_latencies.append(res.latency.llm_generation_ms)
        guard_post_latencies.append(res.latency.guardrails_post_ms)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries...")

    total_time = time.time() - start_suite
    e2e_arr = np.array(total_e2e_latencies)

    p50 = float(np.percentile(e2e_arr, 50))
    p70 = float(np.percentile(e2e_arr, 70))
    p90 = float(np.percentile(e2e_arr, 90))
    p99 = float(np.percentile(e2e_arr, 99))
    p100 = float(np.max(e2e_arr))
    mean = float(np.mean(e2e_arr))

    print("\n============================================================")
    print("           LATENCY BENCHMARK RESULTS (P50 / P70 / P100)      ")
    print("============================================================")
    print(f" Total Queries Executed:  {num_queries}")
    print(f" Total Benchmark Time:    {total_time:.2f} s")
    print(f" Mean Latency:            {mean:.2f} ms")
    print(f" Min Latency:             {np.min(e2e_arr):.2f} ms")
    print(f" P50 Latency (Median):    {p50:.2f} ms")
    print(f" P70 Latency:             {p70:.2f} ms")
    print(f" P90 Latency:             {p90:.2f} ms")
    print(f" P99 Latency:             {p99:.2f} ms")
    print(f" P100 Latency (Max):      {p100:.2f} ms")
    print("------------------------------------------------------------")
    print(" Stage Breakdown (Averages):")
    print(f"  - Guardrails Pre-Check: {np.mean(guard_pre_latencies):.2f} ms")
    print(f"  - Vector Retrieval:     {np.mean(retrieval_latencies):.2f} ms")
    print(f"  - LLM Generation:       {np.mean(llm_latencies):.2f} ms")
    print(f"  - Guardrails Post-Check:{np.mean(guard_post_latencies):.2f} ms")
    print("============================================================\n")

    # Latency target check (< 4000ms)
    if p100 < 4000:
        print("[SUCCESS] TARGET ACHIEVED: All requests completed under the 4.0s SLA requirement!")
    else:
        print("[WARNING] Note: Some requests exceeded 4.0s.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark_suite(n)
