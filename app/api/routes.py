"""API endpoints for Voice RAG, Text RAG, Semantic Search, Chunking Preview, Ingestion & Benchmarking."""

import time
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from app.models.schemas import (
    RAGRequest,
    RAGResponse,
    VoiceRAGResponse,
    SearchQueryRequest,
    SearchResponse,
    ChunkingStrategyEnum,
    DocumentChunk,
    IngestionRequest,
    IngestionResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    LatencyPercentiles,
    LatencyBreakdown,
)
from app.chunking import chunk_document
from app.services.vector_store import vector_store
from app.services.stt_service import stt_service
from app.core.harness import rag_harness

router = APIRouter()


@router.post("/query", response_model=RAGResponse, summary="Execute Text-based Multilingual RAG Query")
async def rag_query(request: RAGRequest):
    """Executes full RAG flow: Guardrails -> Embed -> FAISS Top-K -> Groq Llama 3.1 -> Hallucination Check."""
    try:
        response = rag_harness.execute_rag_pipeline(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")


@router.post("/voice", response_model=VoiceRAGResponse, summary="Execute Voice-Enabled Multilingual RAG")
async def rag_voice(
    file: UploadFile = File(..., description="Audio file (WAV, WebM, MP3, etc.)"),
    language: str = Form(default="hi-IN"),
    strategy: ChunkingStrategyEnum = Form(default=ChunkingStrategyEnum.METADATA_AWARE),
    top_k: int = Form(default=4)
):
    """
    Ingests voice audio recording -> Sarvam STT Transcription -> RAG Pipeline -> Structured Answer.
    """
    try:
        audio_bytes = await file.read()
        transcript, detected_lang, stt_latency = stt_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=file.filename or "recording.wav",
            language_code=language
        )

        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio input")

        rag_req = RAGRequest(
            query=transcript,
            language=detected_lang[:2] if detected_lang else "hi",
            strategy=strategy,
            top_k=top_k
        )
        rag_res = rag_harness.execute_rag_pipeline(rag_req)
        # Update STT attribution in latency breakdown
        rag_res.latency.stt_ms = stt_latency
        rag_res.latency.total_e2e_ms += stt_latency

        return VoiceRAGResponse(
            transcription=transcript,
            detected_language=detected_lang,
            rag=rag_res
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice RAG processing failed: {str(e)}")


@router.post("/search", response_model=SearchResponse, summary="Direct FAISS Semantic Search")
async def semantic_search(request: SearchQueryRequest):
    """Searches vector index with query string and optional strategy filter."""
    start_t = time.time()
    results = vector_store.search(
        query=request.query,
        top_k=request.top_k,
        strategy_filter=request.strategy,
        language_filter=request.language
    )
    elapsed = (time.time() - start_t) * 1000.0
    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
        latency_ms=elapsed
    )


@router.post("/chunking/preview", summary="Preview all 4 Chunking Strategies on input text")
async def preview_chunking(
    text: str = Form(...),
    doc_id: str = Form(default="preview_doc"),
    language: str = Form(default="hi")
):
    """Visualizes and compares how text is broken down by all 4 chunking strategies."""
    results = {}
    for strategy in ChunkingStrategyEnum:
        chunks = chunk_document(text=text, strategy=strategy, doc_id=doc_id, language=language)
        results[strategy.value] = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "word_count": len(c.text.split()),
                "metadata": c.metadata.model_dump()
            }
            for c in chunks
        ]
    return results


@router.post("/ingest", response_model=IngestionResponse, summary="Ingest passages into FAISS index")
async def ingest_documents(request: IngestionRequest):
    """Chunks and indexes arbitrary documents into the FAISS vector database."""
    start_t = time.time()
    all_chunks: List[DocumentChunk] = []

    for idx, doc in enumerate(request.documents):
        text = doc.get("text") or doc.get("passage_text") or doc.get("content", "")
        doc_id = str(doc.get("doc_id") or doc.get("id") or f"doc_{idx}")
        extra = {k: v for k, v in doc.items() if k not in ["text", "passage_text", "content"]}
        
        chunks = chunk_document(
            text=text,
            strategy=request.strategy,
            doc_id=doc_id,
            language=request.language,
            extra_metadata=extra
        )
        all_chunks.extend(chunks)

    vector_store.add_chunks(all_chunks)
    vector_store.save()
    elapsed = (time.time() - start_t) * 1000.0

    return IngestionResponse(
        status="success",
        total_documents=len(request.documents),
        total_chunks_created=len(all_chunks),
        index_size=vector_store.total_count,
        time_taken_ms=elapsed
    )


@router.post("/benchmark", response_model=BenchmarkResponse, summary="Benchmark Latency across sample queries (P50/P70/P100)")
async def run_benchmark(request: BenchmarkRequest):
    """Executes a series of benchmark queries to compute accurate P50, P70, P90, P99, P100 latency percentiles."""
    sample_queries = [
        "भारत की राजधानी क्या है?",
        "ताजमहल का निर्माण किसने करवाया था?",
        "सौर ऊर्जा के प्रमुख लाभ क्या हैं?",
        "कंप्यूटर मेमोरी के प्रकार बताएं।",
        "जल संरक्षण क्यों आवश्यक है?",
        "भारतीय संविधान कब लागू हुआ था?",
        "पेड़-पौधे प्रकाश संश्लेषण कैसे करते हैं?",
        "स्वस्थ जीवनशैली के नियम क्या हैं?",
        "अंतरिक्ष अनुसंधान का क्या महत्व है?",
        "योग और ध्यान के शारीरिक फायदे क्या हैं?"
    ]

    latencies = []
    stage_stt = []
    stage_guard_pre = []
    stage_retrieval = []
    stage_llm = []
    stage_guard_post = []

    start_bench = time.time()
    num_runs = request.num_queries

    for i in range(num_runs):
        q = sample_queries[i % len(sample_queries)]
        rag_req = RAGRequest(
            query=q,
            strategy=request.strategy,
            top_k=3
        )
        res = rag_harness.execute_rag_pipeline(rag_req)
        latencies.append(res.latency.total_e2e_ms)
        stage_guard_pre.append(res.latency.guardrails_pre_ms)
        stage_retrieval.append(res.latency.retrieval_ms)
        stage_llm.append(res.latency.llm_generation_ms)
        stage_guard_post.append(res.latency.guardrails_post_ms)

    lat_arr = np.array(latencies)
    total_bench_ms = (time.time() - start_bench) * 1000.0

    percentiles = LatencyPercentiles(
        p50=float(np.percentile(lat_arr, 50)),
        p70=float(np.percentile(lat_arr, 70)),
        p90=float(np.percentile(lat_arr, 90)),
        p99=float(np.percentile(lat_arr, 99)),
        p100=float(np.max(lat_arr)),
        mean=float(np.mean(lat_arr)),
        min=float(np.min(lat_arr)),
        max=float(np.max(lat_arr))
    )

    stage_avgs = {
        "guardrails_pre_avg_ms": float(np.mean(stage_guard_pre)),
        "retrieval_avg_ms": float(np.mean(stage_retrieval)),
        "llm_generation_avg_ms": float(np.mean(stage_llm)),
        "guardrails_post_avg_ms": float(np.mean(stage_guard_post)),
    }

    return BenchmarkResponse(
        num_queries_run=num_runs,
        strategy=request.strategy,
        total_time_ms=total_bench_ms,
        latency_percentiles=percentiles,
        stage_averages=stage_avgs
    )
