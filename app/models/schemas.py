"""Pydantic schemas for structured I/O across all API endpoints and pipeline components."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkingStrategyEnum(str, Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE_SEMANTIC = "sentence_semantic"
    METADATA_AWARE = "metadata_aware"
    HIERARCHICAL = "hierarchical"


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    language: str = "hi"
    strategy: ChunkingStrategyEnum
    parent_id: Optional[str] = None
    start_char: Optional[int] = 0
    end_char: Optional[int] = 0
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Query in Hindi, English, or other Indic languages")
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: Optional[ChunkingStrategyEnum] = None
    language: Optional[str] = "hi"


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]
    latency_ms: float


class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    guardrails_pre_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_generation_ms: float = 0.0
    guardrails_post_ms: float = 0.0
    total_e2e_ms: float = 0.0


class GuardrailStatus(BaseModel):
    is_safe: bool = True
    is_on_topic: bool = True
    is_hallucination: bool = False
    flag_reason: Optional[str] = None


class RAGRequest(BaseModel):
    query: str = Field(..., description="User query text")
    language: str = Field(default="hi", description="Language code e.g. 'hi', 'en'")
    strategy: ChunkingStrategyEnum = Field(
        default=ChunkingStrategyEnum.METADATA_AWARE,
        description="Chunking strategy to prioritize"
    )
    top_k: int = Field(default=4, ge=1, le=10)
    stream: bool = False


class RAGResponse(BaseModel):
    query: str
    answer: str
    language: str
    retrieved_contexts: List[SearchResultItem]
    chunking_strategy: ChunkingStrategyEnum
    guardrails: GuardrailStatus
    latency: LatencyBreakdown
    model_used: str
    groundedness_score: float = 1.0


class VoiceRAGResponse(BaseModel):
    transcription: str
    detected_language: str
    rag: RAGResponse


class IngestionRequest(BaseModel):
    documents: List[Dict[str, Any]]
    language: str = "hi"
    strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.METADATA_AWARE


class IngestionResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks_created: int
    index_size: int
    time_taken_ms: float


class BenchmarkRequest(BaseModel):
    num_queries: int = Field(default=20, ge=5, le=100)
    strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.METADATA_AWARE


class LatencyPercentiles(BaseModel):
    p50: float
    p70: float
    p90: float
    p99: float
    p100: float
    mean: float
    min: float
    max: float


class BenchmarkResponse(BaseModel):
    num_queries_run: int
    strategy: ChunkingStrategyEnum
    total_time_ms: float
    latency_percentiles: LatencyPercentiles
    stage_averages: Dict[str, float]
