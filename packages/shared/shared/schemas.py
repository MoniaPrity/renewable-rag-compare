"""
Shared data contracts.

Both the FastAPI engine and the offline scripts (01_collect_corpus.py,
02_build_vector_index.py, 03_build_graph_index.py, 04_run_evaluation.py)
import from this package so the shape of a "chunk", an "answer", etc. is
defined exactly once. This is the "shared data package" of the monorepo.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single unit of retrievable text, produced by the ingestion scripts."""
    chunk_id: str
    source: str  # e.g. "wikipedia:Solar_power" or "arxiv:2401.01234"
    title: str
    text: str
    order: int = 0  # position of this chunk within its source document


class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    score: float = 0.0  # similarity score (vector), or 1.0 (raw graph), or cross-encoder score (reranked graph)
    source: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = 5


class RAGAnswer(BaseModel):
    # "graph_reranked" added: same 2-hop candidate pool as "graph", but the
    # chunk selection step reranks candidates with a local Cross-Encoder
    # instead of taking an arbitrary unordered-set slice. See app/graph_rag.py.
    method: Literal["vector", "graph", "graph_reranked"]
    question: str
    answer: str
    contexts: list[RetrievedContext]
    latency_seconds: float
    extra: dict = Field(default_factory=dict)  # e.g. matched graph entities


class JudgeScore(BaseModel):
    groundedness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    rationale: str = ""


class CompareResult(BaseModel):
    question: str
    vector_answer: RAGAnswer
    graph_answer: RAGAnswer
    graph_reranked_answer: RAGAnswer
    vector_judge: JudgeScore
    graph_judge: JudgeScore
    graph_reranked_judge: JudgeScore
