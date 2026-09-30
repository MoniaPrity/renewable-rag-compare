"""
app/reranker.py

Local, free Cross-Encoder reranker for the Graph RAG pipeline.

Why a Cross-Encoder and not just cosine similarity again:
  - The vector index already uses a bi-encoder (all-MiniLM-L6-v2) to embed
    query and chunks SEPARATELY, then compares vectors. That's what Vector
    RAG already does, so reusing it here wouldn't isolate the graph-specific
    fix.
  - A Cross-Encoder instead feeds (query, passage) TOGETHER through one
    small transformer and outputs a single relevance score. It's slower
    per-pair but far more accurate for "is this passage relevant to this
    exact question" — which is precisely the ranking Graph RAG is missing
    (root cause b: unordered set -> arbitrary top_k).
  - cross-encoder/ms-marco-MiniLM-L-6-v2 is ~80MB, CPU-friendly, and was
    literally trained for passage reranking (MS MARCO). Free, runs locally,
    no API call, no cost, no rate limit.

Usage:
    from app.reranker import rerank
    ranked = rerank(question, [(chunk_id, chunk_text), ...], top_k=5)
    # -> [(chunk_id, chunk_text, score), ...] sorted best-first
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_cache: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazy-load and cache the reranker model. Loaded once per process."""
    global _reranker_cache
    if _reranker_cache is None:
        print(f"[reranker] loading {RERANKER_MODEL_NAME} (first call only, "
              f"downloads ~80MB from Hugging Face if not cached)...")
        _reranker_cache = CrossEncoder(RERANKER_MODEL_NAME)
        print("[reranker] model ready.")
    return _reranker_cache


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """
    Score every (query, chunk_text) pair with the Cross-Encoder and return
    the top_k candidates sorted by relevance score, descending.

    Args:
        query: the user's question
        candidates: list of (chunk_id, chunk_text) — the 2-hop candidate pool
        top_k: how many to keep after reranking

    Returns:
        list of (chunk_id, chunk_text, score), length <= top_k, best first.
        Empty list in, empty list out (no crash on empty candidate pool).
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [[query, text] for _, text in candidates]
    raw_scores = model.predict(pairs)  # numpy array of floats (unbounded logits)

    scored = [
        (cid, text, float(score))
        for (cid, text), score in zip(candidates, raw_scores)
    ]
    scored.sort(key=lambda item: -item[2])
    return scored[:top_k]
