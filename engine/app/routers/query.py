from fastapi import APIRouter, HTTPException
from shared.schemas import QueryRequest, RAGAnswer
from app import vector_rag, graph_rag

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/vector/query", response_model=RAGAnswer)
def query_vector(req: QueryRequest):
    try:
        return vector_rag.answer(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/graph/query", response_model=RAGAnswer)
def query_graph(req: QueryRequest):
    try:
        return graph_rag.answer(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/graph_reranked/query", response_model=RAGAnswer)
def query_graph_reranked(req: QueryRequest):
    """Same 2-hop traversal as /graph/query, but chunk selection is reranked
    by a local Cross-Encoder (sentence-transformers) instead of an
    unordered-set slice. $0 cost, runs on CPU."""
    try:
        return graph_rag.answer(req.question, top_k=req.top_k, use_reranker=True)
    except Exception as e:
        raise HTTPException(500, str(e))
