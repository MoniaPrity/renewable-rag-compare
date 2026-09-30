import json
from fastapi import APIRouter, HTTPException
from shared.schemas import QueryRequest, CompareResult
from app import evaluation
from app.config import PROCESSED_DIR

router = APIRouter(prefix="/api/compare", tags=["compare"])

EVAL_RESULTS_PATH = PROCESSED_DIR / "eval_results.json"


@router.post("", response_model=CompareResult)
def compare(req: QueryRequest):
    try:
        return evaluation.compare_question(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/history")
def history():
    """Returns the batch evaluation results produced by
    scripts/04_run_evaluation.py, for the frontend dashboard."""
    if not EVAL_RESULTS_PATH.exists():
        return {"results": [], "summary": None}
    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    return data
