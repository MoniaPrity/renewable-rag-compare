"""
engine/app/routers/stats.py  (NEW)

Serves scripts/06_statistical_analysis.py's output (stats_results.json) to
the frontend's /stats dashboard page. Read-only, $0 cost — just reads a
JSON file already produced locally by scipy/numpy, no new computation.

Wire up in app/main.py:
    from app.routers import stats
    app.include_router(stats.router)
"""
import json
from fastapi import APIRouter, HTTPException
from app.config import PROCESSED_DIR

router = APIRouter(prefix="/api/stats", tags=["stats"])

STATS_PATH = PROCESSED_DIR / "stats_results.json"


@router.get("")
def get_stats():
    """
    Returns the full stats_results.json contents: {"overall": {...},
    "single_hop": {...}, "multi_hop": {...}} — each with bootstrap_ci and
    wilcoxon sections, exactly as scripts/06_statistical_analysis.py wrote
    them. 404s with a helpful message if the script hasn't been run yet,
    so the frontend can render a clear "run this command" hint instead of
    a raw fetch failure.
    """
    if not STATS_PATH.exists():
        raise HTTPException(
            404,
            "No statistical analysis found. Run "
            "'python scripts/06_statistical_analysis.py' in the engine "
            "folder (after 04_run_evaluation.py) to generate stats_results.json.",
        )
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))
