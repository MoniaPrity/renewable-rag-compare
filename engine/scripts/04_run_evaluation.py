"""
Runs every question in eval_questions.json through THREE pipelines —
Vector RAG, raw Graph RAG, and Cross-Encoder-Reranked Graph RAG — judges each
answer with the LLM, and saves aggregate + per-question results to
engine/data/processed/eval_results.json, which the Next.js /compare
dashboard reads via GET /api/compare/history.

CHANGES vs original:
  - Checkpointing: results are written to disk after EVERY question, not
    just at the end. If a Groq rate-limit error kills the run at question
    30/40, you keep questions 1-29 instead of losing the whole batch.
  - Resume: if eval_results.json already has results for some questions
    (matched by exact question text), those are skipped on the next run
    instead of re-spending API calls on them. Use --restart to force a
    full re-run from scratch.
  - Per-question error handling: if compare_question() raises (e.g. Groq
    outage, malformed judge JSON after retries exhausted), that question is
    logged as failed and skipped, and the run continues instead of dying.

NOTE: This script assumes app/evaluation.py's compare_question(q) has been
updated to also call graph_rag.answer(q, use_reranker=True) and populate
CompareResult.graph_reranked_answer / graph_reranked_judge (see schemas.py).

Run from the engine/ folder:
    python scripts/04_run_evaluation.py
    python scripts/04_run_evaluation.py --restart      # ignore existing checkpoint
    python scripts/04_run_evaluation.py --delay 2.0     # slower pacing if you hit 429s
"""
import argparse
import time
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PROCESSED_DIR, settings
from app import evaluation

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
OUT_PATH = PROCESSED_DIR / "eval_results.json"

METHODS = ("vector", "graph", "graph_reranked")


def summarize(results: list[dict]) -> dict:
    def avg(method: str, field: str) -> float:
        vals = [r[f"{method}_judge"][field] for r in results]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def avg_latency(method: str) -> float:
        vals = [r[f"{method}_answer"]["latency_seconds"] for r in results]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {}
    for method in METHODS:
        summary[method] = {
            "avg_groundedness": avg(method, "groundedness"),
            "avg_relevance": avg(method, "relevance"),
            "avg_completeness": avg(method, "completeness"),
            "avg_latency_seconds": avg_latency(method),
        }
    return summary


def summarize_by_category(results: list[dict], questions_meta: list[dict]) -> dict:
    """
    Splits scores by single-hop vs multi-hop so you can directly answer
    "does reranking close the multi-hop gap?". Requires each entry in
    eval_questions.json to be an object with a "category" key. If your
    eval_questions.json is a flat list of strings, this returns {} and is
    skipped.
    """
    if not questions_meta or not isinstance(questions_meta[0], dict):
        return {}

    cat_by_question = {m["question"]: m.get("category", "uncategorized") for m in questions_meta}
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        cat = cat_by_question.get(r["question"], "uncategorized")
        by_cat.setdefault(cat, []).append(r)

    return {cat: summarize(rs) for cat, rs in by_cat.items()}


def save_checkpoint(results: list[dict], questions_meta: list[dict]) -> None:
    summary = summarize(results)
    category_summary = summarize_by_category(results, questions_meta)
    output = {"results": results, "summary": summary}
    if category_summary:
        output["summary_by_category"] = category_summary
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true",
                         help="Ignore any existing eval_results.json checkpoint and start fresh")
    parser.add_argument("--delay", type=float, default=0.5,
                         help="Extra pause (seconds) between questions, on top of any "
                              "intra-question delay inside evaluation.py")
    args = parser.parse_args()

    print(f"[config] Active GROQ_MODEL for this run: {settings.groq_model}")

    raw_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    # Support both a flat list of strings (legacy) and a list of
    # {"question": ..., "category": "single_hop"|"multi_hop"} objects.
    if raw_questions and isinstance(raw_questions[0], dict):
        questions = [q["question"] for q in raw_questions]
        questions_meta = raw_questions
    else:
        questions = raw_questions
        questions_meta = []

    results: list[dict] = []
    already_done: set[str] = set()

    if not args.restart and OUT_PATH.exists():
        try:
            prior = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            results = prior.get("results", [])
            already_done = {r["question"] for r in results}
            if already_done:
                print(f"[resume] Found existing checkpoint with {len(already_done)} "
                      f"completed questions — skipping those. Use --restart to redo everything.")
        except (json.JSONDecodeError, KeyError):
            print("[warn] Existing eval_results.json was unreadable — starting fresh.")

    failed_questions = []
    remaining = [q for q in questions if q not in already_done]
    print(f"Running {len(remaining)} of {len(questions)} questions "
          f"({len(already_done)} already completed)\n")

    for i, q in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {q}")
        try:
            result = evaluation.compare_question(q)
            results.append(json.loads(result.model_dump_json()))
        except Exception as e:
            print(f"  [FAILED] {type(e).__name__}: {e}")
            failed_questions.append(q)
            continue
        finally:
            # Checkpoint after every question, success or failure, so a crash
            # never loses more than the single in-flight question.
            save_checkpoint(results, questions_meta)
        time.sleep(args.delay)

    summary = summarize(results)
    category_summary = summarize_by_category(results, questions_meta)

    print("\n=== Summary (all questions) ===")
    print(json.dumps(summary, indent=2))
    if category_summary:
        print("\n=== Summary by category (single_hop vs multi_hop) ===")
        print(json.dumps(category_summary, indent=2))
    else:
        print(
            "\n[note] eval_questions.json is a flat list of strings, so no "
            "single-hop/multi-hop split was computed. Convert entries to "
            '{"question": "...", "category": "single_hop"|"multi_hop"} to get that split.'
        )

    if failed_questions:
        print(f"\n[warn] {len(failed_questions)} question(s) failed and were skipped:")
        for q in failed_questions:
            print(f"  - {q}")
        print("Re-run this script (without --restart) to retry just the failed/missing ones.")

    print(f"\nSaved full results to {OUT_PATH}")
