"""
Adds statistical rigor on top of eval_results.json — entirely local, no API
calls, $0 cost. For every method (vector / graph / graph_reranked) and every
judge metric (groundedness / relevance / completeness):

  1. Bootstrap 95% confidence interval on the mean score (10,000 resamples).
  2. Paired Wilcoxon signed-rank test between method pairs (graph vs
     graph_reranked is the one your research question hinges on), since all
     methods answer the SAME questions — a paired test is the right tool,
     not an unpaired t-test.

Runs this overall AND split by category (single_hop / multi_hop) if
eval_questions.json has categories, so you can see whether "Reranked Graph
RAG beats Raw Graph RAG on multi-hop" is actually statistically supported
yet, or just noise from a small sample.

Requires: pip install scipy   (numpy is already a transitive dependency of
sentence-transformers, but add both to requirements.txt to be safe)

Run from the engine/ folder, after 04_run_evaluation.py:
    python scripts/06_statistical_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PROCESSED_DIR

RESULTS_PATH = PROCESSED_DIR / "eval_results.json"
QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
OUT_PATH = PROCESSED_DIR / "stats_results.json"

METHODS = ("vector", "graph", "graph_reranked")
METRICS = ("groundedness", "relevance", "completeness")
PAIRS = [("graph", "graph_reranked"), ("vector", "graph"), ("vector", "graph_reranked")]

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
SEED = 42


def bootstrap_ci(values: list[float], n_boot: int = N_BOOTSTRAP, ci: float = CI_LEVEL, seed: int = SEED):
    if not values:
        return {"mean": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return {
        "mean": round(float(arr.mean()), 3),
        "ci_95_low": round(float(lo), 3),
        "ci_95_high": round(float(hi), 3),
        "n": len(arr),
    }


def paired_wilcoxon(a_vals: list[float], b_vals: list[float]):
    diffs = np.array(a_vals) - np.array(b_vals)
    if np.all(diffs == 0):
        return {"statistic": None, "p_value": None,
                "note": "identical scores on every question — test undefined"}
    try:
        stat, p = stats.wilcoxon(a_vals, b_vals)
        return {"statistic": round(float(stat), 3), "p_value": round(float(p), 4)}
    except ValueError as e:
        return {"statistic": None, "p_value": None, "note": str(e)}


def load_category_map() -> dict[str, str]:
    raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if raw and isinstance(raw[0], dict):
        return {q["question"]: q.get("category", "uncategorized") for q in raw}
    return {}


def analyze(results: list[dict], label: str) -> dict:
    report = {"label": label, "n": len(results), "bootstrap_ci": {}, "wilcoxon": {}}
    if not results:
        return report

    for method in METHODS:
        report["bootstrap_ci"][method] = {
            metric: bootstrap_ci([r[f"{method}_judge"][metric] for r in results])
            for metric in METRICS
        }

    for m1, m2 in PAIRS:
        key = f"{m1}_vs_{m2}"
        report["wilcoxon"][key] = {}
        for metric in METRICS:
            a = [r[f"{m1}_judge"][metric] for r in results]
            b = [r[f"{m2}_judge"][metric] for r in results]
            report["wilcoxon"][key][metric] = paired_wilcoxon(a, b)

    return report


def print_report(section: dict) -> None:
    print(f"\n=== {section['label']} (n={section['n']}) ===")
    if section["n"] == 0:
        print("  (no results)")
        return

    print("\n-- Bootstrap 95% CI on mean judge scores --")
    for method, metrics in section["bootstrap_ci"].items():
        print(f"  {method}:")
        for metric, v in metrics.items():
            print(f"    {metric:14s} mean={v['mean']:.2f}  95% CI=[{v['ci_95_low']:.2f}, {v['ci_95_high']:.2f}]")

    print("\n-- Paired Wilcoxon signed-rank test (same questions, different method) --")
    for pair, metrics in section["wilcoxon"].items():
        print(f"  {pair}:")
        for metric, v in metrics.items():
            if v["p_value"] is not None:
                sig = "SIGNIFICANT (p<0.05)" if v["p_value"] < 0.05 else "not significant"
                print(f"    {metric:14s} p={v['p_value']:.4f}  -> {sig}")
            else:
                print(f"    {metric:14s} {v.get('note')}")


if __name__ == "__main__":
    if not RESULTS_PATH.exists():
        print(f"No results found at {RESULTS_PATH}. Run scripts/04_run_evaluation.py first.")
        sys.exit(1)

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results = data["results"]
    cat_map = load_category_map()

    output = {"overall": analyze(results, "All questions")}

    if cat_map:
        by_cat: dict[str, list[dict]] = {}
        for r in results:
            cat = cat_map.get(r["question"], "uncategorized")
            by_cat.setdefault(cat, []).append(r)
        for cat, rs in by_cat.items():
            output[cat] = analyze(rs, cat)
    else:
        print("[note] eval_questions.json has no 'category' field — skipping per-category breakdown.")

    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print_report(output["overall"])
    for cat, section in output.items():
        if cat != "overall":
            print_report(section)

    print(f"\nSaved full stats to {OUT_PATH}")
    print(
        "\nReminder: with n < ~20 per category, Wilcoxon p-values are usually "
        "underpowered to detect anything but large effects. Treat 'not significant' "
        "results here as 'not yet distinguishable from noise', not 'no effect' — "
        "run scripts/05_generate_questions.py to grow the eval set before concluding."
    )
