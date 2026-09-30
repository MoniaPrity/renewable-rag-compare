"""
Auto-generates additional eval questions directly from the knowledge graph —
NO LLM calls, $0 cost. Templates 2-hop paths into multi-hop questions and
high-degree single nodes into single-hop questions, then appends new,
non-duplicate entries to eval_questions.json in the existing
{"question": ..., "category": ...} format.

Run from the engine/ folder:
    python scripts/05_generate_questions.py
    python scripts/05_generate_questions.py --single-hop-target 15 --multi-hop-target 25

Targets are TOTAL desired counts (existing + new), not "how many to add".
Safe to re-run: it will never duplicate an existing question string, and it
will just top up whatever's still missing.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.graph_rag import load_graph, MIN_ENTITY_NAME_LEN, _load_chunk_texts
from app.domain_filters import is_relevant_node

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
SEED = 42

SINGLE_HOP_TEMPLATES = [
    "What is {name} and what role does it play in renewable energy?",
    "How does {name} function within renewable energy systems?",
    "What are the main characteristics or benefits of {name}?",
    "Why is {name} significant in the context of renewable energy?",
]

MULTI_HOP_TEMPLATES = [
    "How does {a} relate to {c}, given its connection to {b}?",
    "What is the relationship between {a} and {c} through {b}?",
    "Considering that {a} {rel1} {b} and {b} {rel2} {c}, how are {a} and {c} connected?",
]


def _name(graph, key: str) -> str:
    return graph.nodes[key].get("name", key)


def generate_single_hop(graph, chunks_by_id: dict, existing_questions: set, target: int) -> list[dict]:
    random.seed(SEED)
    candidates = [
        k for k in graph.nodes
        if len(_name(graph, k)) >= MIN_ENTITY_NAME_LEN
        and graph.degree(k) >= 2
        and is_relevant_node(graph, k, chunks_by_id)  # skip contaminated entities (e.g. "LSST DESC")
    ]
    # Prioritize higher-degree (more central, better-covered) entities first —
    # these are more likely to have enough chunk context for a good answer.
    candidates.sort(key=lambda k: -graph.degree(k))

    new_questions = []
    for key in candidates:
        if len(new_questions) >= target:
            break
        name = _name(graph, key)
        template = random.choice(SINGLE_HOP_TEMPLATES)
        q = template.format(name=name)
        if q not in existing_questions:
            new_questions.append({"question": q, "category": "single_hop"})
            existing_questions.add(q)
    return new_questions


def generate_multi_hop(graph, chunks_by_id: dict, existing_questions: set, target: int) -> list[dict]:
    random.seed(SEED + 1)
    seen_pairs = set()
    new_questions = []
    relevance_cache: dict[str, bool] = {}

    def _relevant(key: str) -> bool:
        if key not in relevance_cache:
            relevance_cache[key] = is_relevant_node(graph, key, chunks_by_id)
        return relevance_cache[key]

    edges = list(graph.edges(data=True))
    random.shuffle(edges)

    for u, v, data1 in edges:
        if len(new_questions) >= target:
            break
        if len(_name(graph, u)) < MIN_ENTITY_NAME_LEN or len(_name(graph, v)) < MIN_ENTITY_NAME_LEN:
            continue
        if not (_relevant(u) and _relevant(v)):
            continue  # don't seed a multi-hop path from a contaminated entity
        for _, w, data2 in graph.out_edges(v, data=True):
            if len(new_questions) >= target:
                break
            if w == u or len(_name(graph, w)) < MIN_ENTITY_NAME_LEN:
                continue
            if not _relevant(w):
                continue  # the 3rd hop can still be contaminated even if u,v aren't
            pair_key = tuple(sorted((u, w)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            a, b, c = _name(graph, u), _name(graph, v), _name(graph, w)
            rel1 = data1.get("relation", "relates to")
            rel2 = data2.get("relation", "relates to")
            template = random.choice(MULTI_HOP_TEMPLATES)
            q = template.format(a=a, b=b, c=c, rel1=rel1, rel2=rel2)
            if q not in existing_questions:
                new_questions.append({"question": q, "category": "multi_hop"})
                existing_questions.add(q)
    return new_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-hop-target", type=int, default=15,
                         help="Total single_hop questions desired (existing + new)")
    parser.add_argument("--multi-hop-target", type=int, default=25,
                         help="Total multi_hop questions desired (existing + new)")
    args = parser.parse_args()

    current = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    existing_texts = {q["question"] for q in current}
    current_single = sum(1 for q in current if q.get("category") == "single_hop")
    current_multi = sum(1 for q in current if q.get("category") == "multi_hop")

    need_single = max(0, args.single_hop_target - current_single)
    need_multi = max(0, args.multi_hop_target - current_multi)

    graph = load_graph()
    chunks_by_id = _load_chunk_texts()
    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Current: {current_single} single_hop, {current_multi} multi_hop")
    print(f"Target:  {args.single_hop_target} single_hop, {args.multi_hop_target} multi_hop")
    print(f"Generating up to: {need_single} new single_hop, {need_multi} new multi_hop")
    print("(filtering out off-topic/contaminated entities via app/domain_filter.py)\n")

    new_single = generate_single_hop(graph, chunks_by_id, existing_texts, need_single)
    new_multi = generate_multi_hop(graph, chunks_by_id, existing_texts, need_multi)

    updated = current + new_single + new_multi
    QUESTIONS_PATH.write_text(json.dumps(updated, indent=2), encoding="utf-8")

    print(f"Added {len(new_single)} single_hop, {len(new_multi)} multi_hop questions.")
    print(f"Total questions now: {len(updated)}")

    if len(new_single) < need_single or len(new_multi) < need_multi:
        print(
            "\n[note] The graph didn't have enough distinct high-quality candidates to "
            "fully reach your targets (this happens with a smaller corpus / sparse "
            "graph). Consider lowering --single-hop-target / --multi-hop-target, or "
            "rebuilding the graph index (03_build_graph_index.py) over a larger corpus."
        )
