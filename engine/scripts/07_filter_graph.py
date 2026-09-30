"""
Cleans the ALREADY-BUILT knowledge graph of contamination — NO LLM calls,
$0 cost, no rebuild needed.

What it does:
  1. Scans chunks.jsonl for any chunk whose title+text matches
     domain_filters.OFF_TOPIC_CHUNK_KEYWORDS -> marks it "contaminated".
  2. For every graph node, strips references to contaminated chunk_ids.
     If a node ends up with zero chunk_ids left, it's dropped entirely.
  3. Drops any node whose name is in domain_filters.OFF_TOPIC_ENTITY_NAMES
     or MANUAL_ENTITY_BLOCKLIST, regardless of which chunks it came from.
  4. NEW: drops any node whose TYPE is in domain_filters.MANUAL_TYPE_
     BLOCKLIST (e.g. "Study", "Currency", "Authors") — a different problem
     from topic contamination: a node can have a perfectly on-topic name
     and still be structural noise because of its type.
  5. Drops any edge whose recorded chunk_id is contaminated, or whose
     endpoints no longer exist after steps 2-4.
  6. Backs up the original graph file, then writes the cleaned graph in
     place (so app/graph_rag.load_graph() picks it up with no other code
     changes needed).

Run from the engine/ folder, any time — safe to re-run:
    python scripts/07_filter_graph.py
    python scripts/07_filter_graph.py --dry-run   # report only, no changes written
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import CHUNKS_PATH, GRAPH_PATH
from app.domain_filters import text_is_off_topic, entity_name_is_off_topic, entity_type_is_noise
from shared.schemas import Chunk


def load_chunks() -> dict[str, Chunk]:
    chunks = {}
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = Chunk(**json.loads(line))
            chunks[c.chunk_id] = c
    return chunks


def find_contaminated_chunk_ids(chunks: dict[str, Chunk]) -> set[str]:
    contaminated = set()
    for cid, chunk in chunks.items():
        blob = f"{chunk.title} {chunk.text}"
        if text_is_off_topic(blob):
            contaminated.add(cid)
    return contaminated


def clean_graph(graph: nx.MultiDiGraph, contaminated_chunk_ids: set[str]) -> tuple[nx.MultiDiGraph, dict]:
    stats = {
        "nodes_before": graph.number_of_nodes(),
        "edges_before": graph.number_of_edges(),
        "dropped_nodes_denylist": [],
        "dropped_nodes_noisy_type": [],
        "dropped_nodes_no_chunks_left": [],
        "dropped_edges": 0,
    }

    cleaned = nx.MultiDiGraph()

    # Step 1: rebuild nodes, applying the entity-name denylist, the
    # entity-TYPE denylist (new), and stripping contaminated chunk_ids.
    for key, data in graph.nodes(data=True):
        name = data.get("name", key)
        node_type = data.get("type", "unknown")

        if entity_name_is_off_topic(name):
            stats["dropped_nodes_denylist"].append(name)
            continue

        if entity_type_is_noise(node_type):
            stats["dropped_nodes_noisy_type"].append(f"{name} ({node_type})")
            continue

        original_chunk_ids = data.get("chunk_ids", [])
        remaining_chunk_ids = [cid for cid in original_chunk_ids if cid not in contaminated_chunk_ids]

        if original_chunk_ids and not remaining_chunk_ids:
            # every chunk backing this entity was contaminated
            stats["dropped_nodes_no_chunks_left"].append(name)
            continue

        cleaned.add_node(key, name=name, type=node_type, chunk_ids=remaining_chunk_ids)

    # Step 2: rebuild edges, dropping any tied to a contaminated chunk or to
    # a node that got dropped above.
    for u, v, data in graph.edges(data=True):
        if u not in cleaned or v not in cleaned:
            stats["dropped_edges"] += 1
            continue
        if data.get("chunk_id") in contaminated_chunk_ids:
            stats["dropped_edges"] += 1
            continue
        cleaned.add_edge(u, v, relation=data.get("relation", "related_to"), chunk_id=data.get("chunk_id"))

    stats["nodes_after"] = cleaned.number_of_nodes()
    stats["edges_after"] = cleaned.number_of_edges()
    return cleaned, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would change without writing anything")
    args = parser.parse_args()

    if not GRAPH_PATH.exists():
        print(f"No graph found at {GRAPH_PATH}. Run scripts/03_build_graph_index.py first.")
        sys.exit(1)

    print("Loading chunks and graph...")
    chunks = load_chunks()
    raw_data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    graph = nx.node_link_graph(raw_data, multigraph=True, directed=True)

    contaminated_chunk_ids = find_contaminated_chunk_ids(chunks)
    print(f"Found {len(contaminated_chunk_ids)} contaminated chunk(s) out of {len(chunks)} total.")
    if contaminated_chunk_ids:
        print("Contaminated sources:")
        for cid in sorted(contaminated_chunk_ids):
            c = chunks[cid]
            print(f"  - {c.source} :: {c.title}")

    cleaned, stats = clean_graph(graph, contaminated_chunk_ids)

    print(f"\nNodes: {stats['nodes_before']} -> {stats['nodes_after']}")
    print(f"Edges: {stats['edges_before']} -> {stats['edges_after']}")
    if stats["dropped_nodes_denylist"]:
        print(f"\nDropped by explicit name denylist ({len(stats['dropped_nodes_denylist'])}):")
        for name in stats["dropped_nodes_denylist"]:
            print(f"  - {name}")
    if stats["dropped_nodes_noisy_type"]:
        print(f"\nDropped by entity-type denylist ({len(stats['dropped_nodes_noisy_type'])}):")
        for name in stats["dropped_nodes_noisy_type"]:
            print(f"  - {name}")
    if stats["dropped_nodes_no_chunks_left"]:
        print(f"\nDropped — every backing chunk was contaminated ({len(stats['dropped_nodes_no_chunks_left'])}):")
        for name in stats["dropped_nodes_no_chunks_left"]:
            print(f"  - {name}")
    print(f"\nDropped edges: {stats['dropped_edges']}")

    if args.dry_run:
        print("\n[dry run] No files were changed. Re-run without --dry-run to apply.")
        sys.exit(0)

    nothing_to_clean = (
        not contaminated_chunk_ids
        and not stats["dropped_nodes_denylist"]
        and not stats["dropped_nodes_noisy_type"]
    )
    if nothing_to_clean:
        print("\nNothing to clean — graph file left untouched.")
        sys.exit(0)

    backup_path = GRAPH_PATH.with_suffix(GRAPH_PATH.suffix + ".bak")
    shutil.copy2(GRAPH_PATH, backup_path)
    print(f"\nBacked up original graph to {backup_path}")

    data = nx.node_link_data(cleaned)
    GRAPH_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"Wrote cleaned graph to {GRAPH_PATH}")
    print(
        "\nNote: this only cleans the GRAPH. If you also plan to re-run "
        "03_build_graph_index.py from scratch later, first re-run "
        "01_collect_corpus.py so contaminated source pages aren't "
        "re-collected in the first place."
    )
