"""
Reads engine/data/processed/chunks.jsonl (produced by script 02) and builds
the entity/relationship graph by calling the LLM once per chunk.

This is the slow, "expensive" step: with ~150-250 chunks and one Gemini
call each, expect this to take a while and to consume free-tier API quota.
Start with a subset while testing (see LIMIT below).

Run from the engine/ folder:
    python scripts/03_build_graph_index.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import CHUNKS_PATH, GRAPH_PATH
from shared.schemas import Chunk
from app import graph_rag

# Set this to 15 to only process a few chunks at a time for your quota
LIMIT = 100


def load_chunks() -> list[Chunk]:
    chunks = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk(**json.loads(line)))
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    if LIMIT:
        chunks = chunks[:LIMIT]
    print(f"Building graph from {len(chunks)} chunks (one LLM call per chunk)...")
    graph = graph_rag.build_graph(chunks, sleep_between_calls=4.0)
    graph_rag.save_graph(graph, GRAPH_PATH)
    print(f"Done. Graph has {graph.number_of_nodes()} nodes and "
          f"{graph.number_of_edges()} edges. Saved to {GRAPH_PATH}")