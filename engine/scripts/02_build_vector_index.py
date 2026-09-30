"""
Reads engine/data/raw/*.json, splits each document into overlapping
word-window chunks, saves the flat chunk list to
engine/data/processed/chunks.jsonl (used by BOTH vector and graph RAG),
and builds the Chroma vector index.

CHANGES vs original (allowlist redo):
  - load_raw_documents() now applies domain_filters.is_relevant_document()
    as a SECOND gate, at the whole-document level, before any chunking or
    embedding happens. This matters because scripts/01_collect_corpus.py's
    filtering only covers what gets collected in future runs — it can't
    retroactively clean raw JSON files that already exist on disk (e.g. if
    you skip re-collecting and just rerun this script against old raw
    files). It also matters because the OLD contamination cleanup
    (scripts/07_filter_graph.py) only ever touched the GRAPH file — it
    never touched chunks.jsonl or the Chroma vector index, so Vector RAG
    could still have been retrieving off-topic chunks even after the graph
    "looked" clean. This gate closes that gap for both retrieval methods.

Run from the engine/ folder:
    python scripts/02_build_vector_index.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import RAW_DIR, CHUNKS_PATH, settings
from app.domain_filters import is_relevant_document
from shared.schemas import Chunk
from app import vector_rag


def chunk_text(text: str, words_per_chunk: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, words_per_chunk - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + words_per_chunk]
        if len(piece) < 20:  # drop tiny trailing fragments
            continue
        chunks.append(" ".join(piece))
        if start + words_per_chunk >= len(words):
            break
    return chunks


def load_raw_documents() -> list[dict]:
    docs = []
    for path in RAW_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            docs.extend(data)
        else:
            docs.append(data)
    return docs


def build_chunks() -> list[Chunk]:
    docs = load_raw_documents()

    kept_docs = []
    skipped_docs = []
    for doc in docs:
        if is_relevant_document(doc.get("title", ""), doc["text"]):
            kept_docs.append(doc)
        else:
            skipped_docs.append(doc.get("title", doc.get("source", "?")))

    if skipped_docs:
        print(f"  filtered {len(skipped_docs)} off-topic document(s) before chunking:")
        for title in skipped_docs:
            print(f"    - {title}")

    all_chunks: list[Chunk] = []
    counter = 0
    for doc in kept_docs:
        pieces = chunk_text(doc["text"], settings.chunk_words, settings.chunk_overlap_words)
        for order, piece in enumerate(pieces):
            all_chunks.append(Chunk(
                chunk_id=f"c{counter:05d}",
                source=doc["source"],
                title=doc.get("title", ""),
                text=piece,
                order=order,
            ))
            counter += 1
    return all_chunks


if __name__ == "__main__":
    print("Loading raw documents and chunking...")
    chunks = build_chunks()
    print(f"  {len(chunks)} chunks created")

    CHUNKS_PATH.write_text(
        "\n".join(c.model_dump_json() for c in chunks), encoding="utf-8"
    )
    print(f"  saved chunk list to {CHUNKS_PATH}")

    print("Building Chroma vector index (downloads the embedding model on first run)...")
    n = vector_rag.build_index(chunks)
    print(f"Done. Indexed {n} chunks into Chroma at engine/data/index/chroma/")
