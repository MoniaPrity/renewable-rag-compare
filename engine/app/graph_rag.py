import json
import time
import difflib
import networkx as nx
from shared.schemas import Chunk, RetrievedContext, RAGAnswer
from app.config import settings, GRAPH_PATH
from app.llm_client import generate_json, generate_text
from app.reranker import rerank

EXTRACTION_SYSTEM_PROMPT = """You extract a knowledge graph from a passage about
renewable energy. Identify entities (technologies, organizations, materials,
locations, metrics, policies, etc.) and the relationships between them found
IN THIS PASSAGE ONLY. Keep entity names short and consistent (e.g. always
"Solar Photovoltaic" not sometimes "PV" and sometimes "Solar PV").

Return JSON of the form:
{
  "entities": [{"name": "...", "type": "..."}],
  "relations": [{"source": "...", "relation": "...", "target": "..."}]
}
Extract at most 8 entities and 8 relations. If the passage has nothing
graph-worthy, return {"entities": [], "relations": []}.
"""


def _extract_from_chunk(chunk: Chunk) -> dict:
    prompt = f"Passage:\n{chunk.text}"
    try:
        data = generate_json(prompt, system=EXTRACTION_SYSTEM_PROMPT)
        if not isinstance(data, dict):
            return {"entities": [], "relations": []}
        return data
    except ValueError as e:
        print(f"  [warn] extraction failed for {chunk.chunk_id}: {e}")
        return {"entities": [], "relations": []}


def build_graph(chunks: list[Chunk], sleep_between_calls: float = 0.0) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for i, chunk in enumerate(chunks):
        data = _extract_from_chunk(chunk)
        for ent in data.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            if graph.has_node(key):
                graph.nodes[key].setdefault("chunk_ids", [])
                if chunk.chunk_id not in graph.nodes[key]["chunk_ids"]:
                    graph.nodes[key]["chunk_ids"].append(chunk.chunk_id)
            else:
                graph.add_node(
                    key,
                    name=name,
                    type=ent.get("type", "unknown"),
                    chunk_ids=[chunk.chunk_id],
                )
        for rel in data.get("relations", []):
            s, t = rel.get("source", "").strip(), rel.get("target", "").strip()
            if not s or not t:
                continue
            sk, tk = s.lower(), t.lower()
            if not graph.has_node(sk):
                graph.add_node(sk, name=s, type="unknown", chunk_ids=[chunk.chunk_id])
            if not graph.has_node(tk):
                graph.add_node(tk, name=t, type="unknown", chunk_ids=[chunk.chunk_id])
            graph.add_edge(sk, tk, relation=rel.get("relation", "related_to"), chunk_id=chunk.chunk_id)
        print(f"  processed chunk {i+1}/{len(chunks)} -> "
              f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges so far")
        if sleep_between_calls:
            time.sleep(sleep_between_calls)
    return graph


def save_graph(graph: nx.MultiDiGraph, path=GRAPH_PATH) -> None:
    data = nx.node_link_data(graph)
    path.write_text(json.dumps(data), encoding="utf-8")


_graph_cache: nx.MultiDiGraph | None = None


def load_graph(path=GRAPH_PATH) -> nx.MultiDiGraph:
    global _graph_cache
    if _graph_cache is None:
        if not path.exists():
            raise FileNotFoundError(
                f"No graph found at {path}. Run scripts/03_build_graph_index.py first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        _graph_cache = nx.node_link_graph(data, multigraph=True, directed=True)
    return _graph_cache


# --- FIX for root cause (a): short/noise entity names (e.g. "EM", "DOE")
# were fuzzy-matching against the question by chance character overlap and
# seeding the traversal with irrelevant neighborhoods. Applies to ALL graph
# retrieval (raw and reranked) since it's a seeding-quality bug, not a
# ranking bug — keeping it universal is also what makes the reranked-vs-raw
# comparison a clean, single-variable experiment.
MIN_ENTITY_NAME_LEN = 3


def _match_entities(question: str, graph: nx.MultiDiGraph, limit: int = 5) -> list[str]:
    """Fuzzy-match words/phrases in the question against graph node names.
    No extra ML dependency needed: difflib is stdlib and good enough here."""
    node_names = {k: d.get("name", k) for k, d in graph.nodes(data=True)}
    q_lower = question.lower()
    scored = []
    for key, name in node_names.items():
        name_l = name.lower()
        if len(name_l) < MIN_ENTITY_NAME_LEN:
            continue  # skip "EM", "DOE", etc. — too short to fuzzy-match reliably
        if name_l in q_lower:
            scored.append((1.0, key))
        else:
            ratio = difflib.SequenceMatcher(None, name_l, q_lower).ratio()
            if ratio > 0.35:
                scored.append((ratio, key))
    scored.sort(key=lambda x: -x[0])
    return [key for _, key in scored[:limit]]


def _load_chunk_texts() -> dict[str, Chunk]:
    from app.config import CHUNKS_PATH
    chunks = {}
    if CHUNKS_PATH.exists():
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = Chunk(**json.loads(line))
                chunks[c.chunk_id] = c
    return chunks


# Upper bound on how many candidate chunks get scored by the cross-encoder
# in one call, ONLY applied if the 2-hop pool genuinely exceeds this (it
# won't, at this corpus size — this just guards against a pathologically
# dense subgraph making a single query slow). This replaces the old
# `candidate_pool_size=20` PRE-rerank truncation, which was the actual bug
# behind the multi-hop regression — see retrieve() below.
RERANK_SAFETY_CAP = 200


def retrieve(
    question: str,
    top_k: int = 5,
    hops: int | None = None,
    use_reranker: bool = False,
    candidate_pool_size: int = RERANK_SAFETY_CAP,
) -> tuple[list[RetrievedContext], dict]:
    """
    use_reranker=False -> ORIGINAL behavior (Raw Graph RAG baseline):
        2-hop traversal, chunk_ids collected into an unordered set,
        list(chunk_ids)[:top_k] taken with no ranking.

    use_reranker=True -> Reranked Graph RAG:
        Same 2-hop traversal. ALL candidate chunks from the traversal are
        scored by the local Cross-Encoder, and only THEN is the pool cut
        down to top_k by relevance.

        BUGFIX (multi-hop regression): the previous implementation did
        `list(chunk_ids)[:candidate_pool_size]` BEFORE reranking, where
        chunk_ids is an unordered Python set. That silently discarded
        chunks in arbitrary hash order before the reranker ever saw them.
        On multi-hop questions, the single "bridging" chunk connecting hop
        1 to hop 2 could easily land outside that arbitrary slice — no
        cross-encoder can rank a chunk it never receives. `candidate_pool_size`
        (now defaulting to RERANK_SAFETY_CAP=200) is applied only AFTER
        collecting the full pool, and only as a performance guard for
        unusually dense subgraphs, not as a pre-filter.
    """
    graph = load_graph()
    hops = hops if hops is not None else settings.graph_hops
    seed_keys = _match_entities(question, graph)

    subgraph_nodes: set[str] = set(seed_keys)
    for key in seed_keys:
        if key in graph:
            ego = nx.ego_graph(graph.to_undirected(as_view=True), key, radius=hops)
            subgraph_nodes.update(ego.nodes)

    chunk_ids: set[str] = set()
    relations_used = []
    for key in subgraph_nodes:
        chunk_ids.update(graph.nodes[key].get("chunk_ids", []))
    for u, v, data in graph.edges(data=True):
        if u in subgraph_nodes and v in subgraph_nodes:
            relations_used.append(f"{graph.nodes[u]['name']} --{data.get('relation')}--> {graph.nodes[v]['name']}")

    all_chunks = _load_chunk_texts()
    reranker_scores: dict[str, float] = {}
    candidate_ids: list[str] | None = None

    if use_reranker:
        all_ids = list(chunk_ids)
        # Only truncate if the pool is genuinely larger than the safety cap
        # -- and even then, this is a size guard, not a relevance decision;
        # the reranker still sees everything up to that cap before choosing
        # top_k, instead of top_k being pre-decided by set iteration order.
        candidate_ids = all_ids if len(all_ids) <= candidate_pool_size else all_ids[:candidate_pool_size]
        candidates = [(cid, all_chunks[cid].text) for cid in candidate_ids if cid in all_chunks]
        ranked = rerank(question, candidates, top_k=top_k)
        selected = [(cid, text) for cid, text, _ in ranked]
        reranker_scores = {cid: score for cid, _, score in ranked}
    else:
        selected_ids = list(chunk_ids)[:top_k]
        selected = [(cid, all_chunks[cid].text) for cid in selected_ids if cid in all_chunks]

    contexts = []
    for cid, _text in selected:
        chunk = all_chunks.get(cid)
        if chunk:
            score = reranker_scores.get(cid, 1.0)
            contexts.append(RetrievedContext(chunk_id=cid, text=chunk.text, score=score, source=chunk.source))

    extra = {
        "matched_entities": [graph.nodes[k]["name"] for k in seed_keys if k in graph],
        "relations_used": relations_used[:15],
        "subgraph_size": len(subgraph_nodes),
        "reranked": use_reranker,
        "candidate_pool_size": len(candidate_ids) if candidate_ids is not None else None,
    }
    return contexts, extra


ANSWER_SYSTEM_PROMPT = (
    "You are a renewable-energy research assistant. Answer the user's question "
    "using ONLY the provided context passages and the listed graph relationships. "
    "If the information is insufficient, say so explicitly. Cite passages inline "
    "like [1], [2] matching their order in the context."
)


def answer(question: str, top_k: int = 5, use_reranker: bool = False) -> RAGAnswer:
    start = time.time()
    contexts, extra = retrieve(question, top_k=top_k, use_reranker=use_reranker)
    context_block = "\n\n".join(
        f"[{i+1}] (source: {c.source})\n{c.text}" for i, c in enumerate(contexts)
    )
    relations_block = "\n".join(extra.get("relations_used", [])) or "(none found)"
    prompt = (
        f"Graph relationships found:\n{relations_block}\n\n"
        f"Context passages:\n{context_block}\n\nQuestion: {question}"
    )
    text = generate_text(prompt, system=ANSWER_SYSTEM_PROMPT)
    return RAGAnswer(
        method="graph_reranked" if use_reranker else "graph",
        question=question,
        answer=text,
        contexts=contexts,
        latency_seconds=round(time.time() - start, 3),
        extra=extra,
    )
