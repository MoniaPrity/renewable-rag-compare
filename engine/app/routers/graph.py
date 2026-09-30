"""
Serves the built knowledge graph (graph.gpickle.json) as JSON for the
frontend's graph-explorer page. Read-only — this never mutates the graph,
it just reshapes what graph_rag.load_graph() already loads into a format
a force-directed graph visualization library can consume directly.

CHANGES vs original:
  - /api/graph/data now accepts optional `exclude_types` and `min_degree`
    query params, so you can test a cleaner cut of the graph in the viewer
    without re-running scripts/07_filter_graph.py or rebuilding anything.
    e.g. GET /api/graph/data?exclude_types=Unknown&min_degree=2
  - /api/graph/overview now excludes nodes on the domain denylist
    (app.domain_filters.OFF_TOPIC_ENTITY_NAMES) from "top_nodes_by_degree"
    so a contaminated node can't dominate that list even before you've run
    the cleanup script — this is a display-only filter, it doesn't touch
    the underlying graph file.
  - NEW: both endpoints also exclude nodes whose TYPE is on
    domain_filters.MANUAL_TYPE_BLOCKLIST (e.g. "Study", "Currency",
    "Authors") — this is a display-only filter for structural noise types,
    separate from the name-based topic-contamination filter above. A node
    can have a perfectly on-topic name and still be noise because of its
    type.
"""
from fastapi import APIRouter, HTTPException, Query
from app.graph_rag import load_graph
from app.domain_filters import entity_name_is_off_topic, entity_type_is_noise

router = APIRouter(prefix="/api/graph", tags=["graph-viz"])


@router.get("/overview")
def graph_overview():
    """Summary stats: node/edge counts, entity type breakdown, most-connected entities."""
    try:
        graph = load_graph()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    node_types: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        t = data.get("type", "unknown") or "unknown"
        node_types[t] = node_types.get(t, 0) + 1

    # Display-only filter: don't let a known off-topic entity, or a
    # structurally-noisy entity type, occupy a slot in the "top connected"
    # list even if the underlying graph file hasn't been cleaned yet with
    # scripts/07_filter_graph.py.
    candidates = [
        (key, degree) for key, degree in graph.degree
        if not entity_name_is_off_topic(graph.nodes[key].get("name", key))
        and not entity_type_is_noise(graph.nodes[key].get("type", "unknown"))
    ]
    top_by_degree = sorted(candidates, key=lambda item: -item[1])[:15]
    top_nodes = [
        {"name": graph.nodes[key].get("name", key), "degree": degree}
        for key, degree in top_by_degree
    ]

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "node_types": node_types,
        "top_nodes_by_degree": top_nodes,
    }


@router.get("/data")
def graph_data(
    limit_nodes: int = 150,
    min_degree: int = Query(0, description="Only include nodes with at least this many connections"),
    exclude_types: str = Query("", description="Comma-separated entity types to exclude, e.g. 'Unknown,Metric'"),
):
    """
    Returns nodes + edges shaped for react-force-graph.

    If the graph has more than `limit_nodes` entities, keeps only the
    highest-degree (most-connected) ones — rendering every single extracted
    entity from a large corpus would both choke the browser and produce an
    unreadable hairball. Pass ?limit_nodes=500 etc. from the frontend if you
    want to see more (or fewer).

    `min_degree` and `exclude_types` let you test a cleaner cut on the fly,
    e.g. ?min_degree=2&exclude_types=Unknown to hide low-signal noise nodes
    without touching the underlying graph file. Nodes on the manual
    entity-type blocklist (domain_filters.MANUAL_TYPE_BLOCKLIST) are
    excluded automatically regardless of this param.
    """
    try:
        graph = load_graph()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    excluded_types = {t.strip().lower() for t in exclude_types.split(",") if t.strip()}

    def _keep_node(key: str) -> bool:
        data = graph.nodes[key]
        if entity_name_is_off_topic(data.get("name", key)):
            return False
        node_type = (data.get("type", "unknown") or "unknown").lower()
        if entity_type_is_noise(node_type):
            return False
        if graph.degree(key) < min_degree:
            return False
        if node_type in excluded_types:
            return False
        return True

    eligible = [key for key in graph.nodes if _keep_node(key)]
    was_truncated = len(eligible) > limit_nodes

    if was_truncated:
        eligible.sort(key=lambda key: -graph.degree(key))
        eligible = eligible[:limit_nodes]

    keep = set(eligible)

    nodes = [
        {
            "id": key,
            "name": graph.nodes[key].get("name", key),
            "type": graph.nodes[key].get("type", "unknown") or "unknown",
            "chunk_count": len(graph.nodes[key].get("chunk_ids", [])),
        }
        for key in keep
    ]

    edges = [
        {"source": u, "target": v, "relation": data.get("relation", "related_to")}
        for u, v, data in graph.edges(data=True)
        if u in keep and v in keep
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": was_truncated,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }
