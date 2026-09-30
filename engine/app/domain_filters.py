"""
Single source of truth for "is this renewable-energy content" filtering.
History: allowlist -> denylist (3 contamination clusters) -> manual
entity-NAME blocklist -> manual entity-TYPE blocklist (this version).

v2 note: RELEVANT_ARXIV_CATEGORIES removed - arXiv collection was dropped
entirely from scripts/01_collect_corpus.py in the dataset rebuild (see
that file's docstring). Wikipedia was never the contamination source.
"""

RENEWABLE_ENERGY_KEYWORDS = [
    "renewable energy", "clean energy", "solar power", "solar energy",
    "solar photovoltaic", "photovoltaic", "pv module", "pv system",
    "solar cell", "wind power", "wind energy", "wind turbine", "wind farm",
    "offshore wind", "hydroelectric", "hydropower", "geothermal energy",
    "geothermal power", "biomass energy", "bioenergy", "biofuel",
    "tidal energy", "tidal power", "wave energy", "wave power",
    "energy storage", "battery storage", "grid energy storage",
    "grid integration", "smart grid", "microgrid", "feed-in tariff",
    "net metering", "renewable energy directive", "concentrated solar power",
    "green hydrogen", "hydrogen production", "carbon neutrality",
    "carbon emissions reduction", "energy transition", "decarbonization",
    "fossil fuel alternative",
]

OFF_TOPIC_CHUNK_KEYWORDS = [
    "dark energy", "dark matter", "cosmological", "cosmology", "supernova",
    "galaxy survey", "large-scale structure", "lsst desc", "lsst-desc",
    "particle accelerator", "collider", "wfmos", "lep2",
    "cosmic microwave background", " cmb ", "redshift survey",
    "baryon acoustic", "gravitational lensing", "exoplanet",
    "ilc technology network", "international linear collider", "neutrino",
    "lepton asymmetry",
    "precision grazing", "grazing management", "pasture biomass",
    "vegetation index", "ndvi", "csiro",
    "solar flare", "solar orbiter", "solar corona", "coronal mass ejection",
    "solar energetic particle", "heliospher", "heliophysic",
]

OFF_TOPIC_ENTITY_NAMES = {
    "lsst desc", "des", "desc", "wfmos", "lep2", "lsst",
    "dark energy survey", "dark energy science collaboration",
    "csiro", "solar flare", "solar orbiter", "solar corona",
}

# Live, add lowercase entity names as junk is spotted in the graph viewer.
# Persists across every rebuild; respected automatically by
# 07_filter_graph.py, app/routers/graph.py, and 05_generate_questions.py.
MANUAL_ENTITY_BLOCKLIST: set[str] = {
    # e.g. "some mangled duplicate name",
}

# NEW: entity TYPES the extraction LLM invents that are structurally noisy
# rather than topically off-topic - things like a paper's own metadata
# ("Authors", "Study", "Mission") rather than a renewable-energy concept.
# This is a DIFFERENT problem from topic contamination above: a node can
# have a perfectly on-topic name and still be junk because of its type, or
# vice versa. Don't conflate the two lists.
# Seeded from what was actually observed in the graph viewer (Study 1,
# Mission 1, Authors 1, Currency 1, Observer 1, Source 1, Data Source 1,
# Field 1, Condition 2, Control Method 1, Time Period 2, Model Structure 1,
# Factor 1) - all singleton/near-singleton noise types.
MANUAL_TYPE_BLOCKLIST: set[str] = {
    "currency", "authors", "study", "mission", "observer", "source",
    "data source", "field", "condition", "control method", "time period",
    "model structure", "factor",
}


def text_is_off_topic(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in OFF_TOPIC_CHUNK_KEYWORDS)


def text_matches_allowlist(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in RENEWABLE_ENERGY_KEYWORDS)


def is_relevant_document(title: str, text: str) -> bool:
    blob = f"{title} {text}"
    if text_is_off_topic(blob):
        return False
    return text_matches_allowlist(blob)


def entity_name_is_off_topic(name: str) -> bool:
    n = name.strip().lower()
    return n in OFF_TOPIC_ENTITY_NAMES or n in MANUAL_ENTITY_BLOCKLIST


def entity_type_is_noise(entity_type: str) -> bool:
    """True if this entity TYPE (e.g. 'Study', 'Currency') is structural
    noise rather than a real renewable-energy concept type."""
    return (entity_type or "").strip().lower() in MANUAL_TYPE_BLOCKLIST


def is_relevant_node(graph, key: str, chunks_by_id: dict) -> bool:
    """Drop a node if its name is denylisted, its TYPE is denylisted, or
    every chunk backing it is off-topic. Kept relevant if at least one
    backing chunk is clean, or if it has no recorded chunk_ids (judged on
    name/type only)."""
    data = graph.nodes[key]
    name = data.get("name", key)
    if entity_name_is_off_topic(name):
        return False
    if entity_type_is_noise(data.get("type", "unknown")):
        return False
    chunk_ids = data.get("chunk_ids", [])
    if not chunk_ids:
        return True
    for cid in chunk_ids:
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        if not text_is_off_topic(f"{chunk.title} {chunk.text}"):
            return True
    return False
