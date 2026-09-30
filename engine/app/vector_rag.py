import time
import chromadb
from chromadb.utils import embedding_functions
from shared.schemas import Chunk, RetrievedContext, RAGAnswer
from app.config import settings, CHROMA_DIR
from app.llm_client import generate_text

_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.embedding_model  # runs on CPU, ~90MB model
)

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
COLLECTION_NAME = "renewable_energy_chunks"


def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embed_fn
    )


def build_index(chunks: list[Chunk], batch_size: int = 64) -> int:
    """(Re)build the vector index from scratch. Returns number of chunks indexed."""
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{"source": c.source, "title": c.title, "order": c.order} for c in batch],
        )
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")
    return len(chunks)


def retrieve(question: str, top_k: int = 5) -> list[RetrievedContext]:
    collection = get_collection()
    res = collection.query(query_texts=[question], n_results=top_k)
    contexts = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    for cid, doc, dist, meta in zip(ids, docs, dists, metas):
        contexts.append(
            RetrievedContext(
                chunk_id=cid,
                text=doc,
                score=1 - dist,  # convert distance to a similarity-like score
                source=meta.get("source"),
            )
        )
    return contexts


ANSWER_SYSTEM_PROMPT = (
    "You are a renewable-energy research assistant. Answer the user's question "
    "using ONLY the provided context passages. If the context is insufficient, "
    "say so explicitly rather than guessing. Cite passages inline like [1], [2] "
    "matching their order in the context."
)


def answer(question: str, top_k: int = 5) -> RAGAnswer:
    start = time.time()
    contexts = retrieve(question, top_k=top_k)
    context_block = "\n\n".join(
        f"[{i+1}] (source: {c.source})\n{c.text}" for i, c in enumerate(contexts)
    )
    prompt = f"Context passages:\n{context_block}\n\nQuestion: {question}"
    text = generate_text(prompt, system=ANSWER_SYSTEM_PROMPT)
    return RAGAnswer(
        method="vector",
        question=question,
        answer=text,
        contexts=contexts,
        latency_seconds=round(time.time() - start, 3),
    )
