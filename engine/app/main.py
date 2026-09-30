import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import query, compare, graph, stats

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Renewable Energy RAG Comparison Engine",
    description="Vector RAG vs Graph RAG over a renewable-energy text corpus",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(compare.router)
app.include_router(graph.router)
app.include_router(stats.router)


@app.on_event("startup")
async def _log_active_model():
    # Makes it impossible to silently run an eval batch against the wrong
    # model — the active GROQ_MODEL is now printed in the uvicorn log on
    # every startup, not just buried in .env.
    from app.config import settings
    logger.info(f"[config] Active GROQ_MODEL for this run: {settings.groq_model}")


@app.get("/api/health")
def health():
    return {"status": "ok"}
