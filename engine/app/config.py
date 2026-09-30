from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# engine/  <- this file lives in engine/app/config.py, so parents[1] == engine/
ENGINE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ENGINE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
GRAPH_PATH = INDEX_DIR / "graph.gpickle.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENGINE_DIR / ".env"), extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_words: int = 300
    chunk_overlap_words: int = 50
    graph_hops: int = 2
    top_k_default: int = 5


settings = Settings()

for d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)
