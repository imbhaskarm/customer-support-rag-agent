import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ⚠️ Updated from deprecated GPT/OpenAI setup → Groq + local embeddings (latest as of 2025)
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600

    # ⚠️ Updated from deprecated OpenAI embeddings → HuggingFaceEmbeddings (latest as of 2025)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    retrieval_top_k: int = 3
    retrieval_similarity_threshold: float = 0.40
    faithfulness_threshold: float = 0.75

    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 200

    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400
    session_ttl_seconds: int = 7200

    groq_api_key: str = ""
    redis_url: str = ""
    jwt_secret: str = ""
    faiss_index_path: str = "./faiss_indexes"


def load_config() -> Config:
    cfg = Config()
    cfg.groq_api_key = os.getenv("GROQ_API_KEY", "")
    cfg.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    cfg.jwt_secret = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    cfg.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_indexes")
    return cfg


CONFIG = load_config()
