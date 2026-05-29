import json
import hashlib
from typing import Optional, List

import numpy as np
import redis
from langchain_huggingface import HuggingFaceEmbeddings

from customer_support_bot.config import CONFIG

_redis_client = None
_embeddings_client = None

CACHE_PREFIX = "sem_cache:"
SESSION_PREFIX = "session:"
EMBED_PREFIX = "embed_cache:"


def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        _redis_client = redis.from_url(
            CONFIG.redis_url,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_client = None
        return None


def get_embeddings_client():
    global _embeddings_client
    if _embeddings_client is None:
        # ⚠️ Updated from deprecated OpenAIEmbeddings → HuggingFaceEmbeddings (latest as of 2025)
        _embeddings_client = HuggingFaceEmbeddings(
            model_name=CONFIG.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_client


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_query_embedding(query: str) -> List[float]:
    redis_client = get_redis_client()
    if redis_client is None:
        return get_embeddings_client().embed_query(query)

    query_hash = hashlib.sha256(query.encode()).hexdigest()
    cache_key = f"{EMBED_PREFIX}{query_hash}"

    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except redis.RedisError:
        pass

    embedding = get_embeddings_client().embed_query(query)
    try:
        redis_client.setex(cache_key, 3600, json.dumps(embedding))
    except redis.RedisError:
        pass
    return embedding


def check_semantic_cache(query: str, tenant_id: str) -> Optional[str]:
    redis_client = get_redis_client()
    if redis_client is None:
        return None

    try:
        query_embedding = get_query_embedding(query)
        pattern = f"{CACHE_PREFIX}{tenant_id}:*"
        cursor = 0
        best_match_answer = None
        best_similarity = 0.0

        while True:
            cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                cached_data = redis_client.get(key)
                if not cached_data:
                    continue
                entry = json.loads(cached_data)
                similarity = _cosine_similarity(query_embedding, entry["embedding"])
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= CONFIG.cache_similarity_threshold:
                        best_match_answer = entry["answer"]
            if cursor == 0:
                break

        return best_match_answer
    except redis.RedisError:
        return None


def store_in_cache(query: str, answer: str, tenant_id: str) -> None:
    redis_client = get_redis_client()
    if redis_client is None:
        return
    try:
        query_embedding = get_query_embedding(query)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_key = f"{CACHE_PREFIX}{tenant_id}:{query_hash}"
        payload = {"embedding": query_embedding, "answer": answer, "query": query}
        redis_client.setex(cache_key, CONFIG.cache_ttl_seconds, json.dumps(payload))
    except redis.RedisError:
        pass


def get_session_history(session_id: str) -> List[dict]:
    redis_client = get_redis_client()
    if redis_client is None:
        return []
    try:
        key = f"{SESSION_PREFIX}{session_id}"
        data = redis_client.get(key)
        return json.loads(data) if data else []
    except redis.RedisError:
        return []


def append_to_session(session_id: str, role: str, content: str) -> None:
    redis_client = get_redis_client()
    if redis_client is None:
        return
    try:
        key = f"{SESSION_PREFIX}{session_id}"
        history = get_session_history(session_id)
        history.append({"role": role, "content": content})
        history = history[-10:]
        redis_client.setex(key, CONFIG.session_ttl_seconds, json.dumps(history))
    except redis.RedisError:
        pass
