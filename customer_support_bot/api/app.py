import logging
import re
import time
import uuid

import jwt
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from customer_support_bot.agent.graph import AGENT
from customer_support_bot.agent.cache import check_semantic_cache, store_in_cache, append_to_session
from customer_support_bot.ingestion.embedder import run_ingestion_pipeline
from customer_support_bot.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Support Bot API", version="1.0.0")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    session_id: str
    faithfulness_score: float
    cache_hit: bool
    latency_ms: int


class IngestRequest(BaseModel):
    filepath: str
    tenant_id: str


def get_tenant_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, CONFIG.jwt_secret, algorithms=["HS256"])
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Token missing tenant_id claim")
        return tenant_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def mask_pii(text: str) -> str:
    text = re.sub(r'\b(\d{4}[\s-]){3}\d{4}\b', '[CARD_MASKED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_MASKED]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', text)
    text = re.sub(r'(?i)(account\s*#?\s*)(\d{6,12})', r'\1[ACCT_MASKED]', text)
    return text


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, tenant_id: str = Depends(get_tenant_id)):
    start_time = time.time()
    clean_question = mask_pii(request.question)

    cached_answer = check_semantic_cache(clean_question, tenant_id)
    if cached_answer:
        latency_ms = int((time.time() - start_time) * 1000)
        return QueryResponse(
            answer=cached_answer,
            citations=["Cached response"],
            session_id=request.session_id,
            faithfulness_score=1.0,
            cache_hit=True,
            latency_ms=latency_ms,
        )

    initial_state = {
        "question": clean_question,
        "session_id": request.session_id,
        "tenant_id": tenant_id,
        "query_embedding": None,
        "retrieved_chunks": None,
        "graded_chunks": None,
        "generation": None,
        "context_used": None,
        "faithfulness_score": None,
        "unsupported_claims": None,
        "final_answer": None,
        "citations": None,
        "route_decision": None,
        "error": None,
    }

    try:
        final_state = await AGENT.ainvoke(initial_state)
    except Exception as e:
        logger.error("Agent invocation failed: %s", e)
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")

    final_answer = final_state.get("final_answer", "")
    if not final_answer:
        raise HTTPException(status_code=500, detail="No answer generated")

    citations = final_state.get("citations", [])
    faithfulness = final_state.get("faithfulness_score", 0.0) or 0.0

    if faithfulness >= CONFIG.faithfulness_threshold:
        store_in_cache(clean_question, final_answer, tenant_id)

    append_to_session(request.session_id, "user", clean_question)
    append_to_session(request.session_id, "assistant", final_answer)

    latency_ms = int((time.time() - start_time) * 1000)
    return QueryResponse(
        answer=final_answer,
        citations=citations or [],
        session_id=request.session_id,
        faithfulness_score=faithfulness,
        cache_hit=False,
        latency_ms=latency_ms,
    )


@app.post("/ingest")
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
):
    if request.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Cannot ingest to another tenant")
    background_tasks.add_task(run_ingestion_pipeline, request.filepath, tenant_id)
    return {"status": "accepted", "message": "Document ingestion started."}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}
