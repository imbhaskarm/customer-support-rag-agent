import json
import logging
import os
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from customer_support_bot.config import CONFIG
from customer_support_bot.prompts import (
    SYSTEM_PROMPT,
    RETRIEVAL_GRADER_PROMPT,
    FAITHFULNESS_PROMPT,
    GENERATION_PROMPT,
)
from customer_support_bot.agent.state import AgentState, RetrievedChunk

logger = logging.getLogger(__name__)

if not CONFIG.groq_api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
    )

llm = ChatGroq(
    model=CONFIG.llm_model,
    temperature=CONFIG.llm_temperature,
    max_tokens=CONFIG.llm_max_tokens,
    groq_api_key=CONFIG.groq_api_key,
)

# ⚠️ Updated from deprecated OpenAIEmbeddings → HuggingFaceEmbeddings (latest as of 2025)
embeddings = HuggingFaceEmbeddings(
    model_name=CONFIG.embedding_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def _get_faiss_store(tenant_id: str) -> FAISS:
    index_path = os.path.join(CONFIG.faiss_index_path, f"tenant_{tenant_id}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found for tenant '{tenant_id}' at {index_path}. "
            "Run `python -m customer_support_bot.ingestion.ingest --tenant-id acme --filepath sample_docs/acme_sample_policy.docx` first."
        )
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    question = state["question"]
    tenant_id = state["tenant_id"]

    try:
        vectorstore = _get_faiss_store(tenant_id)
        raw_results = vectorstore.similarity_search_with_score(
            question,
            k=CONFIG.retrieval_top_k * 2,
        )

        chunks = []
        for doc, distance in raw_results:
            similarity = float(1.0 / (1.0 + distance))
            if similarity >= CONFIG.retrieval_similarity_threshold:
                meta = doc.metadata
                chunks.append(
                    RetrievedChunk(
                        text=doc.page_content,
                        source_doc=meta.get("source_doc", "Unknown"),
                        page_number=meta.get("page_number", 0),
                        chunk_id=meta.get("chunk_id", ""),
                        similarity_score=similarity,
                        relevance_grade="ungraded",
                        section=meta.get("section", "Unknown"),
                    )
                )

        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        chunks = chunks[: CONFIG.retrieval_top_k]
        return {"retrieved_chunks": chunks, "query_embedding": None}

    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return {"error": f"Retrieval failed: {e}", "retrieved_chunks": []}


def grade_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("retrieved_chunks", [])
    question = state["question"]

    if not chunks:
        return {"graded_chunks": [], "route_decision": "fallback"}

    graded_chunks = []
    for chunk in chunks:
        prompt = RETRIEVAL_GRADER_PROMPT.format(question=question, chunk=chunk.text)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            grade = result.get("score", "irrelevant")
            chunk.relevance_grade = grade
            if grade in ("relevant", "partial"):
                graded_chunks.append(chunk)
        except Exception as e:
            logger.warning("Grading failed; keeping chunk conservatively: %s", e)
            chunk.relevance_grade = "ungraded"
            graded_chunks.append(chunk)

    return {
        "graded_chunks": graded_chunks,
        "route_decision": "generate" if graded_chunks else "fallback",
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    chunks = state["graded_chunks"] or []
    question = state["question"]

    context_parts = []
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk.source_doc}, Section {chunk.section}, Page {chunk.page_number}]\n{chunk.text}"
        )
        citations.append(f"{chunk.source_doc}, {chunk.section}")

    context_str = "\n---\n".join(context_parts)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=GENERATION_PROMPT.format(context=context_str, question=question)),
    ]

    try:
        response = llm.invoke(messages)
        return {
            "generation": response.content,
            "context_used": context_str,
            "citations": citations,
        }
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return {"error": f"Generation failed: {e}", "generation": None}


def faithfulness_node(state: AgentState) -> Dict[str, Any]:
    generation = state.get("generation")
    context = state.get("context_used", "")

    if not generation:
        return {"route_decision": "fallback", "faithfulness_score": 0.0}

    try:
        prompt = FAITHFULNESS_PROMPT.format(answer=generation, context=context)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        score = float(result.get("faithfulness_score", 0.0))
        unsupported = result.get("unsupported_claims", [])

        if score >= CONFIG.faithfulness_threshold:
            return {
                "faithfulness_score": score,
                "unsupported_claims": unsupported,
                "final_answer": generation,
                "route_decision": "approved",
            }

        return {
            "faithfulness_score": score,
            "unsupported_claims": unsupported,
            "final_answer": None,
            "route_decision": "fallback",
        }
    except Exception as e:
        logger.error("Faithfulness check failed: %s", e)
        return {"route_decision": "fallback", "faithfulness_score": 0.0}


def fallback_node(state: AgentState) -> Dict[str, Any]:
    return {
        "final_answer": (
            "I want to make sure I give you accurate information about this. "
            "For this specific question, please contact our support team directly "
            "at support@acmefinancial.com or call 1-800-ACME-SUP. "
            "A specialist will be happy to help you."
        ),
        "route_decision": "complete",
        "citations": [],
    }
