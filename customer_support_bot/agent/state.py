from typing import TypedDict, Optional, List
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source_doc: str
    page_number: int
    chunk_id: str
    similarity_score: float
    relevance_grade: str
    section: str = "Unknown"


class AgentState(TypedDict):
    question: str
    session_id: str
    tenant_id: str

    query_embedding: Optional[List[float]]
    retrieved_chunks: Optional[List[RetrievedChunk]]
    graded_chunks: Optional[List[RetrievedChunk]]

    generation: Optional[str]
    context_used: Optional[str]

    faithfulness_score: Optional[float]
    unsupported_claims: Optional[List[str]]

    final_answer: Optional[str]
    citations: Optional[List[str]]
    route_decision: Optional[str]
    error: Optional[str]
