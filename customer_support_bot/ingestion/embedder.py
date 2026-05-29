import os
import logging
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

from customer_support_bot.ingestion.chunker import Chunk
from customer_support_bot.config import CONFIG

logger = logging.getLogger(__name__)

# ⚠️ Updated from deprecated OpenAIEmbeddings → HuggingFaceEmbeddings (latest as of 2025)
embeddings_client = HuggingFaceEmbeddings(
    model_name=CONFIG.embedding_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def embed_and_save(chunks: List[Chunk], tenant_id: str) -> dict:
    collection_name = f"tenant_{tenant_id}"
    index_path = os.path.join(CONFIG.faiss_index_path, collection_name)
    os.makedirs(CONFIG.faiss_index_path, exist_ok=True)

    docs = [LCDocument(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings_client)
    vectorstore.save_local(index_path)

    return {
        "total_chunks": len(chunks),
        "saved": len(chunks),
        "index_path": index_path,
    }


def run_ingestion_pipeline(filepath: str, tenant_id: str) -> dict:
    from customer_support_bot.ingestion.document_loader import load_document
    from customer_support_bot.ingestion.chunker import chunk_blocks

    logger.info("Starting ingestion: %s → tenant %s", filepath, tenant_id)
    blocks = load_document(filepath)
    chunks = chunk_blocks(blocks)
    result = embed_and_save(chunks, tenant_id)
    logger.info("Ingestion complete: %s", result)
    return result
