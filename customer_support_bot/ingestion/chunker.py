import hashlib
from dataclasses import dataclass
from typing import List

from customer_support_bot.config import CONFIG
from customer_support_bot.ingestion.document_loader import ContentBlock


@dataclass
class Chunk:
    text: str
    metadata: dict


def chunk_blocks(blocks: List[ContentBlock]) -> List[Chunk]:
    chunks = []
    current_text = ""
    current_metadata = {}
    chunk_index = 0

    def flush_chunk(text: str, metadata: dict, index: int):
        text = text.strip()
        if len(text) < 50:
            return None
        chunk_id = hashlib.md5(text.encode()).hexdigest()
        return Chunk(
            text=text,
            metadata={
                **metadata,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "char_count": len(text),
            },
        )

    for block in blocks:
        if block.block_type == "table":
            if current_text.strip():
                chunk = flush_chunk(current_text, current_metadata, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                current_text = ""

            if len(block.text) <= CONFIG.chunk_max_chars:
                chunk = flush_chunk(block.text, block.metadata, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                rows = block.text.split("\n")
                table_chunk_text = ""
                for row in rows:
                    if len(table_chunk_text) + len(row) > CONFIG.chunk_max_chars:
                        chunk = flush_chunk(table_chunk_text, block.metadata, chunk_index)
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1
                        table_chunk_text = row + "\n"
                    else:
                        table_chunk_text += row + "\n"
                if table_chunk_text.strip():
                    chunk = flush_chunk(table_chunk_text, block.metadata, chunk_index)
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
            continue

        if block.block_type == "heading":
            if current_text.strip():
                chunk = flush_chunk(current_text, current_metadata, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
            current_text = block.text + "\n"
            current_metadata = block.metadata
            continue

        if not current_metadata:
            current_metadata = block.metadata

        if len(current_text) + len(block.text) <= CONFIG.chunk_max_chars:
            current_text += block.text + "\n"
        else:
            if current_text.strip():
                chunk = flush_chunk(current_text, current_metadata, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
            overlap_text = current_text[-CONFIG.chunk_overlap_chars:] if current_text else ""
            current_text = overlap_text + block.text + "\n"
            current_metadata = block.metadata

    if current_text.strip():
        chunk = flush_chunk(current_text, current_metadata, chunk_index)
        if chunk:
            chunks.append(chunk)

    return chunks
