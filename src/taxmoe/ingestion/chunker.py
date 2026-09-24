from __future__ import annotations
from taxmoe.schemas.common import stable_hash
from .....taxmoe.ingestion.models import SourceChunk


def chunk_text(source_id: str, text: str, max_chars: int = 4000) -> list[SourceChunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[SourceChunk] = []
    buffer = ""
    ordinal = 0
    for paragraph in paragraphs:
        candidate = paragraph if not buffer else buffer + "\n\n" + paragraph
        if buffer and len(candidate) > max_chars:
            chunks.append(SourceChunk(
                chunk_id=f"CHUNK-{stable_hash(source_id, ordinal, buffer)[:20]}",
                source_id=source_id,
                text=buffer,
                ordinal=ordinal,
            ))
            ordinal += 1
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(SourceChunk(
            chunk_id=f"CHUNK-{stable_hash(source_id, ordinal, buffer)[:20]}",
            source_id=source_id,
            text=buffer,
            ordinal=ordinal,
        ))
    return chunks
