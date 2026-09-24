from __future__ import annotations
from pathlib import Path
from ...src.taxmoe.taxmoe.ingestion.extractor import extract_text
from ...src.taxmoe.taxmoe.ingestion.normalizer import normalize_text
from ...src.taxmoe.taxmoe.ingestion.chunker import chunk_text


def process_source(source_id: str, path: str | Path, max_chars: int = 4000):
    extracted = extract_text(source_id, path)
    normalized = normalize_text(extracted.text)
    chunks = chunk_text(source_id, normalized, max_chars=max_chars)
    return extracted, normalized, chunks
