from __future__ import annotations
from pathlib import Path
from .....taxmoe.ingestion.models import ExtractedDocument


def extract_text(source_id: str, path: str | Path) -> ExtractedDocument:
    path = Path(path)
    if path.suffix.lower() in {".txt", ".md", ".html", ".htm"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractedDocument(source_id=source_id, text=text, pages=[text], extractor="text")
    raise ValueError(f"No extractor configured for {path.suffix}; add a PDF extractor in your implementation pass")
