from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from pydantic import Field
from pypdf import PdfReader

from taxmoe.data.readers import CPTInputRecord
from taxmoe.ingestion.hashing import sha256_file, stable_hash, stable_int
from taxmoe.ingestion.registry import SourceRegistry
from taxmoe.schemas.common import TaxMoEModel


class CPTSourceSummary(TaxMoEModel):
    source_id: str
    title: str
    raw_path: str
    sha256: str
    pages: int
    records: int
    characters: int


class CPTCorpusManifest(TaxMoEModel):
    schema_version: str = "0.1"
    corpus_id: str
    source_manifest_id: str
    source_manifest_hash: str
    tax_year: int | None = None
    extraction_backend: str = "pypdf"
    chunk_max_chars: int = 6000
    validation_percent: int = 10
    records: int = 0
    train_records: int = 0
    validation_records: int = 0
    characters: int = 0
    source_summaries: list[CPTSourceSummary] = Field(default_factory=list)
    content_hash: str = ""


def normalize_pdf_text(text: str) -> str:
    """Deterministic, conservative normalization for PDF-extracted IRS text."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    previous_blank = False
    for raw in text.split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw).strip()
        if not line:
            if lines and not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if max_chars < 512:
        raise ValueError("CPT-CHUNK-MAX-CHARS-TOO-SMALL")
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                if end < len(paragraph):
                    boundary = paragraph.rfind(" ", start, end)
                    if boundary > start + max_chars // 2:
                        end = boundary
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                start = end
                while start < len(paragraph) and paragraph[start].isspace():
                    start += 1
            continue

        projected = current_len + (2 if current else 0) + len(paragraph)
        if current and projected > max_chars:
            flush()
        current.append(paragraph)
        current_len += (2 if len(current) > 1 else 0) + len(paragraph)
    flush()
    return chunks


def _initial_split(record_id: str, validation_percent: int) -> str:
    if validation_percent <= 0:
        return "train"
    if validation_percent >= 100:
        return "validation"
    return "validation" if stable_int("cpt-split-v1", record_id, bits=32) % 100 < validation_percent else "train"


def build_cpt_records(
    source_manifest: str | Path,
    *,
    project_root: str | Path = ".",
    chunk_max_chars: int = 6000,
    validation_percent: int = 10,
) -> tuple[list[CPTInputRecord], CPTCorpusManifest]:
    manifest_path = Path(source_manifest)
    registry = SourceRegistry.from_yaml(manifest_path)
    project_root = Path(project_root)
    source_manifest_hash = sha256_file(manifest_path)

    records: list[CPTInputRecord] = []
    summaries: list[CPTSourceSummary] = []
    tax_years: set[int] = set()

    for source in registry.all():
        if source.tax_year is not None:
            tax_years.add(source.tax_year)
        raw_path = project_root / source.raw_path
        if not raw_path.exists():
            raise FileNotFoundError(f"CPT-SOURCE-MISSING:{raw_path}")
        actual_sha = sha256_file(raw_path)
        if actual_sha.lower() != source.sha256.lower():
            raise ValueError(f"CPT-SOURCE-HASH-MISMATCH:{source.source_id}")
        if raw_path.suffix.lower() != ".pdf":
            raise ValueError(f"CPT-SOURCE-UNSUPPORTED-TYPE:{raw_path.suffix}")

        reader = PdfReader(str(raw_path))
        source_records = 0
        source_characters = 0
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = normalize_pdf_text(page.extract_text() or "")
            if not page_text:
                continue
            chunks = _split_long_text(page_text, chunk_max_chars)
            for chunk_index, chunk in enumerate(chunks, start=1):
                record_id = f"CPT-{source.source_id}-P{page_index:04d}-C{chunk_index:03d}"
                chunk_id = f"{source.source_id}:page={page_index}:chunk={chunk_index}"
                records.append(
                    CPTInputRecord(
                        record_id=record_id,
                        text=chunk,
                        split=_initial_split(record_id, validation_percent),
                        source_id=source.source_id,
                        chunk_id=chunk_id,
                    )
                )
                source_records += 1
                source_characters += len(chunk)
        summaries.append(
            CPTSourceSummary(
                source_id=source.source_id,
                title=source.title,
                raw_path=source.raw_path,
                sha256=actual_sha,
                pages=len(reader.pages),
                records=source_records,
                characters=source_characters,
            )
        )

    if not records:
        raise ValueError("CPT-CORPUS-EMPTY")

    # A tiny source set can hash entirely to train. Preserve a validation signal
    # without modifying record contents.
    if validation_percent > 0 and len(records) > 1 and not any(r.split == "validation" for r in records):
        selected = min(records, key=lambda r: stable_int("cpt-validation-fallback-v1", r.record_id, bits=64))
        selected_index = next(i for i, r in enumerate(records) if r.record_id == selected.record_id)
        records[selected_index] = selected.model_copy(update={"split": "validation"})

    records.sort(key=lambda r: (r.source_id or "", r.chunk_id or "", r.record_id))
    corpus_payload = [r.model_dump(mode="json") for r in records]
    corpus_id = "CPT-" + stable_hash(source_manifest_hash, corpus_payload)[:20]
    manifest = CPTCorpusManifest(
        corpus_id=corpus_id,
        source_manifest_id=registry.manifest.manifest_id,
        source_manifest_hash=source_manifest_hash,
        tax_year=next(iter(tax_years)) if len(tax_years) == 1 else None,
        chunk_max_chars=chunk_max_chars,
        validation_percent=validation_percent,
        records=len(records),
        train_records=sum(r.split == "train" for r in records),
        validation_records=sum(r.split == "validation" for r in records),
        characters=sum(len(r.text) for r in records),
        source_summaries=summaries,
    )
    manifest = manifest.model_copy(
        update={"content_hash": stable_hash(manifest.model_dump(exclude={"content_hash"}), corpus_payload)}
    )
    return records, manifest


def write_cpt_corpus(
    output_dir: str | Path,
    records: Iterable[CPTInputRecord],
    manifest: CPTCorpusManifest,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "cpt_records.jsonl"
    tmp = records_path.with_suffix(records_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(records_path)
    manifest_path = output_dir / "cpt_corpus_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return records_path, manifest_path
