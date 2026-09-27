from pathlib import Path

from taxmoe.ingestion.cpt import build_cpt_records, normalize_pdf_text


def test_normalize_pdf_text_is_conservative_and_deterministic():
    value = " Form  1040\r\n\r\n  Tax\tYear 2025  \n"
    assert normalize_pdf_text(value) == "Form 1040\n\nTax Year 2025"


def test_real_irs_manifest_builds_cpt_records():
    root = Path(__file__).resolve().parents[3]
    records, manifest = build_cpt_records(
        root / "sources/registry/source_manifest_2025.yaml",
        project_root=root,
        chunk_max_chars=6000,
        validation_percent=10,
    )
    assert records
    assert manifest.records == len(records)
    assert manifest.train_records > 0
    assert manifest.validation_records > 0
    assert {r.source_id for r in records} >= {
        "SOURCE-2025-W2",
        "SOURCE-2025-1040",
        "SOURCE-2025-1099B",
        "SOURCE-2025-8949",
        "SOURCE-2025-SCHEDULE-D",
    }
    assert all(r.text.strip() for r in records)
    assert manifest.content_hash
