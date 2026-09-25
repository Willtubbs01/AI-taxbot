from __future__ import annotations

import json

from taxmoe.data.readers import read_sft


def test_stage3_example_id_is_accepted(tmp_path):
    path = tmp_path / "stage3.jsonl"
    row = {
        "example_id": "EXAMPLE-123",
        "dataset_family": "tax_identify_topics",
        "task_id": "identify_topics",
        "split": "train",
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
        ],
        "metadata": {"tax_year": 2025},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = list(read_sft(path))
    assert len(records) == 1
    assert records[0].record_id == "EXAMPLE-123"
    assert records[0].metadata["dataset_family"] == "tax_identify_topics"
    assert records[0].metadata["task_id"] == "identify_topics"
