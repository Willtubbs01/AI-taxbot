from __future__ import annotations
import json
from pathlib import Path
from pydantic import Field
from taxmoe.modeling.base import Stage4Model
from taxmoe.modeling.inputs import ChatMessage

class CPTInputRecord(Stage4Model):
    record_id: str
    text: str
    split: str = "train"
    source_id: str | None = None
    chunk_id: str | None = None

class SFTInputRecord(Stage4Model):
    record_id: str
    messages: list[ChatMessage]
    split: str = "train"
    metadata: dict = Field(default_factory=dict)

    def validate_v01_conversation(self):
        if not self.messages or self.messages[-1].role != "assistant":
            raise ValueError("SFT v0.1 requires a final assistant message")
        if sum(m.role == "assistant" for m in self.messages) != 1:
            raise ValueError("SFT v0.1 requires exactly one assistant target")
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("SFT v0.1 requires a user message")
        return self


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno,line in enumerate(f,1):
            if not line.strip(): continue
            yield lineno,json.loads(line)


def read_cpt(path: str | Path):
    for _, row in iter_jsonl(path):
        yield CPTInputRecord(
            record_id=row.get("record_id") or row.get("id") or row.get("example_id"),
            text=row["text"],
            split=row.get("split") or row.get("metadata",{}).get("split","train"),
            source_id=row.get("source_id") or row.get("metadata",{}).get("source_id"),
            chunk_id=row.get("chunk_id") or row.get("metadata",{}).get("chunk_id"),
        )


def read_sft(path: str | Path):
    for _, row in iter_jsonl(path):
        messages = [ChatMessage(**m) for m in row["messages"]]
        record_id = row.get("record_id") or row.get("id") or row.get("example_id")
        if not record_id:
            raise ValueError("SFT record is missing record_id/id/example_id")
        metadata = dict(row.get("metadata", {}))
        if row.get("dataset_family") is not None:
            metadata["dataset_family"] = row["dataset_family"]
        if row.get("task_id") is not None:
            metadata["task_id"] = row["task_id"]
        rec = SFTInputRecord(
            record_id=record_id,
            messages=messages,
            split=row.get("split") or metadata.get("split", "train"),
            metadata=metadata,
        )
        rec.validate_v01_conversation()
        yield rec
