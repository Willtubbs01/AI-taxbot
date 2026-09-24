from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from json import dumps
from typing import Any

from pydantic import BaseModel, ConfigDict


class TaxMoEModel(BaseModel):
    """Shared strict Pydantic base model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=False)
        return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def content_hash(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def stable_hash(*parts: Any) -> str:
    serialized = "\x1f".join(
        dumps(part, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        for part in parts
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
