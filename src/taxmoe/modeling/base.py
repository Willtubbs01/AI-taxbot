from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Stage4Model(BaseModel):
    """Strict base model for deterministic Stage 4 artifacts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
