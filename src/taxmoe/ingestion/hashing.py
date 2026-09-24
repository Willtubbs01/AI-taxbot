from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

def stable_hash(*parts: Any) -> str:
    return sha256_bytes(canonical_json_bytes(parts))

def stable_int(*parts: Any, bits: int = 64) -> int:
    digest = stable_hash(*parts)
    hex_chars = bits // 4
    return int(digest[:hex_chars], 16)
