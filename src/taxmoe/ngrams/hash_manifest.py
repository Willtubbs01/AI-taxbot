from __future__ import annotations

import hashlib, json
from taxmoe.modeling.base import Stage4Model
from .hashing import NGramHashConfig


class NGramHashManifest(Stage4Model):
    version: str
    algorithm: str
    key_id: str
    namespaces: dict[str, dict[str, int]]
    real_bucket_count: int
    embedding_rows: int
    manifest_hash: str

    @classmethod
    def from_config(cls, config: NGramHashConfig):
        payload = {
            "version": config.version,
            "algorithm": config.algorithm,
            "key_id": config.key_id,
            "namespaces": {k.value: v.model_dump() for k, v in config.namespaces.items()},
            "real_bucket_count": config.real_bucket_count,
            "embedding_rows": config.embedding_rows,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(**payload, manifest_hash=digest)
