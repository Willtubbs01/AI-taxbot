from __future__ import annotations
import hashlib,json
from .cache_models import TrainingCacheManifest

def compute_cache_content_hash(manifest: TrainingCacheManifest) -> str:
    payload=manifest.model_dump(exclude={'content_hash'})
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
