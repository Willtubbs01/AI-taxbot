from __future__ import annotations
import hashlib
from pathlib import Path
from .cache_models import TrainingCacheManifest

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def verify_cache(manifest:TrainingCacheManifest,root:str|Path):
    root=Path(root)
    records=tokens=0
    for shard in manifest.files:
        p=root/shard.path
        if not p.exists():raise FileNotFoundError(p)
        if p.stat().st_size!=shard.bytes:raise ValueError(f'cache shard size mismatch: {p}')
        if sha256_file(p)!=shard.sha256:raise ValueError(f'cache shard hash mismatch: {p}')
        records+=shard.records;tokens+=shard.tokens
    if records!=manifest.total_records or tokens!=manifest.total_tokens:raise ValueError('cache totals mismatch')
    return True
