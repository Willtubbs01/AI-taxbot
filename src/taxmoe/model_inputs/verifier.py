from __future__ import annotations
import hashlib,json
from pathlib import Path
from .release_models import ModelInputReleaseManifest

def release_manifest_hash(manifest:ModelInputReleaseManifest):
    payload=manifest.model_dump(exclude={'release_content_hash'},mode='json')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def verify_release_manifest(manifest:ModelInputReleaseManifest):
    if manifest.status.value!='frozen':raise ValueError('input release is not frozen')
    if release_manifest_hash(manifest)!=manifest.release_content_hash:raise ValueError('input release content hash mismatch')
    return True
