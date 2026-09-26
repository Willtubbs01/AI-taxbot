from pathlib import Path
from taxmoe.ingestion.hashing import sha256_file,stable_hash
from .models import TaxDenseReleaseManifest,ModelReleaseStatus
def manifest_hash(m): return stable_hash(m.model_dump(exclude={'release_content_hash'}))
def verify_taxdense_release(root):
    root=Path(root); m=TaxDenseReleaseManifest.model_validate_json((root/'model_manifest.json').read_text())
    if m.status!=ModelReleaseStatus.FROZEN: raise ValueError('TAXDENSE-RELEASE-NOT-FROZEN')
    if manifest_hash(m)!=m.release_content_hash: raise ValueError('TAXDENSE-RELEASE-MANIFEST-HASH-MISMATCH')
    for f in m.model_files:
        p=root/f.path
        if not p.exists(): raise FileNotFoundError(f.path)
        if sha256_file(p)!=f.sha256: raise ValueError(f'TAXDENSE-RELEASE-FILE-HASH-MISMATCH:{f.path}')
    return m
