from __future__ import annotations
import shutil,uuid
from pathlib import Path
from taxmoe.ingestion.hashing import sha256_file,stable_hash
from .models import TaxDenseReleaseManifest,ModelReleaseStatus,ReleaseFile
from .verifier import manifest_hash
class TaxDenseFreezer:
    def freeze(self,candidate_model_dir,releases_root,manifest:TaxDenseReleaseManifest):
        if not manifest.optimization_config_hash: raise ValueError('TAXDENSE-FREEZE-OPTIMIZATION-SPEC-MISSING')
        src=Path(candidate_model_dir); root=Path(releases_root); dest=root/f'{manifest.name}-v{manifest.version}'
        if dest.exists(): raise FileExistsError(dest)
        stage=root/'.staging'/f'{manifest.release_id}-{uuid.uuid4().hex[:8]}'; shutil.copytree(src,stage)
        files=[]
        for p in sorted(stage.rglob('*')):
            if p.is_file() and p.name!='model_manifest.json': files.append(ReleaseFile(path=str(p.relative_to(stage)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha256_file(p)))
        frozen=manifest.model_copy(update={'status':ModelReleaseStatus.FROZEN,'model_files':files})
        frozen=frozen.model_copy(update={'release_content_hash':manifest_hash(frozen)})
        (stage/'model_manifest.json').write_text(frozen.model_dump_json(indent=2)+'\n')
        dest.parent.mkdir(parents=True,exist_ok=True); stage.replace(dest); return dest,frozen
