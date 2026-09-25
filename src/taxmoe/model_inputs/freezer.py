from __future__ import annotations
import shutil,uuid
from pathlib import Path
from .release_models import InputReleaseStatus,ModelInputReleaseManifest
from .verifier import release_manifest_hash

class InputReleaseFreezer:
    def freeze(self,candidate_dir:str|Path,releases_root:str|Path,manifest:ModelInputReleaseManifest):
        if not manifest.version:raise ValueError('release version required')
        if manifest.status not in (InputReleaseStatus.CANDIDATE,InputReleaseStatus.FROZEN):raise ValueError('candidate not freezeable')
        candidate_dir=Path(candidate_dir);releases_root=Path(releases_root)
        destination=releases_root/manifest.name
        if destination.exists():raise FileExistsError('INPUT-FREEZE-RELEASE-EXISTS')
        staging=releases_root/'.staging'/f'{manifest.release_id}-{uuid.uuid4().hex[:8]}'
        staging.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(candidate_dir,staging)
        final=manifest.model_copy(update={'status':InputReleaseStatus.FROZEN})
        final=final.model_copy(update={'release_content_hash':release_manifest_hash(final)})
        mpath=staging/'manifests'/'input_release_manifest.json';mpath.parent.mkdir(parents=True,exist_ok=True)
        mpath.write_text(final.model_dump_json(indent=2)+'\n',encoding='utf-8')
        destination.parent.mkdir(parents=True,exist_ok=True);staging.replace(destination)
        return destination,final
