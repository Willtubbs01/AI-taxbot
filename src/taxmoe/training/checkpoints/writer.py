from __future__ import annotations
import json, shutil, uuid
from pathlib import Path
import torch
from taxmoe.ingestion.hashing import sha256_file,stable_hash
from .models import ArtifactFile,TrainingCheckpointManifest,CheckpointType
from .rng import capture_rng_state

def _rec(root,p):
    p=Path(p); return ArtifactFile(path=str(p.relative_to(root)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha256_file(p))
class CheckpointWriter:
    def save(self,root,*,checkpoint_id,run_manifest,progress,model,optimizer=None,scheduler=None,scaler=None,sampler_state=None,training_data_manifest_hash='',model_architecture_hash='',training_config_hash='',checkpoint_type=CheckpointType.FULL):
        root=Path(root); final=root/checkpoint_id
        if final.exists(): raise FileExistsError(final)
        tmp=root/'.tmp'/f'{checkpoint_id}-{uuid.uuid4().hex[:8]}'; tmp.mkdir(parents=True,exist_ok=True)
        model_dir=tmp/'model'; model.save_pretrained(model_dir,safe_serialization=True)
        model_files=[_rec(tmp,p) for p in sorted(model_dir.glob('*')) if p.is_file()]
        opt=sched=scal=rng=samp=prog=None
        if checkpoint_type==CheckpointType.FULL:
            if optimizer is None or scheduler is None: raise ValueError('CKPT-FULL-STATE-MISSING')
            torch.save(optimizer.state_dict(),tmp/'optimizer.pt'); opt=_rec(tmp,tmp/'optimizer.pt')
            torch.save(scheduler.state_dict(),tmp/'scheduler.pt'); sched=_rec(tmp,tmp/'scheduler.pt')
            if scaler is not None: torch.save(scaler.state_dict(),tmp/'scaler.pt'); scal=_rec(tmp,tmp/'scaler.pt')
            torch.save(capture_rng_state(),tmp/'rng_state.pt'); rng=_rec(tmp,tmp/'rng_state.pt')
            (tmp/'sampler_state.json').write_text(json.dumps(sampler_state or {},sort_keys=True,indent=2)+'\n'); samp=_rec(tmp,tmp/'sampler_state.json')
            (tmp/'progress.json').write_text(progress.model_dump_json(indent=2)+'\n'); prog=_rec(tmp,tmp/'progress.json')
        m=TrainingCheckpointManifest(checkpoint_id=checkpoint_id,checkpoint_type=checkpoint_type,run_id=run_manifest.run_id,run_spec_hash=run_manifest.run_spec_hash,global_step=progress.global_step,consumed_training_tokens=progress.consumed_tokens,effective_passes=progress.effective_passes,parent_model_fingerprint=run_manifest.parent_model_fingerprint,input_release_hash=run_manifest.input_release_hash,training_data_manifest_hash=training_data_manifest_hash,model_architecture_hash=model_architecture_hash,training_config_hash=training_config_hash,model_files=model_files,optimizer_file=opt,scheduler_file=sched,scaler_file=scal,rng_file=rng,sampler_state_file=samp,progress_file=prog,complete=True)
        m=m.model_copy(update={'checkpoint_content_hash':stable_hash(m.model_dump(exclude={'checkpoint_content_hash'}))})
        (tmp/'checkpoint_manifest.json').write_text(m.model_dump_json(indent=2)+'\n')
        final.parent.mkdir(parents=True,exist_ok=True); tmp.replace(final); return final,m
