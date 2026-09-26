from __future__ import annotations
import json, torch
from pathlib import Path
from .models import TrainingCheckpointManifest,CheckpointType
from .rng import restore_rng_state
from taxmoe.ingestion.hashing import sha256_file,stable_hash
class CheckpointLoader:
    def verify(self,path):
        path=Path(path); m=TrainingCheckpointManifest.model_validate_json((path/'checkpoint_manifest.json').read_text())
        if not m.complete: raise ValueError('CKPT-INCOMPLETE')
        for rec in m.model_files+[x for x in [m.optimizer_file,m.scheduler_file,m.scaler_file,m.rng_file,m.sampler_state_file,m.progress_file] if x]:
            p=path/rec.path
            if not p.exists(): raise FileNotFoundError(f'CKPT-FILE-MISSING:{rec.path}')
            if sha256_file(p)!=rec.sha256: raise ValueError(f'CKPT-FILE-HASH-MISMATCH:{rec.path}')
        if stable_hash(m.model_dump(exclude={'checkpoint_content_hash'}))!=m.checkpoint_content_hash: raise ValueError('CKPT-MANIFEST-HASH-MISMATCH')
        return m
    def restore_training_state(self,path,optimizer,scheduler,scaler=None):
        path=Path(path); m=self.verify(path)
        if m.checkpoint_type!=CheckpointType.FULL: raise ValueError('CKPT-NOT-RESUMABLE')
        optimizer.load_state_dict(torch.load(path/m.optimizer_file.path,map_location='cpu',weights_only=False)); scheduler.load_state_dict(torch.load(path/m.scheduler_file.path,map_location='cpu',weights_only=False))
        if m.scaler_file and scaler is not None: scaler.load_state_dict(torch.load(path/m.scaler_file.path,map_location='cpu',weights_only=False))
        restore_rng_state(torch.load(path/m.rng_file.path,map_location='cpu',weights_only=False)); sampler=json.loads((path/m.sampler_state_file.path).read_text()); return m,sampler
