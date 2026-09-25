from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Iterable
import pyarrow as pa
import pyarrow.ipc as ipc
from .cache_models import CachedTrainingExample, CacheShard


def _sha_file(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def example_content_hash(record_id,input_ids,labels,ngrams=None):
    obj={'record_id':record_id,'input_ids':input_ids,'labels':labels,'ngram_feature_ids':ngrams}
    return hashlib.sha256(json.dumps(obj,separators=(",",":"),sort_keys=True).encode()).hexdigest()

class ArrowCacheBuilder:
    def __init__(self, feature_width:int=24):self.feature_width=feature_width

    def write_shard(self, examples:Iterable[CachedTrainingExample], path:str|Path, split:str) -> CacheShard:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        rows=list(examples)
        if not rows: raise ValueError('cannot write empty cache shard')
        has_ngram=rows[0].ngram_feature_ids is not None
        if any((r.ngram_feature_ids is not None)!=has_ngram for r in rows): raise ValueError('mixed cache variants in one shard')
        data={
            'record_id':[r.record_id for r in rows],
            'split':[r.split for r in rows],
            'source_kind':[r.source_kind for r in rows],
            'token_count':pa.array([r.token_count for r in rows],type=pa.int32()),
            'supervised_token_count':pa.array([r.supervised_token_count for r in rows],type=pa.int32()),
            'input_ids':pa.array([r.input_ids for r in rows],type=pa.list_(pa.int32())),
            'labels':pa.array([r.labels for r in rows],type=pa.list_(pa.int32())),
            'content_hash':[r.content_hash for r in rows],
        }
        if has_ngram:
            flat=[[x for row in r.ngram_feature_ids for x in row] for r in rows]
            data['ngram_feature_ids_flat']=pa.array(flat,type=pa.list_(pa.int32()))
        table=pa.table(data)
        tmp=path.with_suffix(path.suffix+'.tmp')
        with pa.OSFile(str(tmp),'wb') as sink:
            with ipc.new_file(sink,table.schema) as writer: writer.write_table(table)
        tmp.replace(path)
        return CacheShard(split=split,path=path.name,records=len(rows),tokens=sum(r.token_count for r in rows),bytes=path.stat().st_size,sha256=_sha_file(path),first_record_id=rows[0].record_id,last_record_id=rows[-1].record_id)
