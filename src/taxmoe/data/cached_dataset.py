from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset

class CachedTaxMoEDataset(Dataset):
    def __init__(self, shard_paths:list[str|Path], feature_width:int=24):
        self.feature_width=feature_width
        self.tables=[]; self.offsets=[]; total=0
        for p in shard_paths:
            source=pa.memory_map(str(p),'r')
            reader=ipc.open_file(source)
            table=reader.read_all()
            self.tables.append((source,table))
            total += table.num_rows
            self.offsets.append(total)
    def __len__(self): return self.offsets[-1] if self.offsets else 0
    def _locate(self,index):
        if index<0:index+=len(self)
        if index<0 or index>=len(self): raise IndexError(index)
        prev=0
        for si,end in enumerate(self.offsets):
            if index<end:return si,index-prev
            prev=end
        raise IndexError(index)
    def __getitem__(self,index):
        si,ri=self._locate(index); table=self.tables[si][1]
        def v(name):return table[name][ri].as_py()
        out={'record_id':v('record_id'),'input_ids':v('input_ids'),'labels':v('labels'),'token_count':v('token_count'),'supervised_token_count':v('supervised_token_count')}
        if 'ngram_feature_ids_flat' in table.column_names:
            flat=v('ngram_feature_ids_flat'); n=out['token_count']; k=self.feature_width
            if len(flat)!=n*k: raise ValueError('CACHE-NGRAM-WIDTH-MISMATCH')
            out['ngram_feature_ids']=[flat[i*k:(i+1)*k] for i in range(n)]
        return out
