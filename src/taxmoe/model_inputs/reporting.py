from __future__ import annotations
import json
from pathlib import Path

def write_json_report(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(obj,'model_dump'):obj=obj.model_dump(mode='json')
    elif hasattr(obj,'__dict__'):obj=obj.__dict__
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
    return p
