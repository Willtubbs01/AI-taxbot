from __future__ import annotations
import json

def strict_json_metrics(text, required_keys=()):
    try: obj=json.loads(text)
    except Exception: return {'json_valid':0.0,'schema_valid':0.0,'value':None}
    valid=isinstance(obj,dict) and all(k in obj for k in required_keys)
    return {'json_valid':1.0,'schema_valid':float(valid),'value':obj}
