from collections import Counter, defaultdict
from .namespaces import namespace_for

def collect_key_counts(results):
    out=defaultdict(Counter)
    for result in results:
        for occ in result.occurrences:
            out[namespace_for(occ.kind,occ.order)][occ.normalized_key]+=1
    return out
