from __future__ import annotations
import hashlib,json
from .release_models import InputReproducibilityResult

def semantic_hash(obj)->str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def compare_rebuilds(candidate:dict,reproduction:dict,candidate_build_id:str,reproduction_build_id:str):
    keys=['tokenization','ngram_hash','alignment','cache_semantic','cache_files']
    matches={k:semantic_hash(candidate.get(k))==semantic_hash(reproduction.get(k)) for k in keys}
    return InputReproducibilityResult(
        passed=all(matches.values()),candidate_build_id=candidate_build_id,reproduction_build_id=reproduction_build_id,
        tokenization_match=matches['tokenization'],ngram_hash_match=matches['ngram_hash'],alignment_match=matches['alignment'],cache_semantic_match=matches['cache_semantic'],cache_file_hash_match=matches['cache_files'],mismatches=[k for k,v in matches.items() if not v])
