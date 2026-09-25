from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from .hashing import NGramHasher
from .namespaces import NGramNamespace


@dataclass(frozen=True)
class NamespaceCollisionStats:
    namespace: str
    unique_keys: int
    bucket_count: int
    occupied_buckets: int
    collision_rate: float
    load_factor: float
    max_bucket_depth: int


def collision_stats(key_counts: dict[NGramNamespace, Counter[str]], hasher: NGramHasher):
    out = []
    for ns, counts in key_counts.items():
        buckets: dict[int, set[str]] = defaultdict(set)
        for key in counts:
            _, bucket = hasher.feature_id(ns, key)
            buckets[bucket].add(key)
        unique = len(counts)
        occupied = len(buckets)
        out.append(NamespaceCollisionStats(
            namespace=ns.value,
            unique_keys=unique,
            bucket_count=hasher.config.namespaces[ns].bucket_count,
            occupied_buckets=occupied,
            collision_rate=(1.0 - occupied / unique) if unique else 0.0,
            load_factor=unique / hasher.config.namespaces[ns].bucket_count,
            max_bucket_depth=max((len(v) for v in buckets.values()), default=0),
        ))
    return out
