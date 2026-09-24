from __future__ import annotations
from collections import defaultdict
from taxmoe.ingestion.hashing import stable_hash, stable_int
from taxmoe.schemas.manifest import SplitManifest, SplitAssignment
from taxmoe.schemas.enums import DatasetSplit
from taxmoe.schemas.scenario import TaxScenario
from .family_graph import DisjointSet

class SplitEngine:
    def __init__(self, seed: int = 42017, split_version: str = "v1", clustering_version: str = "v1"):
        self.seed = seed
        self.split_version = split_version
        self.clustering_version = clustering_version

    def build(self, scenarios: list[TaxScenario]) -> SplitManifest:
        ds = DisjointSet()
        by_family = defaultdict(list)
        by_struct = defaultdict(list)

        for s in scenarios:
            ds.add(s.scenario_id)
            by_family[s.family_id].append(s.scenario_id)
            by_struct[s.structural_fingerprint].append(s.scenario_id)

        for group in list(by_family.values()) + list(by_struct.values()):
            for x in group[1:]:
                ds.union(group[0], x)

        groups = ds.groups()
        scenario_to_cluster = {}
        assignments = []

        for group in groups:
            key = stable_hash(sorted(group))
            cluster_id = f"CLUSTER-{key[:20]}"
            for sid in group:
                scenario_to_cluster[sid] = cluster_id

            bucket = stable_int(self.split_version, self.seed, cluster_id) % 10000
            if bucket < 8000:
                split = DatasetSplit.TRAIN
            elif bucket < 9000:
                split = DatasetSplit.VALIDATION
            else:
                split = DatasetSplit.TEST_ID

            assignments.append(SplitAssignment(
                cluster_id=cluster_id,
                split=split.value,
                split_version=self.split_version,
                split_seed=self.seed,
            ))

        return SplitManifest(
            split_version=self.split_version,
            clustering_version=self.clustering_version,
            dedup_version="v1",
            seed=self.seed,
            scenario_to_cluster=scenario_to_cluster,
            assignments=sorted(assignments, key=lambda x: x.cluster_id),
        )
