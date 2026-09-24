from __future__ import annotations

from collections import defaultdict

from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel, stable_hash
from taxmoe.schemas.enums import DatasetSplit
from taxmoe.schemas.scenario import TaxScenario


class SplitAssignment(TaxMoEModel):
    cluster_id: str
    split: DatasetSplit


class SplitConfig(TaxMoEModel):
    version: str = "v1"
    seed: int = 42017
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    reserved_clusters: dict[str, DatasetSplit] = Field(default_factory=dict)


class SplitEngine:
    def __init__(self, config: SplitConfig):
        total = config.train_fraction + config.validation_fraction + config.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Standard split fractions must sum to 1")
        self.config = config

    def build_clusters(self, scenarios: list[TaxScenario]) -> dict[str, list[TaxScenario]]:
        # v0.1 cluster key = structural fingerprint. Family siblings therefore stay together
        # when their structural fingerprint is shared; explicit family union closes the rest.
        family_to_structures: dict[str, set[str]] = defaultdict(set)
        for s in scenarios:
            family_to_structures[str(s.family_id)].add(s.structural_fingerprint)

        # Families linked to multiple structures get a family-derived cluster key so mutations
        # cannot leak across splits.
        clusters: dict[str, list[TaxScenario]] = defaultdict(list)
        for s in scenarios:
            structures = family_to_structures[str(s.family_id)]
            if len(structures) > 1:
                key = stable_hash("family", str(s.family_id))
            else:
                key = next(iter(structures))
            cluster_id = f"CLUSTER-{key[:20]}"
            clusters[cluster_id].append(s)
        return dict(clusters)

    def assign(self, clusters: dict[str, list[TaxScenario]]) -> dict[str, SplitAssignment]:
        out: dict[str, SplitAssignment] = {}
        train_cut = self.config.train_fraction
        val_cut = train_cut + self.config.validation_fraction
        for cluster_id in sorted(clusters):
            if cluster_id in self.config.reserved_clusters:
                split = self.config.reserved_clusters[cluster_id]
            else:
                value = int(stable_hash(self.config.version, self.config.seed, cluster_id)[:12], 16) / float(16**12)
                if value < train_cut:
                    split = DatasetSplit.TRAIN
                elif value < val_cut:
                    split = DatasetSplit.VALIDATION
                else:
                    split = DatasetSplit.TEST_ID
            out[cluster_id] = SplitAssignment(cluster_id=cluster_id, split=split)
        return out
