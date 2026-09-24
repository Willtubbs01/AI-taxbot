from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from taxmoe.exporters.dataset_exporter import DatasetExporter
from taxmoe.generation.mutation_generator import MutationGenerator
from taxmoe.generation.scenario_generator import ScenarioGenerator
from taxmoe.schemas.common import TaxMoEModel, stable_hash, utc_now
from taxmoe.schemas.enums import MutationTruthEffect
from taxmoe.splitting.splitter import SplitConfig, SplitEngine
from taxmoe.validation.build import BuildValidator


class TemplateBuildSpec(TaxMoEModel):
    id: str
    count: int
    tax_year: int


class MutationBuildSpec(TaxMoEModel):
    concept_id: str
    per_base: int = 0
    truth_effect: MutationTruthEffect = MutationTruthEffect.MAY_CHANGE_ANALYSIS


class DatasetBuildConfig(TaxMoEModel):
    name: str
    seed: int = 42017
    templates: list[TemplateBuildSpec]
    mutations: list[MutationBuildSpec] = Field(default_factory=list)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetBuildConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class DatasetBuildRunner:
    def __init__(
        self,
        generator: ScenarioGenerator,
        mutation_generator: MutationGenerator,
        validator: BuildValidator,
        exporter: DatasetExporter,
    ):
        self.generator = generator
        self.mutation_generator = mutation_generator
        self.validator = validator
        self.exporter = exporter

    def run(self, config: DatasetBuildConfig, output_root: str | Path) -> dict[str, Any]:
        output_root = Path(output_root)
        build_id = f"BUILD-{stable_hash(config.model_dump(mode='json'), utc_now().isoformat())[:20]}"
        build_dir = output_root / build_id
        build_dir.mkdir(parents=True, exist_ok=False)
        (build_dir / "config.snapshot.json").write_text(
            json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2), encoding="utf-8"
        )

        scenarios = []
        for spec in config.templates:
            for index in range(spec.count):
                seed = int(stable_hash(config.seed, spec.id, index)[:16], 16)
                scenarios.append(self.generator.generate(spec.id, spec.tax_year, seed, index))

        base_scenarios = list(scenarios)
        for m in config.mutations:
            if m.per_base <= 0:
                continue
            for base in base_scenarios:
                for mutation_index in range(m.per_base):
                    seed = int(stable_hash(config.seed, base.scenario_id, m.concept_id, mutation_index)[:16], 16)
                    try:
                        scenarios.append(self.mutation_generator.remove_fact(
                            base,
                            m.concept_id,
                            seed,
                            truth_effect=m.truth_effect,
                        ))
                    except ValueError:
                        pass

        # Exact-content dedup.
        by_hash = {}
        for scenario in scenarios:
            by_hash.setdefault(scenario.content_hash_value, scenario)
        scenarios = list(by_hash.values())

        splitter = SplitEngine(config.split)
        clusters = splitter.build_clusters(scenarios)
        assignments = splitter.assign(clusters)
        cluster_of = {
            str(s.scenario_id): cluster_id
            for cluster_id, members in clusters.items()
            for s in members
        }

        validation = self.validator.validate(scenarios, cluster_of, assignments)
        (build_dir / "validation.json").write_text(validation.model_dump_json(indent=2), encoding="utf-8")
        if not validation.eligible_for_export:
            raise RuntimeError(f"Build failed validation: {build_id}")

        export_paths = self.exporter.export(scenarios, cluster_of, assignments, build_dir / "exports")
        files = {}
        for split_name, path in export_paths.items():
            raw = path.read_bytes()
            files[split_name] = {
                "path": str(path.relative_to(build_dir)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

        manifest = {
            "build_id": build_id,
            "name": config.name,
            "spec_hash": stable_hash(config.model_dump(mode="json")),
            "scenario_count": len(scenarios),
            "cluster_count": len(clusters),
            "files": files,
        }
        (build_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
        return manifest
