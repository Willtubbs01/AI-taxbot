from __future__ import annotations
import json
from pathlib import Path
import yaml
from taxmoe.ingestion.hashing import stable_hash
from taxmoe.ingestion.registry import SourceRegistry
from taxmoe.knowledge.taxonomy_registry import TaxonomyRegistry
from taxmoe.knowledge.form_registry import FormRegistry
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.values import TaxValueStore
from taxmoe.scenarios.template_registry import TemplateRegistry
from taxmoe.scenarios.inference import ScenarioInferenceEngine
from taxmoe.generation.scenario_generator import ScenarioGenerator
from taxmoe.generation.mutation_generator import MutationGenerator
from taxmoe.splitting.splitter import SplitEngine
from taxmoe.validation.build import BuildValidator
from taxmoe.exporters.dataset_exporter import DatasetExporter
from taxmoe.reporting.build_summary import write_build_summary
from taxmoe.schemas.scenario import TaxScenario
from taxmoe.schemas.manifest import SplitManifest
from .config import DatasetBuildConfig
from .models import DatasetBuildRecord, BuildStage, StageStatus

class DatasetBuildRunner:
    def __init__(self, project_root: str | Path = "."):
        self.root = Path(project_root).resolve()

    def _resolve(self, p: str) -> Path:
        return (self.root / p).resolve()

    def run(self, config_path: str | Path) -> DatasetBuildRecord:
        config = DatasetBuildConfig.from_yaml(self._resolve(str(config_path)))
        spec_hash = stable_hash(config.model_dump(mode="json"))
        build_id = f"BUILD-{spec_hash[:20]}"
        out = self.root / "artifacts" / "builds" / build_id
        out.mkdir(parents=True, exist_ok=True)

        record = DatasetBuildRecord(
            build_id=build_id,
            build_name=config.name,
            spec_hash=spec_hash,
            stages={stage.value: StageStatus.PENDING for stage in BuildStage},
        )
        self._write_record(out, record)

        try:
            record.current_stage = BuildStage.PREPARE.value
            record.stages[BuildStage.PREPARE.value] = StageStatus.RUNNING
            self._write_record(out, record)

            sources = SourceRegistry.from_yaml(self._resolve(config.source_manifest))
            taxonomy = TaxonomyRegistry.from_yaml(self._resolve(config.taxonomy_domains), self._resolve(config.taxonomy_tasks))
            forms = FormRegistry.from_yaml(self._resolve(config.forms), self._resolve(config.form_versions))
            rules = RuleRegistry.from_yaml_files([self._resolve(x) for x in config.rule_files], version="0.1")
            values = TaxValueStore.from_yaml(self._resolve(config.value_store))
            templates = TemplateRegistry.from_paths([self._resolve(x) for x in config.template_files], version="0.1")
            inference = ScenarioInferenceEngine(rules, values)
            generator = ScenarioGenerator(templates, forms, values, inference)

            record.stages[BuildStage.PREPARE.value] = StageStatus.COMPLETE

            record.current_stage = BuildStage.GENERATE.value
            record.stages[BuildStage.GENERATE.value] = StageStatus.RUNNING
            self._write_record(out, record)

            scenarios = []
            for t in config.templates:
                for i in range(t.count):
                    seed = int(stable_hash(config.seed, t.id, i)[:16], 16)
                    scenarios.append(generator.generate(t.id, config.tax_year, seed, build_id=build_id, index=i))
            self._write_scenarios(out / "scenarios" / "base.jsonl", scenarios)
            record.stages[BuildStage.GENERATE.value] = StageStatus.COMPLETE

            record.current_stage = BuildStage.MUTATE.value
            record.stages[BuildStage.MUTATE.value] = StageStatus.RUNNING
            self._write_record(out, record)

            mut_cfg = yaml.safe_load(self._resolve(config.mutation_config).read_text(encoding="utf-8")) or {}
            mutator = MutationGenerator(inference)
            mutated = []
            limit = int(mut_cfg.get("max_children_per_base", 1))
            for idx, s in enumerate(scenarios):
                if limit <= 0:
                    continue
                concepts = [f.concept_id for f in s.input.facts]
                if "investment.basis" in concepts:
                    mutated.append(mutator.mark_unknown(s, "investment.basis", seed=idx))
            self._write_scenarios(out / "scenarios" / "mutated.jsonl", mutated)
            all_scenarios = scenarios + mutated
            record.stages[BuildStage.MUTATE.value] = StageStatus.COMPLETE

            record.current_stage = BuildStage.SPLIT.value
            record.stages[BuildStage.SPLIT.value] = StageStatus.RUNNING
            self._write_record(out, record)

            split_cfg = yaml.safe_load(self._resolve(config.split_config).read_text(encoding="utf-8")) or {}
            split_engine = SplitEngine(
                seed=int(split_cfg.get("seed", config.seed)),
                split_version=str(split_cfg.get("split_version", "v1")),
                clustering_version=str(split_cfg.get("clustering_version", "v1")),
            )
            split_manifest = split_engine.build(all_scenarios)
            split_path = out / "splits" / "split_manifest.json"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(split_manifest.model_dump_json(indent=2), encoding="utf-8")
            record.stages[BuildStage.SPLIT.value] = StageStatus.COMPLETE

            record.current_stage = BuildStage.VALIDATE.value
            record.stages[BuildStage.VALIDATE.value] = StageStatus.RUNNING
            self._write_record(out, record)
            validator = BuildValidator(inference)
            sres = validator.validate_scenarios(all_scenarios)
            lres = validator.validate_split(all_scenarios, split_manifest)
            passed = sres.passed and lres.passed
            validation = {
                "passed": passed,
                "scenario_issues": [x.model_dump(mode="json") for x in sres.issues],
                "split_issues": [x.model_dump(mode="json") for x in lres.issues],
            }
            vp = out / "validation" / "validation_report.json"
            vp.parent.mkdir(parents=True, exist_ok=True)
            vp.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
            if not passed:
                raise RuntimeError("Validation failed")
            record.passed_validation = True
            record.stages[BuildStage.VALIDATE.value] = StageStatus.COMPLETE

            record.current_stage = BuildStage.EXPORT.value
            record.stages[BuildStage.EXPORT.value] = StageStatus.RUNNING
            self._write_record(out, record)
            export_cfg = yaml.safe_load(self._resolve(config.export_config).read_text(encoding="utf-8")) or {}
            exporter = DatasetExporter()
            renderers = tuple(export_cfg.get("renderers", ["structured", "concise"]))
            examples = exporter.build_examples(all_scenarios, split_manifest, renderers=renderers)
            export_results = exporter.export(examples, out / "exports")
            (out / "exports" / "manifest.json").write_text(json.dumps(export_results, indent=2), encoding="utf-8")
            record.export_id = f"EXPORT-{stable_hash(build_id, export_results)[:20]}"
            record.stages[BuildStage.EXPORT.value] = StageStatus.COMPLETE

            record.current_stage = BuildStage.REPORT.value
            record.stages[BuildStage.REPORT.value] = StageStatus.RUNNING
            self._write_record(out, record)
            summary = {
                "build_id": build_id,
                "name": config.name,
                "spec_hash": spec_hash,
                "base_scenarios": len(scenarios),
                "mutated_scenarios": len(mutated),
                "total_scenarios": len(all_scenarios),
                "exported_examples": len(examples),
                "validation_passed": passed,
                "export_id": record.export_id,
            }
            write_build_summary(out / "reports" / "build_summary.json", summary)
            record.stages[BuildStage.REPORT.value] = StageStatus.COMPLETE
            record.current_stage = None
            self._write_record(out, record)
            return record
        except Exception:
            if record.current_stage:
                record.stages[record.current_stage] = StageStatus.FAILED
            self._write_record(out, record)
            raise

    @staticmethod
    def _write_scenarios(path: Path, scenarios: list[TaxScenario]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in sorted(scenarios, key=lambda x: x.scenario_id):
                f.write(s.model_dump_json())
                f.write("\n")

    @staticmethod
    def _write_record(out: Path, record: DatasetBuildRecord):
        (out / "build.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
