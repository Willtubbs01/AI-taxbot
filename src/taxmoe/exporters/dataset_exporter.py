from __future__ import annotations
from collections import defaultdict
from taxmoe.ingestion.hashing import stable_hash
from taxmoe.schemas.enums import DatasetSplit
from taxmoe.schemas.example import ExampleLineage, TrainingExample
from taxmoe.schemas.manifest import SplitManifest
from taxmoe.schemas.scenario import TaxScenario
from .adapters.qwen import to_qwen_messages
from .renderers.structured import StructuredRenderer
from .renderers.concise import ConciseRenderer
from .jsonl import write_jsonl

class DatasetExporter:
    def __init__(self, exporter_version: str = "0.1"):
        self.exporter_version = exporter_version
        self.renderers = {
            "structured": StructuredRenderer(),
            "concise": ConciseRenderer(),
        }

    def build_examples(self, scenarios: list[TaxScenario], split_manifest: SplitManifest, renderers=("structured",)):
        cluster_split = {a.cluster_id: a.split for a in split_manifest.assignments}
        examples = []
        visible_input_to_target = {}

        for s in scenarios:
            cluster_id = split_manifest.scenario_to_cluster[s.scenario_id]
            split = DatasetSplit(cluster_split[cluster_id])
            for task in s.analysis.task_analyses:
                for renderer_id in renderers:
                    renderer = self.renderers[renderer_id]
                    user_text, target_text = renderer.render(s, task.task_id)

                    input_hash = stable_hash(user_text)
                    target_hash = stable_hash(target_text)
                    previous = visible_input_to_target.get(input_hash)
                    if previous is not None and previous != target_hash:
                        raise RuntimeError("EXPORT-TARGET-CONFLICT: same visible input has incompatible targets")
                    visible_input_to_target[input_hash] = target_hash

                    example_id = f"EXAMPLE-{stable_hash(s.scenario_id, task.task_id, renderer_id, renderer.version)[:24]}"
                    lineage = ExampleLineage(
                        scenario_id=s.scenario_id,
                        scenario_content_hash=s.content_hash,
                        task_id=task.task_id,
                        renderer_id=renderer_id,
                        renderer_version=renderer.version,
                        exporter_version=self.exporter_version,
                        split_version=split_manifest.split_version,
                    )
                    examples.append(TrainingExample(
                        example_id=example_id,
                        scenario_id=s.scenario_id,
                        family_id=s.family_id,
                        cluster_id=cluster_id,
                        task_id=task.task_id,
                        dataset_family=f"tax_{task.task_id}",
                        split=split,
                        messages=to_qwen_messages(user_text, target_text),
                        metadata={
                            "tax_year": s.input.tax_year,
                            "origin": s.origin.value,
                            "renderer_id": renderer_id,
                        },
                        lineage=lineage,
                    ))
        # exact record dedup
        unique = {}
        for e in examples:
            h = stable_hash(e.messages, e.task_id, e.split.value)
            unique.setdefault(h, e)
        return sorted(unique.values(), key=lambda e: e.example_id)

    def export(self, examples: list[TrainingExample], output_root):
        groups = defaultdict(list)
        for e in examples:
            groups[(e.split.value, e.dataset_family)].append(e)

        results = []
        for (split, family), rows in sorted(groups.items()):
            data = [x.model_dump(mode="json") for x in rows]
            path = output_root / split / f"{family}.jsonl"
            results.append(write_jsonl(path, data))
        return results
