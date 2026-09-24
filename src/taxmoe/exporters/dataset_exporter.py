from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from taxmoe.schemas.common import stable_hash
from taxmoe.schemas.enums import DatasetSplit
from taxmoe.schemas.example import TrainingExample
from taxmoe.schemas.scenario import TaxScenario
from taxmoe.splitting.splitter import SplitAssignment


class DatasetExporter:
    VERSION = "0.1"

    SYSTEM = (
        "You are a specialized U.S. federal individual-income-tax reasoning component. "
        "Analyze only the provided task and facts. Do not invent missing taxpayer information."
    )

    def build_example(
        self,
        scenario: TaxScenario,
        cluster_id: str,
        split: DatasetSplit,
        task_id: str,
    ) -> TrainingExample:
        task_analysis = next(a for a in scenario.analysis.task_analyses if a.task_id == task_id)
        user_payload = {
            "task": task_id,
            "jurisdiction": scenario.input.jurisdiction,
            "tax_year": scenario.input.tax_year,
            "facts": [
                {
                    "concept_id": f.concept_id,
                    "state": f.state.value,
                    "value": f.value.model_dump(mode="json") if f.value else None,
                    "event_id": str(f.event_id) if f.event_id else None,
                }
                for f in scenario.input.facts
            ],
            "documents": [
                {"form_id": d.form_id, "form_version_id": d.form_version_id}
                for d in scenario.input.documents
            ],
            "context": [c.model_dump(mode="json") for c in scenario.input.context_items],
        }
        assistant_payload = {
            "status": task_analysis.status.value,
            "topics": task_analysis.active_concept_ids,
            "missing_information": task_analysis.missing_fact_ids,
            "required_rule_lookups": task_analysis.required_rule_lookups,
            "calculation_requests": task_analysis.required_calculations,
            "candidate_forms": task_analysis.candidate_forms,
            "required_forms": task_analysis.required_forms,
        }
        example_id = f"EXAMPLE-{stable_hash(scenario.scenario_id, task_id, cluster_id, self.VERSION)[:20]}"
        return TrainingExample(
            example_id=example_id,
            scenario_id=str(scenario.scenario_id),
            family_id=str(scenario.family_id),
            cluster_id=cluster_id,
            task_id=task_id,
            dataset_family=f"tax_{task_id}",
            split=split,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(assistant_payload, sort_keys=True, separators=(",", ":"))},
            ],
            metadata={"origin": scenario.origin.value, "tax_year": scenario.input.tax_year},
        )

    def export(
        self,
        scenarios: list[TaxScenario],
        cluster_of: dict[str, str],
        assignments: dict[str, SplitAssignment],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records: dict[str, list[TrainingExample]] = defaultdict(list)
        seen: set[str] = set()
        prompt_targets: dict[str, str] = {}

        for scenario in scenarios:
            sid = str(scenario.scenario_id)
            cluster_id = cluster_of[sid]
            split = assignments[cluster_id].split
            for analysis in scenario.analysis.task_analyses:
                example = self.build_example(scenario, cluster_id, split, analysis.task_id)
                prompt = "\n".join(m["content"] for m in example.messages[:-1])
                target = example.messages[-1]["content"]
                prompt_hash = stable_hash(prompt)
                if prompt_hash in prompt_targets and prompt_targets[prompt_hash] != target:
                    raise ValueError(f"EXPORT-TARGET-CONFLICT for prompt hash {prompt_hash}")
                prompt_targets[prompt_hash] = target
                full_hash = stable_hash(prompt, target)
                if full_hash in seen:
                    continue
                seen.add(full_hash)
                records[split.value].append(example)

        outputs: dict[str, Path] = {}
        for split_name, examples in records.items():
            path = output_dir / f"{split_name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for example in sorted(examples, key=lambda x: x.example_id):
                    f.write(json.dumps(example.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n")
            outputs[split_name] = path
        return outputs
