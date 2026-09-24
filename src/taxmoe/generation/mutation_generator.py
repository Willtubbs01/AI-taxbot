from __future__ import annotations
from copy import deepcopy
from taxmoe.ingestion.hashing import stable_hash
from taxmoe.schemas.enums import InformationState, ScenarioOrigin
from taxmoe.schemas.fact import FactAssertion
from taxmoe.schemas.scenario import MutationLineage, TaxScenario
from taxmoe.scenarios.inference import ScenarioInferenceEngine
from .fingerprints import content_hash, semantic_fingerprint, structural_fingerprint

class MutationGenerator:
    def __init__(self, inference: ScenarioInferenceEngine, version: str = "0.1"):
        self.inference = inference
        self.version = version

    def _finish(self, parent: TaxScenario, child: TaxScenario, mutation_type: str, seed: int, targets: list[str], params: dict):
        child.origin = ScenarioOrigin.MUTATED
        child.scenario_id = f"SCENARIO-{stable_hash(parent.scenario_id, mutation_type, seed, targets, params)[:24]}"
        child.family_id = parent.family_id
        child.analysis = self.inference.analyze(
            child.input,
            task_ids=[x.task_id for x in parent.analysis.task_analyses],
        )
        child.content_hash = content_hash(child.input, child.analysis)
        child.semantic_fingerprint = semantic_fingerprint(child.input, child.analysis)
        child.structural_fingerprint = structural_fingerprint(child.input, child.analysis)
        child.mutation = MutationLineage(
            parent_scenario_id=parent.scenario_id,
            mutation_id=f"MUT-{stable_hash(child.scenario_id, mutation_type)[:20]}",
            mutation_type=mutation_type,
            mutation_version=self.version,
            seed=seed,
            target_fact_ids=targets,
            parameters=params,
        )
        return child

    def mark_unknown(self, parent: TaxScenario, concept_id: str, seed: int = 0) -> TaxScenario:
        child = deepcopy(parent)
        targets = []
        for f in child.input.facts:
            if f.concept_id == concept_id:
                f.state = InformationState.UNKNOWN
                f.value = None
                targets.append(f.fact_id)
        if not targets:
            raise ValueError(f"No fact with concept {concept_id}")
        return self._finish(parent, child, "mark_unknown", seed, targets, {"concept_id": concept_id})

    def create_conflict(self, parent: TaxScenario, concept_id: str, conflicting_value, seed: int = 0) -> TaxScenario:
        child = deepcopy(parent)
        targets = []
        for f in child.input.facts:
            if f.concept_id == concept_id:
                original = f.value
                f.state = InformationState.CONFLICTING
                f.assertions = [
                    FactAssertion(assertion_id=f"A-{f.fact_id}-1", value=original, source_type="tax_document"),
                    FactAssertion(assertion_id=f"A-{f.fact_id}-2", value=conflicting_value, source_type="user_statement"),
                ]
                f.value = None
                targets.append(f.fact_id)
        if not targets:
            raise ValueError(f"No fact with concept {concept_id}")
        return self._finish(parent, child, "create_conflict", seed, targets, {"concept_id": concept_id})

    def add_forced_assumption(self, parent: TaxScenario, concept_id: str, suggested_value, seed: int = 0):
        child = deepcopy(parent)
        child.input.context_items.append({
            "context_id": f"CTX-{stable_hash(parent.scenario_id, concept_id, seed)[:20]}",
            "role": "forced_assumption",
            "payload": {"concept_id": concept_id, "suggested_value": suggested_value},
            "truth": "unverified",
            "trusted": False,
        })
        return self._finish(parent, child, "forced_assumption", seed, [], {"concept_id": concept_id, "suggested_value": suggested_value})
