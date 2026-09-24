from __future__ import annotations

from copy import deepcopy

from taxmoe.generation.scenario_generator import SimpleInferenceEngine
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.schemas.common import stable_hash
from taxmoe.schemas.enums import InformationState, MutationTruthEffect, ScenarioOrigin
from taxmoe.schemas.scenario import MutationLineage, TaxScenario


class MutationGenerator:
    VERSION = "0.1"

    def __init__(self, rules: RuleRegistry):
        self.inference = SimpleInferenceEngine(rules)

    def remove_fact(
        self,
        parent: TaxScenario,
        concept_id: str,
        seed: int,
        truth_effect: MutationTruthEffect = MutationTruthEffect.MAY_CHANGE_ANALYSIS,
    ) -> TaxScenario:
        child = deepcopy(parent)
        candidates = [f for f in child.input.facts if f.concept_id == concept_id]
        if not candidates:
            raise ValueError(f"No fact for concept {concept_id}")
        target = candidates[0]
        target.state = InformationState.UNKNOWN
        target.value = None
        target.assertions = []

        child_id = f"SCENARIO-{stable_hash(parent.scenario_id, 'remove_fact', concept_id, seed, self.VERSION)[:20]}"
        child.scenario_id = child_id
        child.input.scenario_id = child_id
        child.origin = ScenarioOrigin.MUTATED
        child.mutation = MutationLineage(
            parent_scenario_id=parent.scenario_id,
            mutation_id=f"MUT-{stable_hash(child_id)[:20]}",
            mutation_type="remove_fact",
            mutation_version=self.VERSION,
            seed=seed,
            target_fact_ids=[str(target.fact_id)],
            parameters={"concept_id": concept_id, "truth_effect": truth_effect.value},
        )
        task_ids = [a.task_id for a in parent.analysis.task_analyses]
        child.analysis = self.inference.analyze(child.input, task_ids)
        child.content_hash_value = child.input.content_hash()
        child.semantic_fingerprint = stable_hash(parent.semantic_fingerprint, "remove_fact", concept_id)
        child.structural_fingerprint = stable_hash(parent.structural_fingerprint, "state", concept_id, "unknown")

        changed = child.analysis.model_dump(mode="json") != parent.analysis.model_dump(mode="json")
        if truth_effect == MutationTruthEffect.MUST_CHANGE_ANALYSIS and not changed:
            raise ValueError("Mutation required an analysis change but produced none")
        if truth_effect == MutationTruthEffect.MUST_PRESERVE_ANALYSIS and changed:
            raise ValueError("Mutation required analysis preservation but changed analysis")
        return child
