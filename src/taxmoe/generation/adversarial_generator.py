from copy import deepcopy
from taxmoe.schemas.common import stable_hash
from taxmoe.schemas.enums import ScenarioOrigin
from taxmoe.schemas.scenario import MutationLineage, ScenarioContextItem, TaxScenario


def add_untrusted_context(parent: TaxScenario, role: str, payload: dict[str, object], seed: int) -> TaxScenario:
    child = deepcopy(parent)
    child_id = f"SCENARIO-{stable_hash(parent.scenario_id, role, payload, seed)[:20]}"
    child.scenario_id = child_id
    child.input.scenario_id = child_id
    child.origin = ScenarioOrigin.MUTATED
    child.input.context_items.append(ScenarioContextItem(
        context_id=f"CTX-{stable_hash(child_id, role)[:20]}",
        role=role,
        content_type="adversarial",
        payload=payload,
        trusted=False,
    ))
    child.mutation = MutationLineage(
        parent_scenario_id=parent.scenario_id,
        mutation_id=f"MUT-{stable_hash(child_id)[:20]}",
        mutation_type="add_untrusted_context",
        mutation_version="0.1",
        seed=seed,
        parameters={"role": role},
    )
    child.content_hash_value = child.input.content_hash()
    return child
