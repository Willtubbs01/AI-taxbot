from taxmoe.ingestion.hashing import stable_hash
from taxmoe.schemas.scenario import ScenarioInput, ScenarioAnalysis

def content_hash(world: ScenarioInput, analysis: ScenarioAnalysis) -> str:
    return stable_hash(world.model_dump(mode="json"), analysis.model_dump(mode="json"))

def semantic_fingerprint(world: ScenarioInput, analysis: ScenarioAnalysis) -> str:
    payload = {
        "tax_year": world.tax_year,
        "concept_states": sorted((f.concept_id, f.state.value) for f in world.facts),
        "events": sorted(e.event_type for e in world.events),
        "documents": sorted(d.form_id for d in world.documents),
        "statuses": sorted((t.task_id, t.status.value) for t in analysis.task_analyses),
        "rules": sorted({rid for t in analysis.task_analyses for rid in t.activated_rule_ids}),
    }
    return stable_hash(payload)

def structural_fingerprint(world: ScenarioInput, analysis: ScenarioAnalysis) -> str:
    payload = {
        "concept_states": sorted((f.concept_id, f.state.value) for f in world.facts),
        "events": sorted(e.event_type for e in world.events),
        "documents": sorted(d.form_id for d in world.documents),
        "tasks": sorted(t.task_id for t in analysis.task_analyses),
    }
    return stable_hash(payload)
