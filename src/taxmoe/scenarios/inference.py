from __future__ import annotations
from typing import Any
from taxmoe.schemas.enums import AnswerabilityStatus, InformationState
from taxmoe.schemas.rule import TaxRule, RuleCondition
from taxmoe.schemas.scenario import ScenarioInput, ScenarioAnalysis, TaskAnalysis
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.values import TaxValueStore

def _fact_map(world: ScenarioInput):
    out = {}
    for f in world.facts:
        out.setdefault(f.concept_id, []).append(f)
    return out

def _condition_matches(cond: RuleCondition, fmap: dict[str, list], tax_year: int, values: TaxValueStore) -> bool:
    facts = fmap.get(cond.concept_id, [])
    present = [f for f in facts if f.state == InformationState.PRESENT]
    op = cond.operator.upper()
    if op == "EXISTS":
        return bool(present)
    if op == "MISSING":
        return not present
    if not present:
        return False
    value = present[0].value
    rhs = cond.value
    if cond.value_ref:
        rhs = values.resolve(cond.value_ref, tax_year)
        if rhs is None:
            return False
    if op == "EQ":
        return value == rhs
    if op == "NE":
        return value != rhs
    if op == "GT":
        return value > rhs
    if op == "GTE":
        return value >= rhs
    if op == "LT":
        return value < rhs
    if op == "LTE":
        return value <= rhs
    raise ValueError(f"Unsupported operator: {cond.operator}")

class ScenarioInferenceEngine:
    def __init__(self, rules: RuleRegistry, values: TaxValueStore):
        self.rules = rules
        self.values = values

    def analyze(self, world: ScenarioInput, task_ids: list[str] | None = None) -> ScenarioAnalysis:
        fmap = _fact_map(world)
        task_ids = task_ids or ["analyze_tax_scenario"]
        active = []
        for rule in self.rules.active_for_year(world.tax_year):
            if all(_condition_matches(c, fmap, world.tax_year, self.values) for c in rule.conditions):
                active.append(rule)

        outputs = []
        for task_id in task_ids:
            topics, missing, conflicts = set(), set(), set()
            source_docs, candidates, required_forms = set(), set(), set()
            lookups, calcs, rule_ids = set(), set(), set()

            for facts in fmap.values():
                for f in facts:
                    if f.state == InformationState.CONFLICTING:
                        conflicts.add(f.concept_id)

            for rule in active:
                rule_ids.add(rule.rule_id)
                for effect in rule.effects:
                    et = effect.effect_type.upper()
                    if et == "ADD_TOPIC":
                        topics.add(effect.target)
                    elif et == "REQUIRE_INFORMATION":
                        facts = fmap.get(effect.target, [])
                        if not any(f.state == InformationState.PRESENT for f in facts):
                            missing.add(effect.target)
                    elif et == "SOURCE_DOCUMENT":
                        source_docs.add(effect.target)
                    elif et == "CANDIDATE_FORM":
                        candidates.add(effect.target)
                    elif et == "REQUIRE_FORM":
                        required_forms.add(effect.target)
                    elif et == "REQUIRE_CALCULATION":
                        calcs.add(effect.target)
                    elif et == "REQUIRE_RULE_LOOKUP":
                        lookups.add(effect.target)

            status = AnswerabilityStatus.ANSWERABLE
            if conflicts:
                status = AnswerabilityStatus.AMBIGUOUS
            elif missing:
                status = AnswerabilityStatus.NEEDS_INFORMATION
            elif lookups:
                status = AnswerabilityStatus.NEEDS_RULE_LOOKUP

            outputs.append(TaskAnalysis(
                task_id=task_id,
                status=status,
                topic_ids=sorted(topics),
                missing_concept_ids=sorted(missing),
                conflicting_concept_ids=sorted(conflicts),
                source_document_ids=sorted(source_docs),
                candidate_forms=sorted(candidates),
                required_forms=sorted(required_forms),
                required_rule_lookups=sorted(lookups),
                required_calculations=sorted(calcs),
                activated_rule_ids=sorted(rule_ids),
            ))

        overall = AnswerabilityStatus.ANSWERABLE
        statuses = {x.status for x in outputs}
        for candidate in [
            AnswerabilityStatus.AMBIGUOUS,
            AnswerabilityStatus.NEEDS_INFORMATION,
            AnswerabilityStatus.NEEDS_RULE_LOOKUP,
            AnswerabilityStatus.OUTSIDE_SCOPE,
        ]:
            if candidate in statuses:
                overall = candidate
                break
        return ScenarioAnalysis(overall_status=overall, task_analyses=outputs)
