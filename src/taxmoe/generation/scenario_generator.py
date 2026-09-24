from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from random import Random
from typing import Any

from taxmoe.knowledge.form_registry import FormRegistry
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.taxonomy_registry import TaxonomyRegistry
from taxmoe.knowledge.values import TaxValueStore
from taxmoe.schemas.common import stable_hash
from taxmoe.schemas.enums import AnswerabilityStatus, FactOrigin, InformationState, ScenarioOrigin, TruthPolarity
from taxmoe.schemas.fact import Money, MoneyValue, ScalarValue, TaxFact, TaxFactValue
from taxmoe.schemas.scenario import (
    ScenarioAnalysis,
    ScenarioDocument,
    ScenarioGenerationLineage,
    ScenarioInput,
    TaskAnalysis,
    TaxEvent,
    TaxScenario,
)
from taxmoe.scenarios.template_registry import TemplateRegistry
from taxmoe.scenarios.templates import ScenarioTemplate, TemplateVariable


class SimpleInferenceEngine:
    """Small deterministic inference engine for the Stage-3 foundation.

    It evaluates a limited subset of rule predicates/effects. Expand this before
    relying on it for broader tax coverage.
    """

    def __init__(self, rules: RuleRegistry):
        self.rules = rules

    @staticmethod
    def _fact_map(scenario_input: ScenarioInput) -> dict[str, list[TaxFact]]:
        out: dict[str, list[TaxFact]] = {}
        for fact in scenario_input.facts:
            out.setdefault(fact.concept_id, []).append(fact)
        return out

    def _predicate(self, predicate, facts: dict[str, list[TaxFact]]) -> bool:
        op = predicate.op
        if op == "and":
            return all(self._predicate(c, facts) for c in predicate.children)
        if op == "or":
            return any(self._predicate(c, facts) for c in predicate.children)
        if op == "not":
            return not self._predicate(predicate.children[0], facts)
        candidates = facts.get(predicate.concept_id or "", [])
        present = [f for f in candidates if f.state == InformationState.PRESENT and f.value is not None]
        if op == "exists":
            return bool(present)
        if not present:
            return False
        actual = _primitive(present[0].value)
        expected = predicate.value
        return {
            "eq": actual == expected,
            "ne": actual != expected,
            "gt": actual > expected,
            "gte": actual >= expected,
            "lt": actual < expected,
            "lte": actual <= expected,
        }[op]

    def analyze(self, scenario_input: ScenarioInput, task_ids: list[str]) -> ScenarioAnalysis:
        facts = self._fact_map(scenario_input)
        analyses: list[TaskAnalysis] = []
        for task_id in task_ids:
            missing: set[str] = set()
            conflicting: set[str] = set()
            concepts: set[str] = set()
            calcs: set[str] = set()
            candidates: set[str] = set()
            required: set[str] = set()
            lookups: set[str] = set()
            trace: list[str] = []
            outside = False

            for concept, entries in facts.items():
                if any(f.state == InformationState.CONFLICTING for f in entries):
                    conflicting.update(str(f.fact_id) for f in entries if f.state == InformationState.CONFLICTING)

            for rule in self.rules.applicable(scenario_input.tax_year):
                if not self._predicate(rule.predicate, facts):
                    continue
                trace.append(str(rule.rule_id))
                for effect in rule.effects:
                    if effect.task_id and effect.task_id != task_id:
                        continue
                    if effect.effect_type == "activate_concept":
                        concepts.add(effect.target)
                    elif effect.effect_type == "require_fact":
                        entries = facts.get(effect.target, [])
                        usable = [f for f in entries if f.state == InformationState.PRESENT and f.value is not None]
                        if not usable:
                            missing.add(effect.target)
                    elif effect.effect_type == "require_calculation":
                        calcs.add(effect.target)
                    elif effect.effect_type == "candidate_form":
                        candidates.add(effect.target)
                    elif effect.effect_type == "require_form":
                        required.add(effect.target)
                        candidates.add(effect.target)
                    elif effect.effect_type == "needs_rule_lookup":
                        lookups.add(effect.target)
                    elif effect.effect_type == "outside_scope":
                        outside = True

            if outside:
                status = AnswerabilityStatus.OUTSIDE_SCOPE
            elif conflicting:
                status = AnswerabilityStatus.AMBIGUOUS
            elif missing:
                status = AnswerabilityStatus.NEEDS_INFORMATION
            elif lookups:
                status = AnswerabilityStatus.NEEDS_RULE_LOOKUP
            else:
                status = AnswerabilityStatus.ANSWERABLE

            analyses.append(TaskAnalysis(
                task_id=task_id,
                status=status,
                active_concept_ids=sorted(concepts),
                missing_fact_ids=sorted(missing),
                conflicting_fact_ids=sorted(conflicting),
                required_rule_lookups=sorted(lookups),
                required_calculations=sorted(calcs),
                candidate_forms=sorted(candidates),
                required_forms=sorted(required),
                rule_trace=trace,
            ))

        order = [
            AnswerabilityStatus.OUTSIDE_SCOPE,
            AnswerabilityStatus.AMBIGUOUS,
            AnswerabilityStatus.NEEDS_INFORMATION,
            AnswerabilityStatus.NEEDS_RULE_LOOKUP,
            AnswerabilityStatus.ANSWERABLE,
        ]
        overall = next(s for s in order if any(a.status == s for a in analyses)) if analyses else AnswerabilityStatus.ANSWERABLE
        return ScenarioAnalysis(overall_status=overall, task_analyses=analyses)


def _primitive(value: TaxFactValue) -> Any:
    if isinstance(value, MoneyValue):
        return value.value.amount_cents
    return value.value


class ScenarioGenerator:
    VERSION = "0.1"

    def __init__(
        self,
        templates: TemplateRegistry,
        taxonomy: TaxonomyRegistry,
        forms: FormRegistry,
        rules: RuleRegistry,
        values: TaxValueStore,
    ):
        self.templates = templates
        self.taxonomy = taxonomy
        self.forms = forms
        self.rules = rules
        self.values = values
        self.inference = SimpleInferenceEngine(rules)

    def generate(
        self,
        template_id: str,
        tax_year: int,
        seed: int,
        generation_index: int = 0,
        revision: int | None = None,
    ) -> TaxScenario:
        template = self.templates.get(template_id, revision)
        if template is None:
            raise KeyError(f"Unknown template {template_id}")
        if template.supported_tax_years and tax_year not in template.supported_tax_years:
            raise ValueError(f"Template {template_id} does not support {tax_year}")

        values: dict[str, TaxFactValue] = {}
        facts: list[TaxFact] = []
        scenario_token = stable_hash(template.template_id, template.revision, self.VERSION, seed, generation_index)[:20]
        scenario_id = f"SCENARIO-{scenario_token}"
        family_id = f"FAMILY-{stable_hash(template.template_id, generation_index)[:20]}"

        for variable in template.variables:
            child_seed = int(stable_hash(seed, template.template_id, variable.variable_id)[:16], 16)
            rng = Random(child_seed)
            value = self._generate_value(variable, tax_year, rng, values, template)
            values[variable.variable_id] = value
            facts.append(TaxFact(
                fact_id=f"FACT-{stable_hash(scenario_id, variable.variable_id)[:20]}",
                concept_id=variable.concept_id,
                state=InformationState.PRESENT,
                value=value,
                origin=FactOrigin.SYNTHETIC,
                truth=TruthPolarity.SYNTHETIC,
                field_path=variable.field_path,
                generation_variable_id=variable.variable_id,
            ))

        fact_by_var = {f.generation_variable_id: f for f in facts}
        events: list[TaxEvent] = []
        for event_spec in template.events:
            event_id = f"EVENT-{stable_hash(scenario_id, event_spec.event_id)[:20]}"
            ids: list[str] = []
            for variable_ref in event_spec.variable_refs:
                fact = fact_by_var[variable_ref]
                fact.event_id = event_id  # validated assignment
                ids.append(str(fact.fact_id))
            events.append(TaxEvent(
                event_id=event_id,
                event_type=event_spec.event_type,
                fact_ids=ids,
                concept_ids=event_spec.concept_ids,
            ))

        documents: list[ScenarioDocument] = []
        for doc in template.documents:
            version = self.forms.resolve_version(doc.form_id, tax_year)
            bound_facts = [str(fact_by_var[v].fact_id) for v in doc.field_bindings.values()]
            documents.append(ScenarioDocument(
                document_instance_id=f"DOC-{stable_hash(scenario_id, doc.document_id)[:20]}",
                form_id=doc.form_id,
                form_version_id=version.form_version_id,
                fact_ids=bound_facts,
            ))

        scenario_input = ScenarioInput(
            scenario_id=scenario_id,
            jurisdiction=template.jurisdiction,
            tax_year=tax_year,
            facts=facts,
            events=events,
            documents=documents,
        )
        analysis = self.inference.analyze(scenario_input, template.task_ids)
        if template.require_complete_base_scenario and analysis.overall_status in {
            AnswerabilityStatus.NEEDS_INFORMATION,
            AnswerabilityStatus.AMBIGUOUS,
        }:
            raise ValueError(f"Base template generated incomplete scenario: {analysis.overall_status}")

        content_hash = scenario_input.content_hash()
        semantic = stable_hash(
            template.template_id,
            sorted(e.event_type for e in events),
            sorted(f.concept_id for f in facts),
            sorted(d.form_id for d in documents),
            sorted(a.task_id for a in analysis.task_analyses),
        )
        structural = stable_hash(
            sorted(e.event_type for e in events),
            sorted(f.concept_id for f in facts),
            sorted(d.form_id for d in documents),
            sorted((a.task_id, a.status.value) for a in analysis.task_analyses),
        )
        return TaxScenario(
            scenario_id=scenario_id,
            family_id=family_id,
            origin=ScenarioOrigin.GENERATED,
            input=scenario_input,
            analysis=analysis,
            semantic_fingerprint=semantic,
            structural_fingerprint=structural,
            content_hash_value=content_hash,
            generation=ScenarioGenerationLineage(
                template_id=template.template_id,
                template_revision=template.revision,
                seed=seed,
                generator_version=self.VERSION,
                generation_index=generation_index,
                rule_store_version=self.rules.version,
                taxonomy_version=self.taxonomy.version,
                form_registry_version=self.forms.version,
                value_store_version=self.values.version,
            ),
        )

    def _generate_value(
        self,
        variable: TemplateVariable,
        tax_year: int,
        rng: Random,
        generated: dict[str, TaxFactValue],
        template: ScenarioTemplate,
    ) -> TaxFactValue:
        spec = variable.generator
        p = spec.parameters
        if spec.type == "constant":
            return ScalarValue(value=p["value"])
        if spec.type == "choice":
            return ScalarValue(value=rng.choice(p["values"]))
        if spec.type == "integer_range":
            return ScalarValue(value=rng.randint(int(p["min"]), int(p["max"])))
        if spec.type == "money_range":
            low = int(p["min_cents"])
            high = int(p["max_cents"])
            step = int(p.get("step_cents", 100))
            count = (high - low) // step
            return MoneyValue(value=Money(amount_cents=low + step * rng.randint(0, count)))
        if spec.type == "tax_year_date":
            start = date(tax_year, 1, 1)
            days = (date(tax_year, 12, 31) - start).days
            return ScalarValue(value=(start + timedelta(days=rng.randint(0, days))).isoformat())
        if spec.type == "relative_date":
            ref = generated[p["reference"]]
            if not isinstance(ref, ScalarValue):
                raise TypeError("relative_date reference must be scalar ISO date")
            base = date.fromisoformat(str(ref.value))
            delta = rng.randint(int(p.get("min_days_after", 1)), int(p.get("max_days_after", 365)))
            return ScalarValue(value=(base + timedelta(days=delta)).isoformat())
        if spec.type == "value_ref":
            tv = self.values.resolve(
                p["value_id"], tax_year=tax_year, jurisdiction=template.jurisdiction,
                dimensions=p.get("dimensions", {}),
            )
            return ScalarValue(value=tv.value)
        raise ValueError(f"Unsupported generator type: {spec.type}")
