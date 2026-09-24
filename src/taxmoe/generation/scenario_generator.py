from __future__ import annotations
import random
from taxmoe.ingestion.hashing import stable_hash, stable_int
from taxmoe.schemas.enums import FactOrigin, InformationState, ScenarioOrigin
from taxmoe.schemas.fact import TaxFact
from taxmoe.schemas.scenario import (
    ScenarioDocument,
    TaxEvent,
    ScenarioInput,
    ScenarioGenerationLineage,
    TaxScenario,
)
from taxmoe.scenarios.template_registry import TemplateRegistry
from taxmoe.scenarios.inference import ScenarioInferenceEngine
from taxmoe.knowledge.form_registry import FormRegistry
from taxmoe.knowledge.values import TaxValueStore
from .value_generators import ValueGenerationContext, generate_value
from .fingerprints import content_hash, semantic_fingerprint, structural_fingerprint

class ScenarioGenerator:
    def __init__(
        self,
        templates: TemplateRegistry,
        forms: FormRegistry,
        values: TaxValueStore,
        inference: ScenarioInferenceEngine,
        generator_version: str = "0.1",
    ):
        self.templates = templates
        self.forms = forms
        self.values = values
        self.inference = inference
        self.generator_version = generator_version

    def _seed_for(self, root_seed: int, template_id: str, variable_id: str, attempt: int) -> int:
        return stable_int(self.generator_version, root_seed, template_id, variable_id, attempt)

    def generate(
        self,
        template_id: str,
        tax_year: int,
        seed: int,
        build_id: str = "BUILD-DEV",
        index: int = 0,
        revision: int | None = None,
    ) -> TaxScenario:
        template = self.templates.get(template_id, revision)
        if template is None:
            raise KeyError(f"Unknown template: {template_id}")
        if template.supported_tax_years and tax_year not in template.supported_tax_years:
            raise ValueError(f"Template {template_id} does not support tax year {tax_year}")

        for attempt in range(1, template.generation.max_attempts + 1):
            generated = {}
            for var in template.variables:
                rng = random.Random(self._seed_for(seed, template_id, var.variable_id, attempt))
                ctx = ValueGenerationContext(tax_year, generated, self.values, rng)
                generated[var.variable_id] = generate_value(var.generator, var.value_type, ctx)

            if not self._constraints_pass(template.constraints, generated):
                continue

            scenario_identity = stable_hash(
                template.template_id, template.revision, self.generator_version, seed, index
            )[:24]
            scenario_id = f"SCENARIO-{scenario_identity}"
            family_id = f"FAMILY-{stable_hash(template.template_id, seed, index)[:20]}"

            event_map = {}
            events = []
            for ev in template.events:
                event_id = f"EVENT-{stable_hash(scenario_id, ev.event_id)[:20]}"
                event_map[ev.event_id] = event_id
                events.append(TaxEvent(event_id=event_id, event_type=ev.event_type))

            facts = []
            var_to_fact = {}
            for var in template.variables:
                fact_id = f"FACT-{stable_hash(scenario_id, var.variable_id)[:20]}"
                var_to_fact[var.variable_id] = fact_id
                event_id = None
                for ev in template.events:
                    if var.variable_id in ev.variable_refs:
                        event_id = event_map[ev.event_id]
                        break
                facts.append(TaxFact(
                    fact_id=fact_id,
                    concept_id=var.concept_id,
                    state=InformationState.PRESENT,
                    value=generated[var.variable_id],
                    origin=FactOrigin.SYNTHETIC,
                    event_id=event_id,
                    generation_variable_id=var.variable_id,
                ))

            event_by_id = {e.event_id: e for e in events}
            for ev in template.events:
                inst = event_by_id[event_map[ev.event_id]]
                inst.fact_ids = [var_to_fact[x] for x in ev.variable_refs]
                inst.concept_ids = [
                    next(v.concept_id for v in template.variables if v.variable_id == x)
                    for x in ev.variable_refs
                ]

            documents = []
            for doc in template.documents:
                form_version = self.forms.get_version_for_year(doc.form_id, tax_year)
                if form_version is None:
                    raise ValueError(f"No form version for {doc.form_id}/{tax_year}")
                documents.append(ScenarioDocument(
                    document_instance_id=f"DOC-{stable_hash(scenario_id, doc.document_id)[:20]}",
                    form_id=doc.form_id,
                    form_version_id=form_version.form_version_id,
                    fact_ids=[var_to_fact[x] for x in doc.variable_refs if x in var_to_fact],
                ))

            world = ScenarioInput(
                jurisdiction=template.jurisdiction,
                tax_year=tax_year,
                facts=facts,
                events=events,
                documents=documents,
            )
            analysis = self.inference.analyze(world, task_ids=template.task_ids or ["analyze_tax_scenario"])
            if template.generation.require_complete_base_scenario and any(
                t.missing_concept_ids for t in analysis.task_analyses
            ):
                continue

            lineage = ScenarioGenerationLineage(
                template_id=template.template_id,
                template_revision=template.revision,
                seed=seed,
                generator_version=self.generator_version,
                generation_index=index,
                accepted_attempt=attempt,
                taxonomy_version="0.1",
                form_registry_version=self.forms.version,
                rule_store_version=self.inference.rules.version,
                value_store_version=self.values.version,
                calculation_registry_version="0.1",
                build_id=build_id,
            )
            return TaxScenario(
                scenario_id=scenario_id,
                family_id=family_id,
                origin=ScenarioOrigin.GENERATED,
                input=world,
                analysis=analysis,
                semantic_fingerprint=semantic_fingerprint(world, analysis),
                structural_fingerprint=structural_fingerprint(world, analysis),
                content_hash=content_hash(world, analysis),
                generation=lineage,
            )

        raise RuntimeError(f"Generation failed after {template.generation.max_attempts} attempts")

    @staticmethod
    def _constraints_pass(constraints, values: dict) -> bool:
        for c in constraints:
            left = values.get(c.left)
            right = values.get(c.right, c.right) if isinstance(c.right, str) else c.right
            kind = c.type.lower()
            if kind in {"before", "lt"} and not (left < right):
                return False
            if kind in {"after", "gt"} and not (left > right):
                return False
            if kind == "lte" and not (left <= right):
                return False
            if kind == "gte" and not (left >= right):
                return False
            if kind == "eq" and not (left == right):
                return False
            if kind == "ne" and not (left != right):
                return False
        return True
