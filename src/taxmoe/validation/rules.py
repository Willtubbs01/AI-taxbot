from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.taxonomy_registry import TaxonomyRegistry
from taxmoe.knowledge.form_registry import FormRegistry


def validate_rules(rules: RuleRegistry, taxonomy: TaxonomyRegistry, forms: FormRegistry) -> list[str]:
    issues = []
    for rule in rules._rules.values():  # registry-internal audit
        if rule.predicate.concept_id and rule.predicate.concept_id not in taxonomy:
            issues.append(f"REF-MISSING-CONCEPT:{rule.rule_id}:{rule.predicate.concept_id}")
        for effect in rule.effects:
            if effect.effect_type in {"require_form", "candidate_form"} and forms.get(effect.target) is None:
                issues.append(f"REF-MISSING-FORM:{rule.rule_id}:{effect.target}")
    return issues
