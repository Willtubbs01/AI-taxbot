from __future__ import annotations

from collections import defaultdict

from pydantic import Field

from taxmoe.generation.scenario_generator import SimpleInferenceEngine
from taxmoe.knowledge.form_registry import FormRegistry
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.taxonomy_registry import TaxonomyRegistry
from taxmoe.schemas.common import TaxMoEModel
from taxmoe.schemas.enums import Severity
from taxmoe.schemas.scenario import TaxScenario
from taxmoe.splitting.splitter import SplitAssignment


class ValidationIssue(TaxMoEModel):
    code: str
    severity: Severity
    message: str
    artifact_id: str | None = None
    details: dict[str, object] = Field(default_factory=dict)


class BuildValidationResult(TaxMoEModel):
    passed: bool
    eligible_for_export: bool
    issues: list[ValidationIssue]


class BuildValidator:
    def __init__(self, taxonomy: TaxonomyRegistry, forms: FormRegistry, rules: RuleRegistry):
        self.taxonomy = taxonomy
        self.forms = forms
        self.rules = rules
        self.inference = SimpleInferenceEngine(rules)

    def validate(
        self,
        scenarios: list[TaxScenario],
        cluster_of: dict[str, str],
        assignments: dict[str, SplitAssignment],
    ) -> BuildValidationResult:
        issues: list[ValidationIssue] = []

        # Reference and recomputation checks.
        for scenario in scenarios:
            sid = str(scenario.scenario_id)
            try:
                for fact in scenario.input.facts:
                    self.taxonomy.require(fact.concept_id)
                for doc in scenario.input.documents:
                    version = self.forms.resolve_version(doc.form_id, scenario.input.tax_year)
                    if doc.form_version_id != version.form_version_id:
                        raise ValueError(f"Form version mismatch for {doc.document_instance_id}")
                fresh = self.inference.analyze(
                    scenario.input,
                    [a.task_id for a in scenario.analysis.task_analyses],
                )
                if fresh.model_dump(mode="json") != scenario.analysis.model_dump(mode="json"):
                    issues.append(ValidationIssue(
                        code="SCENARIO-ANALYSIS-MISMATCH",
                        severity=Severity.ERROR,
                        message="Stored scenario analysis differs from fresh deterministic inference",
                        artifact_id=sid,
                    ))
            except Exception as exc:  # collected into validation report
                issues.append(ValidationIssue(
                    code="SCENARIO-REFERENCE-FAIL",
                    severity=Severity.ERROR,
                    message=str(exc),
                    artifact_id=sid,
                ))

        # Every scenario must resolve to exactly one cluster/split.
        for scenario in scenarios:
            sid = str(scenario.scenario_id)
            cluster_id = cluster_of.get(sid)
            if cluster_id is None or cluster_id not in assignments:
                issues.append(ValidationIssue(
                    code="SPLIT-UNASSIGNED",
                    severity=Severity.FATAL,
                    message="Scenario does not resolve to a cluster assignment",
                    artifact_id=sid,
                ))

        # Family leakage check.
        family_splits: dict[str, set[str]] = defaultdict(set)
        for scenario in scenarios:
            cluster_id = cluster_of.get(str(scenario.scenario_id))
            if cluster_id and cluster_id in assignments:
                family_splits[str(scenario.family_id)].add(assignments[cluster_id].split.value)
        for family_id, splits in family_splits.items():
            if len(splits) > 1:
                issues.append(ValidationIssue(
                    code="SPLIT-FAMILY-LEAK",
                    severity=Severity.FATAL,
                    message=f"Family appears in multiple splits: {sorted(splits)}",
                    artifact_id=family_id,
                ))

        passed = not any(i.severity in {Severity.ERROR, Severity.FATAL} for i in issues)
        return BuildValidationResult(passed=passed, eligible_for_export=passed, issues=issues)
