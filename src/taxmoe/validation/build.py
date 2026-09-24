from pathlib import Path
from taxmoe.schemas.common import ValidationIssue, ValidationResult
from taxmoe.schemas.enums import Severity
from taxmoe.schemas.scenario import TaxScenario
from taxmoe.schemas.manifest import SplitManifest
from taxmoe.ingestion.hashing import sha256_file
from taxmoe.scenarios.inference import ScenarioInferenceEngine
from taxmoe.generation.fingerprints import content_hash, semantic_fingerprint, structural_fingerprint
from taxmoe.splitting.leakage import audit_leakage

class BuildValidator:
    def __init__(self, inference: ScenarioInferenceEngine):
        self.inference = inference

    def validate_scenarios(self, scenarios: list[TaxScenario]) -> ValidationResult:
        issues = []
        for s in scenarios:
            recomputed = self.inference.analyze(s.input, [x.task_id for x in s.analysis.task_analyses])
            if recomputed.model_dump(mode="json") != s.analysis.model_dump(mode="json"):
                issues.append(ValidationIssue(
                    code="SCENARIO-ANALYSIS-MISMATCH",
                    severity=Severity.ERROR,
                    message="Stored analysis differs from recomputed analysis.",
                    artifact_id=s.scenario_id,
                ))
            if content_hash(s.input, recomputed) != s.content_hash:
                issues.append(ValidationIssue(
                    code="SCENARIO-CONTENT-HASH-MISMATCH",
                    severity=Severity.ERROR,
                    message="Scenario content hash differs from recomputed hash.",
                    artifact_id=s.scenario_id,
                ))
            if semantic_fingerprint(s.input, recomputed) != s.semantic_fingerprint:
                issues.append(ValidationIssue(
                    code="SCENARIO-SEMANTIC-FINGERPRINT-MISMATCH",
                    severity=Severity.ERROR,
                    message="Scenario semantic fingerprint mismatch.",
                    artifact_id=s.scenario_id,
                ))
            if structural_fingerprint(s.input, recomputed) != s.structural_fingerprint:
                issues.append(ValidationIssue(
                    code="SCENARIO-STRUCTURAL-FINGERPRINT-MISMATCH",
                    severity=Severity.ERROR,
                    message="Scenario structural fingerprint mismatch.",
                    artifact_id=s.scenario_id,
                ))
        return ValidationResult(passed=not any(i.severity in {Severity.ERROR, Severity.FATAL} for i in issues), issues=issues)

    def validate_split(self, scenarios: list[TaxScenario], manifest: SplitManifest) -> ValidationResult:
        leak = audit_leakage(scenarios, manifest)
        issues = []
        if not leak["passed"]:
            issues.append(ValidationIssue(
                code="SPLIT-LEAKAGE",
                severity=Severity.FATAL,
                message="Family or exact-content leakage detected across splits.",
                details=leak,
            ))
        return ValidationResult(passed=not issues, issues=issues)

def verify_source_hashes(source_registry, project_root: str | Path) -> ValidationResult:
    issues = []
    root = Path(project_root)
    for src in source_registry.all():
        path = root / src.raw_path
        if not path.exists():
            issues.append(ValidationIssue(
                code="SRC-MISSING-RAW",
                severity=Severity.ERROR,
                message=f"Missing raw source: {path}",
                artifact_id=src.source_id,
            ))
            continue
        if src.sha256 and src.sha256 != "REPLACE_ME" and sha256_file(path) != src.sha256:
            issues.append(ValidationIssue(
                code="SRC-HASH-MISMATCH",
                severity=Severity.FATAL,
                message=f"Raw source hash mismatch: {path}",
                artifact_id=src.source_id,
            ))
    return ValidationResult(passed=not any(i.severity in {Severity.ERROR, Severity.FATAL} for i in issues), issues=issues)
