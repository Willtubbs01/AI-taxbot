from enum import StrEnum


class QualityLevel(StrEnum):
    Q0 = "Q0"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class ReviewLevel(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"


class InformationState(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    NOT_ASKED = "not_asked"
    CONFLICTING = "conflicting"
    NOT_APPLICABLE = "not_applicable"


class FactOrigin(StrEnum):
    SYNTHETIC = "synthetic"
    USER_STATEMENT = "user_statement"
    TAX_DOCUMENT = "tax_document"
    DERIVED = "derived"
    EXTERNAL = "external"


class TruthPolarity(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    SYNTHETIC = "synthetic"


class RuleStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"


class RuleCompleteness(StrEnum):
    PARTIAL = "partial"
    COMPLETE = "complete"


class AnswerabilityStatus(StrEnum):
    ANSWERABLE = "answerable"
    NEEDS_INFORMATION = "needs_information"
    NEEDS_RULE_LOOKUP = "needs_rule_lookup"
    AMBIGUOUS = "ambiguous"
    OUTSIDE_SCOPE = "outside_scope"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    DEV_EVAL = "dev_eval"
    TEST_ID = "test_id"
    TEST_TEMPLATE_OOD = "test_template_ood"
    TEST_COMPOSITION_OOD = "test_composition_ood"
    TEST_TEMPORAL_OOD = "test_temporal_ood"
    TEST_ADVERSARIAL = "test_adversarial"
    TEST_ATTACK_OOD = "test_attack_ood"
    TEST_GOLD = "test_gold"


class ScenarioOrigin(StrEnum):
    GENERATED = "generated"
    MANUAL = "manual"
    MUTATED = "mutated"
    COUNTERFACTUAL = "counterfactual"


class MutationTruthEffect(StrEnum):
    MUST_CHANGE_ANALYSIS = "must_change_analysis"
    MUST_PRESERVE_ANALYSIS = "must_preserve_analysis"
    MAY_CHANGE_ANALYSIS = "may_change_analysis"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"
