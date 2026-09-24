from enum import Enum

class QualityLevel(str, Enum):
    Q0 = "Q0"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"

class ReviewLevel(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

class InformationState(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    NOT_ASKED = "not_asked"
    CONFLICTING = "conflicting"
    NOT_APPLICABLE = "not_applicable"

class FactOrigin(str, Enum):
    SYNTHETIC = "synthetic"
    USER = "user"
    DOCUMENT = "document"
    DERIVED = "derived"

class AnswerabilityStatus(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    NEEDS_INFORMATION = "NEEDS_INFORMATION"
    NEEDS_RULE_LOOKUP = "NEEDS_RULE_LOOKUP"
    AMBIGUOUS = "AMBIGUOUS"
    OUTSIDE_SCOPE = "OUTSIDE_SCOPE"

class ScenarioOrigin(str, Enum):
    GENERATED = "generated"
    MANUAL = "manual"
    MUTATED = "mutated"
    COUNTERFACTUAL = "counterfactual"

class DatasetSplit(str, Enum):
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

class ValidationMode(str, Enum):
    FAST = "fast"
    FULL = "full"
    FREEZE = "freeze"

class MutationTruthEffect(str, Enum):
    MUST_CHANGE_ANALYSIS = "must_change_analysis"
    MUST_PRESERVE_ANALYSIS = "must_preserve_analysis"
    MAY_CHANGE_ANALYSIS = "may_change_analysis"
