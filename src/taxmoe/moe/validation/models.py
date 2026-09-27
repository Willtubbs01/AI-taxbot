from __future__ import annotations

from pydantic import Field

from taxmoe.schemas.common import TaxMoEModel


class NumericalComparison(TaxMoEModel):
    max_abs: float
    mean_abs: float
    max_rel: float
    passed: bool


class LayerEquivalenceResult(TaxMoEModel):
    layer_index: int
    comparison: NumericalComparison


class UpcycleValidationReport(TaxMoEModel):
    schema_version: str = "1"
    target_layers: list[int]
    structure_passed: bool
    expert_copy_passed: bool
    local_equivalence: list[LayerEquivalenceResult] = Field(default_factory=list)
    full_model_logits: NumericalComparison | None = None
    lm_loss_delta: float | None = None
    lm_loss_tolerance: float | None = None
    lm_loss_passed: bool | None = None
    passed: bool
    failures: list[str] = Field(default_factory=list)
