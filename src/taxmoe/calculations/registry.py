from __future__ import annotations
from collections.abc import Callable


class CalculationRegistry:
    def __init__(self):
        self._calculators: dict[str, Callable[..., object]] = {}

    def register(self, calculation_id: str, fn: Callable[..., object]) -> None:
        if calculation_id in self._calculators:
            raise ValueError(f"Duplicate calculation: {calculation_id}")
        self._calculators[calculation_id] = fn

    def calculate(self, calculation_id: str, **kwargs):
        try:
            fn = self._calculators[calculation_id]
        except KeyError as exc:
            raise KeyError(f"Unknown calculation: {calculation_id}") from exc
        return fn(**kwargs)
