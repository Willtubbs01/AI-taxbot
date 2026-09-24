from typing import Callable, Any

Calculator = Callable[[dict[str, Any]], Any]

class CalculationRegistry:
    def __init__(self, version: str = "0.1"):
        self.version = version
        self._items: dict[str, Calculator] = {}

    def register(self, calculation_id: str, fn: Calculator) -> None:
        if calculation_id in self._items:
            raise ValueError(f"Duplicate calculation: {calculation_id}")
        self._items[calculation_id] = fn

    def run(self, calculation_id: str, inputs: dict[str, Any]):
        try:
            fn = self._items[calculation_id]
        except KeyError:
            raise KeyError(f"Unknown calculation: {calculation_id}")
        return fn(inputs)

    def has(self, calculation_id: str) -> bool:
        return calculation_id in self._items
