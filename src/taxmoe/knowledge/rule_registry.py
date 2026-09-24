from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from taxmoe.schemas.rule import TaxRule


class RuleRegistry:
    def __init__(self, rules: Iterable[TaxRule], version: str = "0.1"):
        self.version = version
        self._rules: dict[tuple[str, int], TaxRule] = {}
        for rule in rules:
            key = (str(rule.rule_id), rule.revision)
            if key in self._rules:
                raise ValueError(f"Duplicate rule revision: {key}")
            self._rules[key] = rule

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            [TaxRule.model_validate(x) for x in raw.get("rules", [])],
            version=str(raw.get("version", "0.1")),
        )

    def get(self, rule_id: str, revision: int | None = None) -> TaxRule | None:
        candidates = [r for (rid, _), r in self._rules.items() if rid == rule_id]
        if not candidates:
            return None
        if revision is None:
            return max(candidates, key=lambda r: r.revision)
        return self._rules.get((rule_id, revision))

    def applicable(self, tax_year: int) -> list[TaxRule]:
        return [
            r for r in self._rules.values()
            if not r.tax_years or tax_year in r.tax_years
        ]
