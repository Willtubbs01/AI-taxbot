from pathlib import Path
import yaml
from taxmoe.schemas.rule import TaxRule

class RuleRegistry:
    def __init__(self, rules: list[TaxRule], version: str = "0.1"):
        self.version = version
        self.rules = {r.rule_id: r for r in rules}

    @classmethod
    def from_yaml_files(cls, paths: list[str | Path], version: str = "0.1") -> "RuleRegistry":
        rules = []
        for path in paths:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            rules.extend(TaxRule.model_validate(x) for x in data.get("rules", []))
        return cls(rules, version=version)

    def get(self, rule_id: str):
        return self.rules.get(rule_id)

    def active_for_year(self, tax_year: int):
        return [
            r for r in self.rules.values()
            if r.status == "verified"
            and (not r.tax_years or tax_year in r.tax_years)
        ]
