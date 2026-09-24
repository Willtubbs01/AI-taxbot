from pathlib import Path
from taxmoe.knowledge.form_registry import FormRegistry
from taxmoe.knowledge.rule_registry import RuleRegistry
from taxmoe.knowledge.values import TaxValueStore
from taxmoe.scenarios.template_registry import TemplateRegistry
from taxmoe.scenarios.inference import ScenarioInferenceEngine
from taxmoe.generation.scenario_generator import ScenarioGenerator

ROOT = Path(__file__).resolve().parents[1]

def make_generator():
    forms = FormRegistry.from_yaml(ROOT/"knowledge/forms/forms.yaml", ROOT/"knowledge/forms/form_versions_2025.yaml")
    rules = RuleRegistry.from_yaml_files([
        ROOT/"knowledge/rules/wages_2025.yaml",
        ROOT/"knowledge/rules/investment_2025.yaml",
    ])
    values = TaxValueStore.from_yaml(ROOT/"knowledge/values/values_2025.yaml")
    templates = TemplateRegistry.from_paths([
        ROOT/"scenarios/templates/wages/wages_basic.yaml",
        ROOT/"scenarios/templates/investments/stock_sale_basic.yaml",
    ])
    inference = ScenarioInferenceEngine(rules, values)
    return ScenarioGenerator(templates, forms, values, inference)

def test_generation_deterministic():
    g = make_generator()
    a = g.generate("TPL-STOCK-SALE-BASIC", 2025, 123, index=0)
    b = g.generate("TPL-STOCK-SALE-BASIC", 2025, 123, index=0)
    assert a.content_hash == b.content_hash
    assert a.scenario_id == b.scenario_id

def test_generated_stock_scenario_answerable():
    g = make_generator()
    s = g.generate("TPL-STOCK-SALE-BASIC", 2025, 123, index=0)
    assert all(not t.missing_concept_ids for t in s.analysis.task_analyses)
