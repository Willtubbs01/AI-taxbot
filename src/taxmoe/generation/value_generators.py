from __future__ import annotations
import random
from datetime import date, timedelta
from decimal import Decimal
from taxmoe.schemas.common import Money
from taxmoe.scenarios.templates import ValueGeneratorSpec
from taxmoe.knowledge.values import TaxValueStore

class ValueGenerationContext:
    def __init__(self, tax_year: int, generated: dict, value_store: TaxValueStore, rng: random.Random):
        self.tax_year = tax_year
        self.generated = generated
        self.value_store = value_store
        self.rng = rng

def generate_value(spec: ValueGeneratorSpec, value_type: str, ctx: ValueGenerationContext):
    kind = spec.type
    rng = ctx.rng

    if kind == "constant":
        return spec.value
    if kind == "choice":
        return rng.choice(spec.values or [])
    if kind == "weighted_choice":
        return rng.choices(spec.values or [], weights=spec.weights, k=1)[0]
    if kind in {"integer_range", "money_range"}:
        lo, hi = int(spec.min), int(spec.max)
        step = int(spec.step or 1)
        count = (hi - lo) // step
        raw = lo + step * rng.randint(0, count)
        return Money(amount_cents=raw) if kind == "money_range" or value_type == "money" else raw
    if kind == "boolean_probability":
        p = float(spec.value if spec.value is not None else 0.5)
        return rng.random() < p
    if kind == "date_in_tax_year":
        start = date(ctx.tax_year, 1, 1)
        end = date(ctx.tax_year, 12, 31)
        return start + timedelta(days=rng.randint(0, (end - start).days))
    if kind == "date_year_offset":
        offsets = spec.values or [0]
        year = ctx.tax_year + int(rng.choice(offsets))
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return start + timedelta(days=rng.randint(0, (end - start).days))
    if kind == "relative_date":
        ref = ctx.generated[spec.reference]
        return ref + timedelta(days=rng.randint(int(spec.min_days_after or 1), int(spec.max_days_after or 365)))
    if kind == "value_ref":
        value = ctx.value_store.resolve(spec.value_ref, ctx.tax_year)
        if value is None:
            raise ValueError(f"Missing tax value {spec.value_ref}/{ctx.tax_year}")
        return value
    if kind == "value_ref_boundary":
        value = ctx.value_store.resolve(spec.value_ref, ctx.tax_year)
        if value is None:
            raise ValueError(f"Missing tax value {spec.value_ref}/{ctx.tax_year}")
        delta = spec.delta or 1
        if spec.boundary_position == "below":
            return value - delta
        if spec.boundary_position == "above":
            return value + delta
        return value
    raise ValueError(f"Unsupported generator type: {kind}")
