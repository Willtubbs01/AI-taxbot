from taxmoe.schemas.common import Money
from .registry import CalculationRegistry

def capital_gain_or_loss(inputs: dict):
    proceeds = inputs["investment.proceeds"]
    basis = inputs["investment.basis"]
    p = proceeds.amount_cents if isinstance(proceeds, Money) else int(proceeds["amount_cents"])
    b = basis.amount_cents if isinstance(basis, Money) else int(basis["amount_cents"])
    return Money(amount_cents=p - b)

def default_calculation_registry() -> CalculationRegistry:
    r = CalculationRegistry(version="0.1")
    r.register("CALC-CAPITAL-GAIN-LOSS", capital_gain_or_loss)
    return r
