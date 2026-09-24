from taxmoe.schemas.fact import Money


def capital_gain_or_loss(proceeds: Money, basis: Money) -> Money:
    if proceeds.currency != basis.currency:
        raise ValueError("Currency mismatch")
    return Money(amount_cents=proceeds.amount_cents - basis.amount_cents, currency=proceeds.currency)
