from decimal import Decimal
from taxmoe.schemas.common import Money, Jurisdiction
from taxmoe.schemas.enums import InformationState

def test_money_exact():
    assert Money(amount_cents=12345).amount == Decimal("123.45")

def test_jurisdiction_default():
    j = Jurisdiction()
    assert j.country == "US"
    assert j.level == "federal"

def test_information_states_exist():
    assert InformationState.UNKNOWN.value == "unknown"
