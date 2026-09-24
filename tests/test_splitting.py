from taxmoe.schemas.scenario import TaxScenario
from taxmoe.splitting.splitter import SplitEngine

def test_split_engine_stable_for_empty():
    m1 = SplitEngine().build([])
    m2 = SplitEngine().build([])
    assert m1.model_dump() == m2.model_dump()
