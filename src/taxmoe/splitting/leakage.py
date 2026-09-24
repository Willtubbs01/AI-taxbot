from collections import defaultdict
from taxmoe.schemas.scenario import TaxScenario
from .splitter import SplitAssignment


def family_leakage(scenarios: list[TaxScenario], cluster_of: dict[str, str], assignments: dict[str, SplitAssignment]):
    by_family = defaultdict(set)
    for s in scenarios:
        cid = cluster_of[str(s.scenario_id)]
        by_family[str(s.family_id)].add(assignments[cid].split.value)
    return {family: sorted(splits) for family, splits in by_family.items() if len(splits) > 1}
