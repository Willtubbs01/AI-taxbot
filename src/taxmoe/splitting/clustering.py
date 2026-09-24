from collections import defaultdict
from taxmoe.schemas.scenario import TaxScenario


def group_by_structural_fingerprint(scenarios: list[TaxScenario]):
    groups = defaultdict(list)
    for s in scenarios:
        groups[s.structural_fingerprint].append(s)
    return dict(groups)
