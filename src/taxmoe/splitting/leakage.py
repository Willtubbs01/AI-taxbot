from collections import defaultdict
from taxmoe.schemas.scenario import TaxScenario
from taxmoe.schemas.manifest import SplitManifest

def audit_leakage(scenarios: list[TaxScenario], manifest: SplitManifest) -> dict:
    cluster_split = {x.cluster_id: x.split for x in manifest.assignments}
    family_splits = defaultdict(set)
    content_splits = defaultdict(set)

    for s in scenarios:
        cluster = manifest.scenario_to_cluster[s.scenario_id]
        split = cluster_split[cluster]
        family_splits[s.family_id].add(split)
        content_splits[s.content_hash].add(split)

    family_leaks = {k: sorted(v) for k, v in family_splits.items() if len(v) > 1}
    content_leaks = {k: sorted(v) for k, v in content_splits.items() if len(v) > 1}

    return {
        "passed": not family_leaks and not content_leaks,
        "family_leaks": family_leaks,
        "content_leaks": content_leaks,
    }
