from taxmoe.schemas.scenario import TaxScenario


def deduplicate_exact(scenarios: list[TaxScenario]) -> tuple[list[TaxScenario], dict[str, list[str]]]:
    kept = {}
    aliases: dict[str, list[str]] = {}
    for scenario in scenarios:
        h = scenario.content_hash_value
        if h not in kept:
            kept[h] = scenario
            aliases[str(scenario.scenario_id)] = []
        else:
            aliases[str(kept[h].scenario_id)].append(str(scenario.scenario_id))
    return list(kept.values()), aliases
