from taxmoe.schemas.scenario import TaxScenario

class ConciseRenderer:
    renderer_id = "concise"
    version = "0.1"

    def render(self, scenario: TaxScenario, task_id: str) -> tuple[str, str]:
        task = next(x for x in scenario.analysis.task_analyses if x.task_id == task_id)
        facts = "; ".join(f"{f.concept_id}={f.value if f.value is not None else f.state.value}" for f in scenario.input.facts)
        prompt = f"Tax year {scenario.input.tax_year}. Facts: {facts}. Task: {task_id}."
        pieces = [task.status.value]
        if task.missing_concept_ids:
            pieces.append("Missing: " + ", ".join(task.missing_concept_ids))
        if task.required_forms:
            pieces.append("Required forms: " + ", ".join(task.required_forms))
        if task.candidate_forms:
            pieces.append("Candidate forms: " + ", ".join(task.candidate_forms))
        if task.required_calculations:
            pieces.append("Calculations: " + ", ".join(task.required_calculations))
        return prompt, " | ".join(pieces)
