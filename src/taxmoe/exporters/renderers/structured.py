import json
from taxmoe.schemas.scenario import TaxScenario

class StructuredRenderer:
    renderer_id = "structured"
    version = "0.1"

    def render(self, scenario: TaxScenario, task_id: str) -> tuple[str, str]:
        task = next(x for x in scenario.analysis.task_analyses if x.task_id == task_id)
        request = {
            "protocol_version": "0.1",
            "task": task_id,
            "jurisdiction": scenario.input.jurisdiction.model_dump(mode="json"),
            "tax_year": scenario.input.tax_year,
            "facts": [f.model_dump(mode="json") for f in scenario.input.facts],
            "documents": [d.model_dump(mode="json") for d in scenario.input.documents],
            "context": [c.model_dump(mode="json") for c in scenario.input.context_items],
        }
        response = {
            "protocol_version": "0.1",
            "status": task.status.value,
            "topics": task.topic_ids,
            "source_documents": task.source_document_ids,
            "candidate_forms": task.candidate_forms,
            "required_forms": task.required_forms,
            "missing_information": task.missing_concept_ids,
            "conflicts": task.conflicting_concept_ids,
            "retrieval_requests": task.required_rule_lookups,
            "calculation_requests": task.required_calculations,
        }
        return (
            json.dumps(request, ensure_ascii=False, sort_keys=True, default=str),
            json.dumps(response, ensure_ascii=False, sort_keys=True, default=str),
        )
