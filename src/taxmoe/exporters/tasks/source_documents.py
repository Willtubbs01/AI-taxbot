def target(scenario):
    return {"source_documents": sorted({d.form_id for d in scenario.input.documents})}
