def target(task_analysis):
    return {"status": task_analysis.status.value, "missing_information": task_analysis.missing_fact_ids}
