def analysis_changed(parent, child) -> bool:
    return parent.analysis.model_dump(mode="json") != child.analysis.model_dump(mode="json")
