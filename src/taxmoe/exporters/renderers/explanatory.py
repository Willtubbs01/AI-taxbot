class ExplanatoryRenderer:
    id = "explanatory"
    version = "0.1"

    def render(self, payload: dict[str, object]) -> str:
        return "\n".join(f"{k}: {v}" for k, v in payload.items())
