class ConciseRenderer:
    id = "concise"
    version = "0.1"

    def render(self, payload: dict[str, object]) -> str:
        return "; ".join(f"{k}={v}" for k, v in payload.items())
