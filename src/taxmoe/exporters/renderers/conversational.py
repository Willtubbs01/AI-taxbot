class ConversationalRenderer:
    id = "conversational"
    version = "0.1"

    def render(self, payload: dict[str, object]) -> str:
        return "Tax scenario information: " + ", ".join(f"{k}={v}" for k, v in payload.items())
