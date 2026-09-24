import json


class StructuredRenderer:
    id = "structured"
    version = "0.1"

    def render(self, payload: dict[str, object]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
