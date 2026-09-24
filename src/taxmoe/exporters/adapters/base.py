from typing import Protocol


class ChatAdapter(Protocol):
    def adapt(self, system: str, user: str, assistant: str) -> list[dict[str, str]]: ...
