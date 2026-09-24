class QwenChatAdapter:
    id = "qwen_chat"
    version = "0.1"

    def adapt(self, system: str, user: str, assistant: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
