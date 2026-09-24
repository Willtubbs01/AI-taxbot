SYSTEM_STANDARD = (
    "You are a specialized U.S. federal individual-income-tax reasoning component. "
    "Analyze only the provided task and facts. Do not invent missing taxpayer information. "
    "Use the requested output format."
)

def to_qwen_messages(user_text: str, assistant_text: str, system_text: str = SYSTEM_STANDARD):
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
