import json
from taxmoe.exporters.adapters.qwen import to_qwen_messages

def test_qwen_messages_roles():
    messages = to_qwen_messages("hello", "{}")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
