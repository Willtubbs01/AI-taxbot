from __future__ import annotations

from .base import Stage4Model


class TokenizerCapabilities(Stage4Model):
    is_fast: bool
    has_chat_template: bool
    supports_offsets: bool
    supports_assistant_mask: bool
    supports_thinking_switch: bool


class TokenizerManifest(Stage4Model):
    model_id: str
    revision: str
    tokenizer_class: str
    vocab_size: int
    special_tokens_map: dict[str, object]
    chat_template_sha256: str | None = None
    capabilities: TokenizerCapabilities
    transformers_version: str
    tokenizers_version: str | None = None
