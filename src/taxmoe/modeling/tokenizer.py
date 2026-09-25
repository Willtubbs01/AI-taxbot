from __future__ import annotations

import hashlib
import inspect
from typing import Any

from transformers import AutoTokenizer

from .inputs import ChatMessage, TokenSpan, TokenizedText
from .tokenization_config import TokenizationConfig
from .tokenization_manifest import TokenizerCapabilities, TokenizerManifest


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TokenizerCapabilityError(RuntimeError):
    pass


class TaxMoETokenizer:
    """Single controlled tokenizer boundary for Stage 4."""

    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            use_fast=True,
        )
        if config.require_fast_tokenizer and not self.tokenizer.is_fast:
            raise TokenizerCapabilityError("TOKENIZER-FAST-REQUIRED")

    @staticmethod
    def _messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [m.model_dump() for m in messages]

    def render_messages(self, messages: list[ChatMessage], *, add_generation_prompt: bool | None = None) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": self.config.chat.add_generation_prompt if add_generation_prompt is None else add_generation_prompt,
        }
        # Qwen3 supports this kwarg; older templates may ignore/reject it.
        kwargs["enable_thinking"] = self.config.chat.enable_thinking
        return self.tokenizer.apply_chat_template(self._messages(messages), **kwargs)

    def tokenize_text(self, text: str) -> TokenizedText:
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=self.config.require_offsets,
        )
        ids = list(enc["input_ids"])
        if len(ids) > self.config.sequence.max_tokens:
            raise ValueError(f"TOKENIZE-SEQUENCE-OVERFLOW:{len(ids)}>{self.config.sequence.max_tokens}")
        offsets = enc.get("offset_mapping") or []
        specials = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        spans: list[TokenSpan] = []
        for i, tid in enumerate(ids):
            start = end = None
            special = tid in specials
            if offsets:
                a, b = map(int, offsets[i])
                if not special and b > a:
                    start, end = a, b
            spans.append(TokenSpan(token_index=i, token_id=int(tid), start_char=start, end_char=end, special=special))
        return TokenizedText(
            input_ids=ids,
            attention_mask=list(enc.get("attention_mask", [1] * len(ids))),
            spans=spans,
            token_count=len(ids),
            tokenizer_id=self.config.model.model_id,
            tokenizer_revision=self.config.model.revision,
            text_sha256=sha256_text(text),
        )

    def tokenize_messages_with_mask(self, messages: list[ChatMessage]) -> tuple[list[int], list[int], list[int]]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "add_generation_prompt": False,
            "return_assistant_tokens_mask": True,
            "enable_thinking": self.config.chat.enable_thinking,
        }
        out = self.tokenizer.apply_chat_template(self._messages(messages), **kwargs)
        ids = list(out["input_ids"])
        attention = list(out.get("attention_mask", [1] * len(ids)))
        mask = out.get("assistant_masks")
        if mask is None:
            mask = out.get("assistant_tokens_mask")
        if mask is None:
            raise TokenizerCapabilityError("TOKENIZER-ASSISTANT-MASK-UNSUPPORTED")
        mask = list(mask)
        if len(mask) != len(ids):
            raise TokenizerCapabilityError("TOKENIZER-ASSISTANT-MASK-LENGTH-MISMATCH")
        if len(ids) > self.config.sequence.max_tokens:
            raise ValueError(f"TOKENIZE-SEQUENCE-OVERFLOW:{len(ids)}>{self.config.sequence.max_tokens}")
        return ids, attention, mask

    def inspect_capabilities(self) -> TokenizerCapabilities:
        supports_offsets = False
        try:
            enc = self.tokenizer("Form W-2", return_offsets_mapping=True, add_special_tokens=True)
            supports_offsets = len(enc.get("offset_mapping", [])) == len(enc["input_ids"])
        except Exception:
            supports_offsets = False

        has_template = bool(getattr(self.tokenizer, "chat_template", None))
        supports_mask = False
        supports_thinking = False
        if has_template:
            sample = [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="Hi")]
            try:
                _, _, mask = self.tokenize_messages_with_mask(sample)
                supports_mask = any(mask)
            except Exception:
                supports_mask = False
            try:
                _ = self.tokenizer.apply_chat_template(self._messages(sample), tokenize=False, enable_thinking=False)
                supports_thinking = True
            except Exception:
                supports_thinking = False

        caps = TokenizerCapabilities(
            is_fast=bool(self.tokenizer.is_fast),
            has_chat_template=has_template,
            supports_offsets=supports_offsets,
            supports_assistant_mask=supports_mask,
            supports_thinking_switch=supports_thinking,
        )
        if self.config.require_offsets and not caps.supports_offsets:
            raise TokenizerCapabilityError("TOKENIZER-OFFSETS-UNSUPPORTED")
        if self.config.require_assistant_mask and not caps.supports_assistant_mask:
            raise TokenizerCapabilityError("TOKENIZER-ASSISTANT-MASK-UNSUPPORTED")
        return caps

    def manifest(self) -> TokenizerManifest:
        import transformers
        try:
            import tokenizers
            tokenizers_version = tokenizers.__version__
        except Exception:
            tokenizers_version = None
        if not self.config.model.revision:
            raise ValueError("Tokenizer revision must be pinned before Stage 4 freeze")
        template = getattr(self.tokenizer, "chat_template", None)
        return TokenizerManifest(
            model_id=self.config.model.model_id,
            revision=self.config.model.revision,
            tokenizer_class=type(self.tokenizer).__name__,
            vocab_size=len(self.tokenizer),
            special_tokens_map=dict(self.tokenizer.special_tokens_map),
            chat_template_sha256=sha256_text(template) if template else None,
            capabilities=self.inspect_capabilities(),
            transformers_version=transformers.__version__,
            tokenizers_version=tokenizers_version,
        )
