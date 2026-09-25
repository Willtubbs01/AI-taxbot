from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Stage4Model
from .inputs import LossMode


class ModelIdentityConfig(Stage4Model):
    model_id: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None


class ChatTokenizationConfig(Stage4Model):
    enable_thinking: bool = False
    add_generation_prompt: bool = False


class SequenceConfig(Stage4Model):
    max_tokens: int = Field(default=4096, gt=0)
    truncation: bool = False
    overflow_policy: Literal["reject"] = "reject"


class TokenizationConfig(Stage4Model):
    version: str = "1"
    model: ModelIdentityConfig = ModelIdentityConfig()
    chat: ChatTokenizationConfig = ChatTokenizationConfig()
    sequence: SequenceConfig = SequenceConfig()
    cpt_loss: LossMode = LossMode.FULL_CAUSAL
    sft_loss: LossMode = LossMode.ASSISTANT_ONLY
    require_fast_tokenizer: bool = True
    require_offsets: bool = True
    require_assistant_mask: bool = True
