from __future__ import annotations

def inspect_special_tokens(tokenizer):
    return {
        'bos_token': tokenizer.bos_token,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token': tokenizer.eos_token,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token,
        'pad_token_id': tokenizer.pad_token_id,
        'additional_special_tokens': list(tokenizer.additional_special_tokens or []),
        'additional_special_tokens_ids': list(tokenizer.additional_special_tokens_ids or []),
    }
