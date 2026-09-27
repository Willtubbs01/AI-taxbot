from types import SimpleNamespace
import pytest
import torch
from torch import nn
from taxmoe.modeling.taxmoe_loader import _validate_state_dict_result

class TinyTied(nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if tied:
            self.tie_weights()
    def tie_weights(self):
        self.lm_head.weight = self.embed.weight
    def get_input_embeddings(self):
        return self.embed
    def get_output_embeddings(self):
        return self.lm_head

def test_allows_missing_lm_head_when_tied():
    m = TinyTied(True)
    _validate_state_dict_result(m, ["lm_head.weight"], [])
    assert m.embed.weight.data_ptr() == m.lm_head.weight.data_ptr()

def test_rejects_other_missing_key():
    m = TinyTied(True)
    with pytest.raises(ValueError, match="STATE-DICT-MISMATCH"):
        _validate_state_dict_result(m, ["model.layers.0.foo"], [])

def test_rejects_lm_head_missing_when_not_tied():
    m = TinyTied(False)
    with pytest.raises(ValueError, match="STATE-DICT-MISMATCH"):
        _validate_state_dict_result(m, ["lm_head.weight"], [])
