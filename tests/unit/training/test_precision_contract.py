import pytest
import torch

from taxmoe.training.optimization import OptimizationConfig
from taxmoe.training.trainer import DenseCPTTrainer


class _NoopScheduler:
    def step(self):
        pass


def _cfg(precision: str) -> OptimizationConfig:
    return OptimizationConfig(precision=precision, gradient_accumulation_steps=1)


def test_fp16_trainable_parameters_are_rejected_with_grad_scaler(monkeypatch):
    model = torch.nn.Linear(4, 4).half()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="TRAIN-FP16-PARAMETERS-WITH-GRADSCALER"):
        DenseCPTTrainer(
            model,
            optimizer,
            _NoopScheduler(),
            _cfg("fp16"),
            unique_train_tokens=1,
            device="cuda",
        )


def test_bf16_parameters_do_not_enable_grad_scaler():
    model = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = DenseCPTTrainer(
        model,
        optimizer,
        _NoopScheduler(),
        _cfg("bf16"),
        unique_train_tokens=1,
        device="cpu",
    )
    assert trainer.scaler.is_enabled() is False
