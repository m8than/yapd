"""Behavioral checks for PEFT loading and mixed-row LoRA application."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2Model

from server.lora import LoRABank
from server.engine_ar import GPT2Core


class FakeT3:
    def __init__(self):
        self.tfmr = GPT2Model(GPT2Config(
            n_layer=1, n_head=2, n_embd=8, n_positions=32,
        ))


def main() -> None:
    torch.manual_seed(7)
    t3 = FakeT3()
    rank, alpha = 2, 4
    a = torch.randn(rank, 8)
    b = torch.randn(24, rank)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": alpha,
            "fan_in_fan_out": True,
            "bias": "none",
            "inference_mode": True,
            "target_modules": ["c_attn"],
        }))
        save_file({
            "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight": a,
            "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight": b,
        }, directory / "adapter_model.safetensors")
        bank = LoRABank.load(
            [f"voice-a={directory}"], t3=t3, device=torch.device("cpu"),
            dtype=torch.float32, max_rank=8,
        )

    assert bank.resolve(None) == 0
    assert bank.resolve("ResembleAI/chatterbox-turbo") == 0
    assert bank.resolve("voice-a") == 1
    try:
        bank.resolve("missing")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown adapter must fail")

    x = torch.randn(3, 8)
    output = torch.zeros(3, 24)
    indices = torch.tensor([0, 2])
    bank.apply_(
        output, x, layer=0, module="attn.c_attn",
        groups=[(1, indices)],
    )
    expected = torch.zeros_like(output)
    expected[indices] = ((x[indices] @ a.T) @ b.T) * (alpha / rank)
    torch.testing.assert_close(output, expected)
    assert torch.count_nonzero(output[1]) == 0

    untouched = torch.randn(3, 8)

    core = GPT2Core(t3.tfmr, torch.float32, torch.device("cpu"), bank)
    projection = t3.tfmr.h[0].attn.c_attn
    projected = core._project(
        x, projection.weight, projection.bias,
        layer=0, module="attn.c_attn", groups=[(1, indices)],
    )
    expected_projection = torch.addmm(
        projection.bias, x, projection.weight,
    ) + expected
    torch.testing.assert_close(projected, expected_projection)
    base_output = torch.zeros(3, 8)
    bank.apply_(
        base_output, untouched, layer=0, module="attn.c_proj",
        groups=[(1, indices)],
    )
    assert torch.count_nonzero(base_output) == 0
    print("dynamic LoRA tests passed")


if __name__ == "__main__":
    main()
