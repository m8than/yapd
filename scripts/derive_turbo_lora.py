#!/usr/bin/env python3
"""Derive a PEFT LoRA from the projection deltas of a full Turbo T3 checkpoint."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET = re.compile(
    r"tfmr\.(h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj)))\.weight$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fine-tuned", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rank = args.rank
    if rank <= 0:
        raise ValueError("rank must be positive")

    base = safe_open(args.base, framework="pt", device="cpu")
    fine = safe_open(args.fine_tuned, framework="pt", device="cpu")
    base_keys, fine_keys = set(base.keys()), set(fine.keys())
    common = sorted(base_keys & fine_keys)
    targets = [(key, TARGET.fullmatch(key)) for key in common]
    targets = [(key, match.group(1)) for key, match in targets if match]
    if not targets:
        raise ValueError("no supported Turbo GPT-2 projections found")

    tensors = {}
    per_target = {}
    captured_energy = total_energy = 0.0
    for index, (checkpoint_key, peft_key) in enumerate(targets, 1):
        base_weight = base.get_tensor(checkpoint_key).to(args.device, torch.float32)
        fine_weight = fine.get_tensor(checkpoint_key).to(args.device, torch.float32)
        if base_weight.shape != fine_weight.shape or base_weight.ndim != 2:
            raise ValueError(f"shape mismatch for {checkpoint_key}")
        delta = fine_weight - base_weight                 # Conv1D: (in, out)
        actual_rank = min(rank, *delta.shape)
        q = min(actual_rank + 8, *delta.shape)
        u, singular, v = torch.svd_lowrank(delta, q=q, niter=4)
        u, singular, v = u[:, :actual_rank], singular[:actual_rank], v[:, :actual_rank]
        root = singular.clamp_min(0).sqrt()
        lora_a = (root[:, None] * u.T).contiguous()       # (r, in)
        lora_b = (v * root[None, :]).contiguous()         # (out, r)
        prefix = f"base_model.model.transformer.{peft_key}"
        tensors[prefix + ".lora_A.weight"] = lora_a.cpu()
        tensors[prefix + ".lora_B.weight"] = lora_b.cpu()
        total = delta.square().sum().item()
        captured = singular.square().sum().item()
        total_energy += total
        captured_energy += captured
        per_target[peft_key] = {
            "delta_energy": total,
            "captured_energy": captured,
            "retained_fraction": captured / total if total else 1.0,
        }
        print(
            f"[{index:02d}/{len(targets)}] {peft_key}: "
            f"{per_target[peft_key]['retained_fraction']:.2%}",
            flush=True,
        )
        del base_weight, fine_weight, delta, u, singular, v, root, lora_a, lora_b

    save_file(tensors, output / "adapter_model.safetensors")
    (output / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "base_model_name_or_path": "ResembleAI/chatterbox-turbo",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "fan_in_fan_out": True,
        "bias": "none",
        "inference_mode": True,
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "use_rslora": False,
        "use_dora": False,
    }, indent=2))
    (output / "derivation.json").write_text(json.dumps({
        "source_model": args.source_model,
        "source_license": args.license,
        "base_checkpoint": str(Path(args.base).resolve()),
        "fine_tuned_checkpoint": str(Path(args.fine_tuned).resolve()),
        "rank": rank,
        "retained_fraction": captured_energy / total_energy,
        "targets": per_target,
        "limitations": [
            "Projection deltas only; embeddings, norms, heads, and biases are not represented.",
            "Derived by truncated SVD and not published by the source model author.",
        ],
    }, indent=2))
    print(
        f"wrote {output}: global retained delta energy "
        f"{captured_energy / total_energy:.2%}",
    )


if __name__ == "__main__":
    main()
