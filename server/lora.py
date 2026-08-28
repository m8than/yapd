"""PEFT-compatible dynamic LoRA bank for the custom Turbo GPT-2 core."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

_TARGET_RE = re.compile(
    r"(?:^|.*\.)(h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj)))"
    r"\.lora_([AB])(?:\.[^.]+)?\.weight$"
)
_SUPPORTED_SUFFIXES = (
    "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
)


@dataclass(frozen=True)
class LoRAWeights:
    a: torch.Tensor                 # (rank, in)
    b: torch.Tensor                 # (out, rank)
    scale: float


@dataclass
class LoRAAdapter:
    name: str
    path: str
    rank: int
    alpha: float
    weights: dict[str, LoRAWeights]


class LoRABank:
    """Registered adapters. Slot zero is always the unadapted base model."""

    def __init__(self, adapters: list[LoRAAdapter], *, base_model_id: str):
        self.adapters = [
            LoRAAdapter("__base__", "", 0, 0.0, {}), *adapters,
        ]
        self.base_model_id = base_model_id
        self._slots = {adapter.name: i for i, adapter in enumerate(self.adapters)}
        if len(self._slots) != len(self.adapters):
            raise ValueError("duplicate LoRA adapter name")

    @property
    def names(self) -> list[str]:
        return [adapter.name for adapter in self.adapters[1:]]

    def resolve(self, model: str | None) -> int:
        if model in (None, "", "base", "turbo", self.base_model_id):
            return 0
        try:
            return self._slots[model]
        except KeyError as exc:
            raise ValueError(f"unknown model/LoRA adapter: {model}") from exc

    def metadata(self) -> list[dict]:
        return [
            {"name": a.name, "rank": a.rank, "path": a.path}
            for a in self.adapters[1:]
        ]

    def apply_(self, output: torch.Tensor, input: torch.Tensor, *,
               layer: int, module: str,
               groups: list[tuple[int, torch.Tensor]] | None) -> torch.Tensor:
        """Add per-group LoRA deltas to flattened projection output."""
        if not groups:
            return output
        key = f"h.{layer}.{module}"
        for slot, indices in groups:
            if slot == 0:
                continue
            weights = self.adapters[slot].weights.get(key)
            if weights is None or indices.numel() == 0:
                continue
            selected = input.index_select(0, indices)
            low_rank = F.linear(selected, weights.a)
            delta = F.linear(low_rank, weights.b)
            output.index_add_(0, indices, delta * weights.scale)
        return output

    @classmethod
    def load(cls, specs: list[str] | None, *, t3, device: torch.device,
             dtype: torch.dtype, max_rank: int = 64,
             base_model_id: str = "ResembleAI/chatterbox-turbo") -> "LoRABank":
        adapters = []
        for spec in specs or []:
            if "=" not in spec:
                raise ValueError(
                    f"invalid --lora-modules entry {spec!r}; expected name=path")
            name, source = spec.split("=", 1)
            name, source = name.strip(), source.strip()
            if not name or not source or name in {"base", "turbo", base_model_id}:
                raise ValueError(f"invalid LoRA registration: {spec!r}")
            adapters.append(_load_adapter(
                name, source, t3=t3, device=device, dtype=dtype,
                max_rank=max_rank,
            ))
        return cls(adapters, base_model_id=base_model_id)


def _resolve_adapter_dir(source: str) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    return Path(snapshot_download(
        source,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    ))


def _projection(t3, key: str):
    match = re.fullmatch(r"h\.(\d+)\.(.+)", key)
    if not match:
        raise ValueError(f"invalid projection key: {key}")
    layer, suffix = int(match.group(1)), match.group(2)
    if layer >= len(t3.tfmr.h) or suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported LoRA target: {key}")
    block = t3.tfmr.h[layer]
    if suffix == "attn.c_attn":
        return block.attn.c_attn
    if suffix == "attn.c_proj":
        return block.attn.c_proj
    if suffix == "mlp.c_fc":
        return block.mlp.c_fc
    return block.mlp.c_proj


def _load_adapter(name: str, source: str, *, t3, device, dtype,
                  max_rank: int) -> LoRAAdapter:
    directory = _resolve_adapter_dir(source)
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"LoRA {name!r} must contain adapter_config.json and "
            "adapter_model.safetensors")
    config = json.loads(config_path.read_text())
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"LoRA {name!r}: peft_type must be LORA")
    if config.get("bias", "none") != "none":
        raise ValueError(f"LoRA {name!r}: adapter bias is unsupported")
    if config.get("modules_to_save"):
        raise ValueError(f"LoRA {name!r}: modules_to_save is unsupported")
    if config.get("use_dora", False):
        raise ValueError(f"LoRA {name!r}: DoRA is unsupported")
    rank = int(config.get("r", 0))
    alpha = float(config.get("lora_alpha", rank))
    if rank <= 0 or rank > max_rank:
        raise ValueError(
            f"LoRA {name!r}: rank {rank} exceeds valid range 1..{max_rank}")
    scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)

    tensors = load_file(weights_path, device="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    unexpected = []
    for raw_key, tensor in tensors.items():
        if ".lora_" not in raw_key:
            continue
        match = _TARGET_RE.search(raw_key)
        if match is None:
            unexpected.append(raw_key)
            continue
        key, side = match.group(1), match.group(2)
        pairs.setdefault(key, {})[side] = tensor
    if unexpected:
        raise ValueError(
            f"LoRA {name!r} has unsupported targets: {unexpected[:4]}")
    if not pairs:
        raise ValueError(f"LoRA {name!r} contains no supported GPT-2 weights")

    loaded = {}
    for key, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA {name!r}: incomplete A/B pair for {key}")
        projection = _projection(t3, key)
        in_features, out_features = projection.weight.shape
        a, b = pair["A"], pair["B"]
        if a.shape != (rank, in_features) or b.shape != (out_features, rank):
            raise ValueError(
                f"LoRA {name!r}: {key} shapes A={tuple(a.shape)} "
                f"B={tuple(b.shape)}, expected {(rank, in_features)} and "
                f"{(out_features, rank)}")
        loaded[key] = LoRAWeights(
            a=a.to(device=device, dtype=dtype).contiguous(),
            b=b.to(device=device, dtype=dtype).contiguous(),
            scale=scale,
        )
    return LoRAAdapter(name, str(directory), rank, alpha, loaded)
