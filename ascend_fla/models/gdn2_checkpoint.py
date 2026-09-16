"""GDN-2 LitGPT/Fabric checkpoint 的规范化工具。"""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["checkpoint_state_dict"]


def checkpoint_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    """取出模型权重，并剥掉 Fabric/DDP/compile 常见 wrapper 前缀。"""
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not isinstance(state, dict):
        raise TypeError("checkpoint 必须是 state_dict，或包含一个 'model' state_dict")

    prefixes = ("module.", "_orig_mod.", "model.", "_forward_module.")
    normalized: dict[str, torch.Tensor] = {}
    for source_key, value in state.items():
        key = source_key
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"模型 state_dict 含非 tensor 项：{key!r} -> {type(value).__name__}")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key in normalized:
            raise ValueError(
                f"checkpoint wrapper 前缀规范化后出现重复 key {key!r}（含 {source_key!r}）"
            )
        normalized[key] = value
    return normalized
