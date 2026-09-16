"""Standalone ascriptor.kernel-unit/1 hooks for GDN-2 decode short-conv."""
from __future__ import annotations

from ref.reference import make_inputs, reference, validate_inputs, validate_reference


def _check_options(options: dict) -> None:
    if options["device"] != "a5" or options["backend"] != "cce":
        raise ValueError("gdn2_short_conv_decode declares only the a5/cce combination")
    if options["block_dim"] != 8:
        raise ValueError("gdn2_short_conv_decode fixes block_dim=8")


def kernel_for(case: dict | None = None):
    del case
    from kernels.step import gdn2_short_conv_decode_kernel

    return gdn2_short_conv_decode_kernel


def execute(inputs: dict, options: dict) -> dict:
    import torch
    from _unit_runner import launch_kernel

    _check_options(options)
    validate_inputs(inputs)
    y = torch.full((1, 6144), float("nan"), dtype=torch.bfloat16)
    new_cache = torch.full((6144, 4), float("nan"), dtype=torch.bfloat16)
    arguments = (inputs["x"], inputs["cache"], inputs["weight"], y, new_cache)
    got_y, got_cache = launch_kernel(kernel_for(), arguments, options)
    return {"y": got_y, "new_cache": got_cache}


__all__ = [
    "execute",
    "kernel_for",
    "make_inputs",
    "reference",
    "validate_inputs",
    "validate_reference",
]
