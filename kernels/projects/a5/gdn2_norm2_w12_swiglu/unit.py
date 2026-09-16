"""Standalone ascriptor.kernel-unit/1 hooks for the mixed MLP boundary."""
from __future__ import annotations

from ref.reference import make_inputs, reference, validate_inputs, validate_reference


def _check_options(options: dict) -> None:
    if options["device"] != "a5" or options["backend"] != "cce":
        raise ValueError("gdn2_norm2_w12_swiglu declares only a5/cce")
    if options["block_dim"] != 28:
        raise ValueError("gdn2_norm2_w12_swiglu fixes block_dim=28")


def kernel_for(case: dict | None = None):
    del case
    from kernels.step import gdn2_norm2_w12_swiglu_kernel

    return gdn2_norm2_w12_swiglu_kernel


def execute(inputs: dict, options: dict) -> dict:
    import torch
    from _unit_runner import launch_kernel

    _check_options(options)
    validate_inputs(inputs)
    hidden = torch.full((1, 6208), float("nan"), dtype=torch.bfloat16)
    got = launch_kernel(
        kernel_for(),
        (inputs["x"], inputs["gamma"], inputs["paired_weight"], hidden),
        options,
    )
    return {"hidden": got}


__all__ = [
    "execute",
    "kernel_for",
    "make_inputs",
    "reference",
    "validate_inputs",
    "validate_reference",
]
