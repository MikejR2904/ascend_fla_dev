"""Standalone ascriptor.kernel-unit/1 hooks for fused GDN-2 BF16 decode."""
from __future__ import annotations

from ref.reference import make_inputs, reference, validate_inputs, validate_reference


def _check_options(options: dict) -> None:
    if options["device"] != "a5" or options["backend"] != "cce":
        raise ValueError("gdn2_fused_decode declares only the a5/cce combination")
    if options["block_dim"] != 8:
        raise ValueError("gdn2_fused_decode fixes block_dim=8")


def kernel_for(case: dict | None = None):
    del case
    from kernels.step import gdn2_fused_decode_kernel

    return gdn2_fused_decode_kernel


def execute(inputs: dict, options: dict) -> dict:
    import torch
    from _unit_runner import launch_kernel

    _check_options(options)
    validate_inputs(inputs)
    output = torch.full((1, 1, 16, 128), float("nan"), dtype=torch.bfloat16)
    final_state = torch.full((1, 16, 128, 128), float("nan"), dtype=torch.float32)
    arguments = tuple(
        inputs[name]
        for name in (
            "q",
            "k",
            "v",
            "f_raw",
            "b_raw",
            "w_raw",
            "output_gate",
            "decay_rate",
            "dt_bias",
            "norm_weight",
            "initial_state",
        )
    ) + (output, final_state)
    got_output, got_state = launch_kernel(kernel_for(), arguments, options)
    return {"o": got_output, "final_state": got_state}


__all__ = [
    "execute",
    "kernel_for",
    "make_inputs",
    "reference",
    "validate_inputs",
    "validate_reference",
]
