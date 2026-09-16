"""Standalone ascriptor.kernel-unit/1 hooks for GDN-2 recurrent forward."""
from __future__ import annotations

from ref.reference import make_inputs, reference, validate_inputs, validate_reference


def _check_options(options: dict) -> None:
    if options["device"] != "a5" or options["backend"] != "cce":
        raise ValueError("gdn2_fused_recurrent declares only the a5/cce combination")
    if options["block_dim"] != 8:
        raise ValueError("gdn2_fused_recurrent fixes block_dim=8")


def kernel_for(case: dict | None = None):
    del case
    from kernels.step import gdn2_fused_recurrent_kernel

    return gdn2_fused_recurrent_kernel


def execute(inputs: dict, options: dict) -> dict:
    import torch
    from _unit_runner import launch_kernel

    _check_options(options)
    validate_inputs(inputs)
    batch, time, heads, _ = inputs["q"].shape
    output = torch.full(
        (batch, time, heads, 128), float("nan"), dtype=torch.float32
    )
    final_state = torch.full(
        (batch, heads, 128, 128), float("nan"), dtype=torch.float32
    )
    arguments = tuple(
        inputs[name]
        for name in ("q", "k", "v", "g", "erase_gate", "w", "initial_state")
    ) + (output, final_state, batch, time, heads)
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
