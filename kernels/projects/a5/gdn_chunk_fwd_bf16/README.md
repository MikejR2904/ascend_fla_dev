# Native BF16/FP32 grouped GDN forward

Task BF-01. Five CCE Vector launches, direct typed GM inputs/output, FP32
intermediates/state and kernel-side consecutive head indexing. No host casts
or group copies. Zero initial state, raw q/k, fixed scale, K=V128, T multiple64
up to4096, a5 block_dim1/2. See contract.json and the research document.

The canonical unit cases use BF16 input/output storage and compare against
FP32 references; FP32 entries are the same derived implementation's compatibility
path and receive a separate native full grid plus byte comparison with the
unchanged old unit. All support rows remain untested until actually executed.

Run `python run.py reference --output <ignored-output>` for CPU references,
`python run.py check --launcher board --output <ignored-output>` for the explicit
canonical board launcher, or use the task native benchmark once available.
The library's accepted canonical runner is copied unchanged with its hash receipt.

Inputs and references are generated at runtime. B uses an independent FP32
triangular solve. A loads the exact pinned FLA naive using FLA_GDN_NAIVE, checked
against its recorded SHA256. Calibration and fixed floors precede kernel source.
NaN poison belongs to the test launcher, not the production public call.
