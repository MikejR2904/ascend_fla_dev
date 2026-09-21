# a2 gdn2_fused_recurrent (decode) — A2 (Ascend910B3/c220) port

First A2 kernel in this repo. Fully vectorized c220 tensor-vector rewrite of the
a5 @vf source (a5 register model is c310-only). Correct on real 910B silicon,
all 5 contract cases, block_dim=8 (16 heads on 16 vector participants).

## Performance (real_decode B1/T1/H16, card 7, in-process runtime bridge)
- torch_npu recurrence baseline: 4265.4 us/call
- this kernel, eager aclnn:       266.1 us/call   (16.0x)
- this kernel, NPU-Graph replay:   13.63 us/replay (312.9x) — host dispatch amortized
- correctness: o relative-L2 3.2e-6 (eager and after graph replay)

## Key techniques
- brcb-spread + full-tile grouped multiplies (_spread8/_rowscale/_outer): removes
  the 128-iter per-key loop AND fixes the 32B UB alignment fault (spreads read at offset 0).
- block_dim=8 head parallelism (GetVecNum work split), verified correct on silicon.
- Newton-refined rsqrt (hardware fast rsqrt ~2^-11 overshoots the 1e-4 gate).
- 64-lane groups = one vector repeat, so per-row broadcast via src rep/blk strides is exact.

## Gotchas found on c220 (for the chunk kernels)
- brcb consumes 8 source elements/repeat; unary ops misbehave on [8,8] tiles (do
  scalar math on [1,64] rows); [128,128] full-tile ops exceed the 255-repeat cap
  (split); strided [n,64] views need explicit rep-strides; block_dim sweeps must be
  per-process (one op, one build).
