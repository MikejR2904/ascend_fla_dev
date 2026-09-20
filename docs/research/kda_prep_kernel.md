# KDA raw preprocessing: calibration before implementation

BF-07 moves enabled raw q/k normalization, gate transform and beta sigmoid into
a new `kernels/projects/a5/kda_prep` unit on A5. The selected source baseline is
`c09a80ed8dbb96fdde7642eee21bec00d8f38e3a`; three byte-preserved public wrappers
and their SHA256 identities are in `baseline/`. Existing stable forward,
backward and layout kernels remain read-only.

## Current evidence

CPU precision study only: **100 cases**, **66 corrupt-output controls rejected**,
Python3.11.15 / Torch2.10.0+cpu. No candidate kernel has been written, compiled,
simulated or executed on a device. No native acceptance is claimed here.
FP64 is used only to measure preprocessing numerical error; the independent
and pinned FLA end-to-end KDA goldens remain **Torch CPU FP32**.

The seed7007 generator covers K128, B1/B2, C1/C3 and full Kimi B1/T4096/H32,
all four norm input/output dtype combinations and eight independent raw gate
input/A_log/dt_bias dtype tuples. It includes zero/near-zero/large rows,
softplus threshold20 neighbors, sigmoid saturation, exponent over/underflow,
NaN/Inf and signed zero. Inputs are rounded to their declared BF16/FP32 dtype
before either reference path. Log/exp/sigmoid outputs stay FP32; normalized
q/k use BF16 for chunk or v.dtype for decode. Disabled flags preserve identity.

## Frozen ordinary finite numerical budgets

`budgets.json` is the machine-readable authority, fixed in this first commit
before kernel implementation. F is the maximum measured predecessor error
against independent FP64 math for each operation/dtype tuple. Relative-L2
budgets are3F (norm additionally capped at1e-2); elementwise relative budgets
are3 times that group's measured maximum relative error. There is **no additive
absolute tolerance**; zero references must match exactly. Against the old host
path we also report all differences, without replacing the FP64 comparison.

| Operation / dtype tuple | Relative-L2 floor F | Relative-L2 limit | Elementwise relative limit |
|---|---:|---:|---:|
| norm:bf16_bf16 | 0.00168262802843 | 0.00504788408528 | 0.0116729698285 |
| norm:bf16_f32 | 6.29763341467e-08 | 1.8892900244e-07 | 5.42259363867e-07 |
| norm:f32_bf16 | 0.00166844413381 | 0.00500533240143 | 0.0116731448133 |
| norm:f32_f32 | 5.3027837224e-08 | 1.59083511672e-07 | 5.29366160389e-07 |
| gate:bf16_bf16_bf16 | 4.61923600487e-08 | 1.38577080146e-07 | 5.10637936927e-07 |
| gate:bf16_bf16_f32 | 4.88042180112e-08 | 1.46412654034e-07 | 8.14464019558e-07 |
| gate:bf16_f32_bf16 | 4.33976365207e-08 | 1.30192909562e-07 | 5.83186411394e-07 |
| gate:bf16_f32_f32 | 4.57815534948e-08 | 1.37344660484e-07 | 7.29829239364e-07 |
| gate:f32_bf16_bf16 | 4.61968317111e-08 | 1.38590495133e-07 | 8.41324823171e-07 |
| gate:f32_bf16_f32 | 4.76932359888e-08 | 1.43079707967e-07 | 8.00884112086e-07 |
| gate:f32_f32_bf16 | 4.46002123875e-08 | 1.33800637163e-07 | 8.57793647117e-07 |
| gate:f32_f32_f32 | 4.64723427901e-08 | 1.3941702837e-07 | 7.16692450484e-07 |
| beta:bf16 | 3.89619498087e-08 | 1.16885849426e-07 | 3.54302567418e-07 |
| beta:f32 | 3.82180445751e-08 | 1.14654133725e-07 | 3.46300092717e-07 |

BF16 normalized outputs must also differ by at most1ULP from the correctly
rounded reference. The predecessor's FP32 normalized outputs themselves reach
2–3ULP against correctly rounded FP64 (BF16-input maximum3, FP32-input maximum2).
This was reported to PM **before implementation**. FP32 differences above1ULP
remain a review trigger requiring a located reduction/sqrt/division explanation;
these observations are not converted into an automatic2–3ULP allowance.

## Range and nonfinite semantics

These observations do not inflate ordinary budgets or remove input cases. The
native verifier must compare actual predecessor and candidate endpoint masks
and finite values, report FP64 ideal differences, and retain any discrepancy.
No new range gate or silently narrower input domain is authorized.

| Boundary | Predecessor FP32 behavior observed / required comparison |
|---|---|
| K128 zero / near-zero row | Additive epsilon1e-6; zero remains zero, small rows divide by approximately1e-3; compare signed zero |
| x squared or summed squares overflow | At scale1e20 the predecessor returns finite zero norm outputs although FP64 ideal is nonzero; this is a retained FP32 semantic endpoint, not accuracy against FP64 |
| exp(A_log) overflow | Can overflow before multiplication even where ideal FP64 final value would fit; compare NaN/Inf masks, do not reassociate or clip |
| exp(A_log) underflow | Subnormal/zero multiplier; report quantization and signed-zero consequences separately |
| softplus threshold20 | Strict u>20 branch returns u; its derivative is exactly1 on that branch, not an unconditionally evaluated sigmoid |
| large negative softplus | Preserve/report subnormal and zero behavior; no ln(1+exp(u)) cancellation shortcut justified by this CPU study |
| sigmoid saturation | FP32 rounds some large positive outputs to1 and negative outputs to subnormal/zero; no broad relative-error budget from those endpoints |
| NaN / Inf inputs | Compare propagation masks and finite siblings; a comparison over an empty finite subset never counts as numeric success |

End-to-end limits stay o/state/dq/dv/dbeta/dh0=.05, dk=.15, dg=.25 relative-L2
against both CPU FP32 references, with existing elementwise checks. Gate limits
stay forward155/backward105 and require fresh tests at0,100.8,105,155. Block
sizes must produce identical bytes. D-PM-42 arithmetic exceptions and D-PM-44
inherited scan uncertainty retain their original scope.

## Reproduce

From the selected accepted CPU environment, with this unit directory as cwd:

```bash
python ref/calibrate.py --output evidence/calibration
python verify_calibration.py
```

The standalone study imports only Torch and the verified predecessor snapshot;
it does not import candidate kernel code or access devices/network. Text raw
records are `evidence/calibration/pre-kernel.json`; summary and all-file hashes
are adjacent. Inputs and references regenerate at run time from the seed.
Stage2, unless PM splits it into BF-08, must separately calibrate and freeze
its raw/parameter-gradient budgets before its kernel implementation.
