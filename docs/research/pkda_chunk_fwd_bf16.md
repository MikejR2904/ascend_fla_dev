# PKDA native BF16 forward: precision contract

## Implementation decision, fixed before kernel authoring

BF-04 / D-PM-35. The study below is CPU research only, not hardware qualification. Source: FLA v0.6 `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`, naive SHA256 `ec18fab93598982c41447b8e0b3b3e3e1916b71945b131a4557033646ce187c5`; Ascriptor library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`, kernels `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`.

The candidate consumes BF16 q/k/v in GM and writes BF16 o directly. Prepare widens inputs inside its VF. ATK recurrence, compensated gate prefixes, scores, ordered WY solve and recurrent scan remain FP32. Only the final output stage rounds its two matrix-product operand pairs to BF16, uses BF16 cube products with FP32 accumulation, and rounds o to BF16 inside the kernel. Final main and ATK states remain FP32. Raw q/k are never normalized by the operator. Existing FP32 kernels and behavior are unchanged.

## Fixed numerical budgets

For each case and each independent FP32 reference, calibrate F as relative L2 between that reference output and its BF16 rounding. For o and final_state, require relative L2 <= 1e-2 and <= 3F against BOTH A (pinned naive, already BF16-rounded inputs) and B (independent chunk reference). The verifier uses the stricter floor across A/B. final_A_state requires relative L2 <= 1e-4. Record max_abs as well. No tolerance relaxation, case omission or gate-domain narrowing is authorized.

## CPU operand experiment

24 cases: the original 16 PKDA contract cases, uniform chunk gate spans 0, 1e-4, 50, 105, 155 at T192/H2, and three T192/H8 strong/weak adversaries (first log decay -154.9992, remaining -1e-5 per chunk). Includes T4096. Inputs are generated at runtime. BF16 key generation uses 0.99 times a normalized FP32 random row before BF16 rounding to remain in the declared norm domain; this is test generation only. Exact norm boundaries are separate native acceptance cases.

F ranges: o = 0.0015329825691878796..0.0017117612296715379; final_state = 0.001643937430344522..0.001672862796112895. These imply approximately 0.0046..0.0051 effective budgets, stricter than 1e-2. Torch CPU 2.10.0+cpu.

| Variant | Failed cases | max o relL2 vs A | max final_state | max final_A_state |
|---|---:|---:|---:|---:|
| fp32_internal | 0/24 | 0.00171176123 | 6.629890777e-06 | 0 |
| scores_bf16 | 0/24 | 0.003063199576 | 0.0009926152416 | 0 |
| wy_bf16 | 0/24 | 0.00251167221 | 0.001974400831 | 0 |
| scan_bf16 | 0/24 | 0.002807851648 | 0.003231452312 | 0 |
| output_bf16 | 0/24 | 0.003426218871 | 6.629890777e-06 | 0 |
| all_matmul_bf16 | 0/24 | 0.004453872796 | 0.003699050518 | 0 |
| atk_state_bf16 | 23/24 | 0.001712582307 | 8.006107964e-05 | 0.003351417603 |
| prefix_bf16 | 13/24 | 0.03080379777 | 0.05543781072 | 0 |

Maximum per-stage relative L2 against independent FP32 stage results:

| Variant | qn | kn | gc | bk | wv | lower | score | u | wy | states | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32_internal | 0 | 3.2199e-08 | 0 | 0 | 0 | 7.11315e-08 | 2.2469e-07 | 6.23628e-08 | 6.56437e-08 | 1.57559e-07 | 1.21958e-07 |
| scores_bf16 | 0 | 3.2199e-08 | 0 | 0 | 0 | 0.00245396 | 0.0374161 | 0.000854206 | 0.000892842 | 0.000911665 | 0.000886004 |
| wy_bf16 | 0 | 3.2199e-08 | 0 | 0 | 0 | 7.11315e-08 | 2.2469e-07 | 0.00178959 | 0.00203039 | 0.00175662 | 0.00193208 |
| scan_bf16 | 0 | 3.2199e-08 | 0 | 0 | 0 | 7.11315e-08 | 2.2469e-07 | 6.23628e-08 | 6.56437e-08 | 0.00275807 | 0.00142232 |
| output_bf16 | 0 | 3.2199e-08 | 0 | 0 | 0 | 7.11315e-08 | 2.2469e-07 | 6.23628e-08 | 6.56437e-08 | 1.57559e-07 | 1.21958e-07 |
| all_matmul_bf16 | 0 | 3.2199e-08 | 0 | 0 | 0 | 0.00245396 | 0.0374161 | 0.00197666 | 0.00198953 | 0.00320892 | 0.00250723 |
| atk_state_bf16 | 0 | 6.54841e-05 | 0 | 0 | 0 | 5.95497e-05 | 0.000548919 | 2.04638e-05 | 2.11863e-05 | 5.93571e-05 | 3.02635e-05 |
| prefix_bf16 | 0 | 3.2199e-08 | 0.00174861 | 0 | 0 | 0.333386 | 0.0328171 | 0.00205862 | 0.00116737 | 0.0207564 | 0.00205864 |

ATK-state BF16 storage fails 23/24 cases; gate-prefix BF16 storage fails 13/24. These are rejected precision variants, not failures of a deployed kernel. They motivate retaining FP32 internal state/prefix as explicitly permitted by D-PM-35. Every tested matrix-product variant passed, but only output-stage products are selected for this implementation: they yield actual BF16 cube work with ample accuracy headroom and leave sensitive recurrence unchanged.

## Native design and acceptance pending

The new unit has a device-side numeric guard/default initializer followed by five mathematical stages. Host performs metadata checks, allocation, pointer/launch operations and scalar control-status readback only. It performs no tensor arithmetic or dtype conversion. Native TorchDispatchMode traces are required before qualification.

Output tiles are 64 query rows by 128 value channels; each cube owns a complete batch/chunk/head tile, its two vector participants own 32 query rows each. Four independent single-slot L1 operands (64x128,128x128,64x64,64x128 BF16: 72 KiB total), one 64x128 FP32 L0C (32 KiB), and five guarded mutexes. VC ownership persists through last cube/FIX consumption; CV ownership persists through last VF read. Each iteration balances one lock/ready/wait/free per handoff; depth one, no lookahead, no speculative drain. L0C transfers FP32 via SPLITM; VF performs BF16 output conversion. No direct mixed-dtype SPLITM.

Full native workload must precede reduced simulator diagnostics. All required vendors compile before any custom launch, with independent processes for bd1/2/3/4. Required native evidence: original16 cases, tails, H32/B2, independent optional states and continuation, gate endpoints and adversaries, invalid-domain guards, poisoned output coverage, input immutability, cross-bd byte equality, host operator audit, and pre/post FP32 byte equality. Performance is three same-card rounds of existing FP32 / BF16 candidate / existing FP32 at T1024 and T4096, without a speed threshold.

Current qualification is in progress: the selected candidate passed CPU precision research, source emission, bd4 vendor compilation and the bd4 native cases below. Remaining block dimensions and performance are not yet qualified. Historical PK-04 results confer no BF-04 qualification.

## D-PM-37: both dtype paths audited

The original FP32 path has no dtype/layout conversion, but its numerical validator invokes Torch arithmetic. PM clarification [issue103comment5743858594](https://github.com/ddddwee1/ascend_fla_dev/issues/103#issuecomment-5743858594) permits read-only numerical validation, constant-filled allocation, and control-status readback as separately recorded audit categories. Accordingly the FP32 wrapper retains its original validation/default behavior and five original mathematical kernels. The final verifier records the validation phase explicitly and rejects conversion/copy/computational arithmetic elsewhere. BF16 numeric checks and defaults remain device-side; its only host tensor operators are empty/view, with a separately recorded ACL control-code readback. Eleven vendors (six BF16 plus five original FP32) must compile before first custom execution.

The CPU operand experiment is reproducible with `python ref/precision_study.py --output <ignored-output>/precision.json` from an accepted CPU environment and the pinned `FLA_PKDA_NAIVE` source. The delivered runner reproduces every stage metric in the original pre-implementation 24-case report exactly; both reports are retained under `evidence/`.

Initial source emission succeeds for all six BF16 kernels. Output ND-to-NZ transfers carry a performance advisory: four BF16 operand publications use multiple MTE3 bursts, with bounded legal footprints and explicit ownership. The first candidate keeps this sanctioned transfer; no speed claim is made before measurements. The emitted output cube uses local mutex IDs0..9, below the fixed32 limit, and explicit logical cross-side IDs0..4. UB usage is148KiB, L1 is72KiB and L0C is32KiB.

## Preliminary native evidence

The original T4096/H8 workload ran before reduced models. bd4 passed89 full-chain cases: original16, all64 tails T65..128, B2/T130/H32, five uniform spans0/1e-4/50/105/155, and three strong/weak adversaries. Another43 API/before-after checks passed. Maximum relative L2 against pinned naive: o0.003426211886, main state6.61480999e-6, ATK state1.052942977e-7; independent reference and case-calibrated3F also passed. Inputs were unchanged and all19 allocated stage/status buffers fully overwrote NaN poison. Public/staged outputs matched bytes. These are BF-04 executions, not inherited PK-04 claims.

After the full native run, the output-only T129/H1/bd1 diagnostic passed functional and pipe models: three ownership iterations including a tail, balanced events, no hazards/deadlock. Model cycles are not hardware performance. PKDA-related CPU tests passed83 cases on both accepted CPU and native-host Python environments. Final audit categorization after the PM clarification, remaining block dimensions, cross-bd bytes, performance, final reports and archive restoration are pending.
