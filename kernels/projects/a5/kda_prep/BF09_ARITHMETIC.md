# BF-09 arithmetic changes

The production change is confined to three compensated-arithmetic helpers and
their bounded polynomial callsites in `kernels/backward.py`. The coefficients,
branch thresholds, raw domain, casts, work partition, buffers, synchronization
and fixed reduction order remain unchanged. The original frozen gradient
budgets and D-PM-54/55/56 interpretation apply to every new qualification run.

1. **Product residual.** With `p = RN(ah * bh)`, a fused FP32
   `muladddst(e, ah, bh)` starting at `e = -p` computes the product residual
   without mantissa splitting. For a finite normal product whose residual is
   representable, this is an error-free TwoProduct residual. Both DD cross
   products, their addition order and renormalization remain. Products involving
   unbounded gradients/decay retain the primary IEEE infinity and zero-low guard.
   Underflow/flush observations are disclosed under the existing endpoint rules;
   the normal-product argument does not qualify them as CPU-correct.
2. **Constant addition.** Both `_dd_add_constant` callers are Horner steps where
   the positive coefficient dominates the high product. The exp argument has
   absolute value below .313, including its low component; an absolute-series
   bound gives product/coefficient ratio below `.313/(1-.313/2) < .372`.
   In the log series the squared argument is below .112, giving ratio below
   `.112/(1-.112) < .127`. FP32 rounding is far below these strict margins.
   FastTwoSum therefore recovers the same exact addition residual with four
   fewer instructions per step. The low-component addition order is unchanged.
3. **Unreachable overflow guards.** Only the exp/log polynomial products pass
   the compile-time `protect_overflow=False` argument. Exp evaluates the same
   degree-14 polynomial at a clamped argument in approximately [-.3125, 0],
   followed by the same eight compensated squarings. Its high/low magnitudes
   remain conservatively below 2. The log argument is `e/(2+e)` with `e` in
   [0,1]; its square and Horner products remain below 1. Their primary and
   renormalized products cannot overflow. All arithmetic compensation still
   executes; only comparisons/selections that cannot repair these bounded
   values are omitted. Generic division and unbounded products keep the guards.

Static CCE emission for all 16 backward dtype entries succeeds. In the principal
gate contribution VF, the static vector-call count changes from 4,908 to 3,648
after the fused residual, to 3,376 after FastTwoSum, and to 2,784 after bounded
guard removal. These are source counts, not elapsed times or a throughput model.
The 32-row work item, seven UB buffers, 100,928/117,312 byte storage totals,
five-level partial reduction and global merge remain unchanged. No new event,
atomics, lookahead or host computation is introduced.

`backward_arithmetic.py` is a diagnostic wrapper around the actual production
helpers. Its runtime-generated binary32 inputs have independent exact FP64
product/sum references and require both high and residual bits to match. It
includes cancellation, small/large normal products, signed overflow protection,
bounded products and dominating-coefficient addition. A deliberately zeroed
residual must be rejected. These local diagnostics supplement the full native
training, grid and endpoint checks; they do not replace those checks.
