# A2-01：ascriptor 在 A2 上的 split-K FP32 cube 缺陷定性

> **适用范围**：本文件的每个真机数字只对 **Ascend 910B3（a2）/ CANN 9.0.0 / 内置算子包 `ascend910b`、`ascend910_93` /
> ascriptor library `90cfcdc`、kernels `b3b3f9c`** 成立。按 `AGENTS.md` §2，A2-11 之前 A2 真机数字**只作观测记录，
> 不构成算子结论**，也不解除 A2 的总闸。不外推到 A3 / A5（`AGENTS.md` §6）。

## 0. 结论

1. **"split-K FP32 cube 缺陷"就是 ascriptor library 缺陷 M10-081**。c220 系（910B / 910_93）上，两次写同一 L0C 的短
   MMAD 之间硬件不互锁，后一次 `is_init=False` 的累加可能读到前一次尚未落定的累加器，结果静默出错。功能模型与
   管线模型都按程序序精确执行，所以只有真机能看到。
2. **pin 里已带修复，但范围很窄**：`matmul(..., splitk=S)` 展开时，只在 **A2 系且 A/B 两个操作数都是 FP32** 的情况下
   每个 MMAD 后插 `PIPE_M` barrier。手写的多次 `matmul`/`mmad` 累加链只有一条 lowered-IR lint（trap 通道，打到 stderr），
   不自动修。
3. **本机真机复现**：FP32 split-K（带修复）全部逐位正确，包括 M10-081 的原 case 与 KDA `intra.py` 的形状。
   **新发现：BF16 / FP16 split-K 在这台 910B3 上不安全。** M16 上全错：2、3、4、8 个 K 分片都错，几乎每个元素都不对，
   而且不报错。M32 上每次都触发 AI Core 异常（507015）。M64 上全对。
   修复规则按 dtype 把 BF16/FP16 排除在外，所以它们没有 settle。只在生成代码里补同一行 `PipeBarrier<PIPE_M>()`，
   M16 与 M32 就全部逐位；只插注释的对照照样错（§5.4）。
4. **25 个在用 kernel 的命中表**（§7）：
   - FP32 split-K 1 个，是 KDA scores `intra.py`，pin 已自动 settle。
   - FP32 手写累加链且未 settle 3 个：KDA inverse、GDN inverse、GDN bwd finalize。前两个正是 M16 形状，A2 派生单元必须补 barrier。
   - BF16 手写累加链 2 个：KDA recurrent、GDN bwd wu，都是 M64。真机探针在 M64 上没踩到，但不在任何规则保护之内。
   - 其余 19 个未命中。
5. **推荐**：A2 派生单元（A2-03）对所有累加链显式 `barrier(Pipe.M)`，不论 dtype，并保留 pin 的 FP32 规则。
   BF16/FP16 split-K 的规则缺口报给 ascriptor 所有者。修好之前，本仓 a2 路径不用 M<64 的 BF16/FP16 `splitk`，
   按 `AGENTS.md` §7 报错。详见 §8。

## 1. 缺陷原始记录（摘录，均在 ascriptor workspace 的 pin 修订内）

| 来源 | 内容 |
|---|---|
| library `docs/migration/a2-a3-handoff.md:3` | "Status: deferred by the maintainer on 2026-09-06 (D-250)"。0.1.0 只声明 A5 |
| 同上 `:64-97`「Repaired development candidate: split-K FP32 cube bias」 | 生成单元 `examples/api/a2_cube_bias` 十个 case 在 reference / 功能模型 / 管线模型全对；A3 真机前九个对，`f32_splitk_m16_n64_k128`（M16、N64、K128、split-K16、seed 103609、一个 cube）错，重试误差幅度不同。2026-09-15 在 A2、A3 上独立复现，"isolated it to the second short FP32 MMAD reading an unsettled L0C accumulator"，加 M 管线 settle 后原 case 逐位 |
| library `docs/defects/M10-081-a2-family-fp32-mmad-settle.md` | 910B3（`a2`，CANN 9.0 编 `ascend910b`）原 case 最大误差 49.26094102859497。M16/K16 一次 MMAD 正确，凡有第二个分片的 M16 都错，M32 全对。只在第二个 MMAD 后补 `PIPE_M` 就逐位。"The c220 hardware does not interlock this short same-pipe L0C RAW"。状态 fixed-candidate，21 case 边界扫描 A2、A3 各 21/21 |
| 同上，修复 | `ascriptor/passes/desugar.py` 对 A2 系 FP32 split-K 在每个 MMAD 后插 IR 级 `sync.barrier(M)`，"deliberately limited to the c220 device family and FP32; it does not widen unmeasured device families or dtypes"。`ascriptor/ir/lint.py` 对手写流报 A2 系正确性 lint |
| library `docs/migration/fragments/a2-family-fp32-mmad-settle-20260915.json` | 修复文件的 sha256：`desugar.py` = `900610ea…cc295`、`lint.py` = `c2e46626…a3b2e`。**本机导入的 pin 版这两个文件与之逐字相同**（每份回执的 `versions.ascriptor_*_sha256`） |
| library `docs/defects/M10-083-splitk-l0-ready-depth.md` | split-K 内部 L0 双槽 ready/valid 协议"reopened for an autosync redesign on 2026-09-15"。pin 里的配对双槽环（`passes/l0_lease.py:498-499`）"No current A3 model, CCE compilation or A2/A3 silicon result exists for this experiment" |

**为什么当初 defer**：维护者把 0.1.0 限定为 A5（D-250），A2/A3 连同未收口的缺陷整体移交后续任务。M10-081 修复本身
只作为 fixed-candidate，没有进入 A2/A3 的发布承诺。

## 2. "split-K" 在 ascriptor 里具体指什么

指的是 `matmul(dst, a, b, splitk=S)` 这个快捷方式的展开，见 library `ascriptor/passes/desugar.py:365-417`。
它不是把 K 维切给多个子核。它在**同一个 cube 核**上按 K 维每 `S` 个元素切一片，生成一个设备循环 `_subk`。
每片先把 A、B 的 K 窗口从 L1 搬到 L0A/L0B 双槽（MTE1），再发一个 MMAD 写**同一个** L0C：第一片 `is_init=True`，
其余 `is_init=False` 累加。

修复在 `desugar.py:404-413`：

```python
if self.ctx.device.family == "a2" and dta.name == dtb.name == "f32":
    self.make("sync.barrier", (), attrs={"pipe": Ident("M")},
              note="a2 family: settle the fp32 split-K L0C accumulator between MMADs")
```

生成的 CCE 里是循环体内 MMAD 分支之后、计数器递增之前的一行 `PipeBarrier<PIPE_M>();`。

同类构造还有第二种形态，即**手写累加链**：对同一个 L0C 连续调用多次 `matmul`/`mmad`，后面的 `is_init=False`。
这种形态不经过 split-K 展开，所以不会自动插 barrier。只有 `ascriptor/ir/lint.py:409-424` 的
`_a2_family_unsettled_fp32_mmad` 在 lowered IR 上报 trap，而且它只认 A、B、L0C 全是 FP32 的链，
设备限 `b1-b4`/`a3`（`lint.py:619`）。

## 3. a2 profile 与真机核数

| 项 | ascriptor 声明 | 来源 | 本机真机 |
|---|---|---|---|
| facade → 设备 | `a2` → `b3` | library `ascriptor/devices/__init__.py:18` | — |
| 家族 / 指令集 | `family: a2`、`arch: c220`、`npu_arch: dav-2201`、`compile_unit: ascend910b` | `ascriptor/devices/profiles/b3.json:3-6` | aclnn 构建日志 `ASCEND_COMPUTE_UNIT=ascend910b` |
| cube / vec 核数 | 20 / 40 | `b3.json:8-9` | `torch_npu` 报 `cube_core_num=20`、`vector_core_num=40`（`evidence/device/device_props.log:3`）|
| L1 / L0A / L0B / L0C / BT / UB（KB） | 512 / 64 / 64 / 128 / 0.5 / 192 | `b3.json:11-18` | 本任务未单独测 |
| SoC 名 | `debug_chipset: Ascend910B3` | `b3.json:7` | `Ascend910B3` |

**与 A5 的区别**：A5 的 `950` profile 声明 32/64，而 Ascend950PR 物理是 28/56，`block_dim` 超出会死锁（`AGENTS.md` §5）。
**在这台 910B3 上，profile 声明的 20/40 与 `torch_npu` 报的物理核数一致。**
这条只说明核数一致，`block_dim` 的实际上限与多核行为本任务没有测，那是 A2-10 的事。

## 4. 复现脚本

`benchmarks/a2/repro_splitk_fp32.py`，逐 case 生成极小的 cube kernel，对**独立的 CPU float64 参考**逐位比对。

- 输入全是有界二进分数：操作数取 [-1,1] 里的八分之一步长，bias 取四分之一步长。这样 FP32、BF16、FP16 输入下
  乘积与每个部分和在 FP32 里都精确，比较可以是**逐位**的，任何偏差都是错，不是舍入。
- 输出预填 -777，没写到的元素过不了比对。
- `--profile a2|a5` 选 facade，`--launcher reference|sim|pipesim|aclnn` 选执行路径。
- `aclnn` 用 ascriptor `OpExec` 以本机 CANN 编 aclnn 自定义算子，在 logical device 0 上跑，只接受 a2/a3 profile。
- `--hit-table` 扫 25 个 kernel，`--patch-generated sham|settle` 是 §5.4 的诊断实验。

```bash
# 主机侧（任何机器）
PYTHONPATH=<ascriptor>/library python benchmarks/a2/repro_splitk_fp32.py --profile a2 --launcher sim
PYTHONPATH=<ascriptor>/library python benchmarks/a2/repro_splitk_fp32.py --profile a5 --launcher pipesim
ASCRIPTOR_WORKSPACE=<ascriptor> PYTHONPATH=<ascriptor>/library python benchmarks/a2/repro_splitk_fp32.py --hit-table
# a2 真机（卡号由 ASCEND_RT_VISIBLE_DEVICES 选，脚本总用 logical device 0）
ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<ascriptor>/library \
  python benchmarks/a2/repro_splitk_fp32.py --profile a2 --launcher aclnn --repeat 5
```

case 分三类，`--list` 可以打印全部：

| 类 | case | 期望 |
|---|---|---|
| FP32 split-K | `splitk_f32_m16_n64_k128_s16_bias` 是 M10-081 原 case。另有无 bias 版、`k32` 隔离形状、`m64_n64_k128_s64` 即 KDA `intra.py` 形状 | 逐位（pin 已 settle） |
| BF16/FP16 split-K | `splitk_{bf16,f16}_m{16,32,64}_n64_k{32,48,64,128}_s16`，覆盖 2~8 个 K 分片 | 探针（规则不覆盖） |
| 手写累加链 | `chain_{f32,bf16,f16}_m{16,32,64}_n16_k16_t{2,3}_{nobar,bar}`，其中 FP32 M16 即 KDA/GDN inverse 的构造。`chain_*_m16_n64_k16_t8_*` 是 split-K 的手写孪生，8 次 K16 MMAD 写一个 [16,64] L0C。`t1` 是单 MMAD 对照 | `bar` 与 `t1` 逐位，`nobar` 为探针 |

## 5. 结果

### 5.1 主机模型：a2 与 a5 profile 对照（全部逐位，max_abs_diff 0，rel_l2 0）

| profile | launcher | cases | bitwise | max_abs_diff (max over cases) | pipesim hazards (sum) | deadlocks |
|---|---|---|---|---|---|---|
| a2 | sim | 73 | 73 | 0 | - | - |
| a2 | pipesim | 73 | 73 | 0 | 0 | 0 |
| a5 | sim | 72 | 72 | 0 | - | - |
| a5 | pipesim | 72 | 72 | 0 | 0 | 0 |

同一 kernel 在两个 profile 下的 lowered IR 事实（功能模型结果同上，全部逐位）：

| case | a2: M barriers (settle) / trap lints | a5: M barriers (settle) / trap lints |
|---|---|---|
| `chain_f32_m16_n16_k16_t2_bar` | 1 (0) / 0 | 1 (0) / 0 |
| `chain_f32_m16_n16_k16_t2_nobar` | 0 (0) / 1 | 0 (0) / 0 |
| `splitk_bf16_m16_n64_k128_s16` | 0 (0) / 0 | 0 (0) / 0 |
| `splitk_f32_m16_n64_k128_s16` | 1 (1) / 0 | 0 (0) / 0 |
| `splitk_f32_m16_n64_k128_s16_bias` | 1 (1) / 0 | n/a |
| `splitk_f32_m16_n64_k32_s16` | 1 (1) / 0 | 0 (0) / 0 |
| `splitk_f32_m64_n64_k128_s64` | 1 (1) / 0 | 0 (0) / 0 |

### 5.2 为什么 sim / pipesim 复现不出来

**功能模拟器**按程序序逐条执行 Surface IR，MMAD 之间没有时间，后一次累加读到的永远是已完成的 L0C。
**pipesim** 建模的是事件、flag 与跨管线的冒险（MTE1↔M、M↔FIX 等）。**同一条 M 管线内**两次 MMAD 的
L0C 写回延迟不在它的模型里，所以它报 0 hazard。上表里 a2 与 a5 的全部 case 都是这样，包括真机上出错的那些。

这与 M10-081 的记录一致："Its independent reference, functional model and pipe model are bitwise"。
缺陷只在 CANN 编译后的真机执行中出现。所以复现只能走 `--launcher aclnn`（本任务）或 board，这也是 A2-11 要在真机上做的事。

### 5.3 a2 真机（aclnn，block_dim=1，每 case 5 次独立进程运行）

三张同型号的空闲卡，记作 A、B、C。每张卡用前查过 Health=OK 且无进程，运行时持锁。表中数字都取 5 次里的最大值。
"distinct outputs" 是 5 次输出的不同哈希数：1 表示 5 次完全相同。

**split-K**，每行一次作业。variant 为 `as generated` 表示未改动的生成代码；`sham` / `settle` 见 §5.4。AI Core 异常的那一行在第 1 次运行就中止，后续不再运行：

| case | dtype | M,N,K | form | card | variant | bitwise runs | max_abs_diff | rel_l2 (max) | mismatched (max) | distinct outputs |
|---|---|---|---|---|---|---|---|---|---|---|
| `splitk_bf16_m16_n64_k128_s16` | bf16 | 16,64,128 | splitk=16 | A | as generated | 0/5 | 15.6719 | 1.34567 | 1024/1024 | 1 |
| `splitk_f16_m16_n64_k128_s16` | f16 | 16,64,128 | splitk=16 | A | as generated | 0/5 | 15.875 | 1.17102 | 1022/1024 | 1 |
| `splitk_f32_m16_n64_k128_s16` | f32 | 16,64,128 | splitk=16 | A | as generated | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_f32_m16_n64_k128_s16_bias` | f32 | 16,64,128 | splitk=16 +bias | A | as generated | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_f32_m16_n64_k32_s16` | f32 | 16,64,32 | splitk=16 | A | as generated | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_f32_m64_n64_k128_s64` | f32 | 64,64,128 | splitk=64 | A | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_bf16_m16_n64_k128_s16` | bf16 | 16,64,128 | splitk=16 | B | as generated | 0/5 | 39.3845 | 5.67027 | 1024/1024 | 5 |
| `splitk_f16_m16_n64_k128_s16` | f16 | 16,64,128 | splitk=16 | B | as generated | 0/5 | 40.9742 | 5.61071 | 1024/1024 | 5 |
| `splitk_bf16_m16_n64_k32_s16` | bf16 | 16,64,32 | splitk=16 | B | as generated | 0/5 | 15.6719 | 2.1991 | 1022/1024 | 1 |
| `splitk_bf16_m16_n64_k48_s16` | bf16 | 16,64,48 | splitk=16 | B | as generated | 0/5 | 15.5625 | 1.70332 | 1024/1024 | 1 |
| `splitk_bf16_m16_n64_k64_s16` | bf16 | 16,64,64 | splitk=16 | B | as generated | 0/5 | 13.8906 | 1.59758 | 1021/1024 | 1 |
| `splitk_bf16_m64_n64_k128_s16` | bf16 | 64,64,128 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_bf16_m64_n64_k32_s16` | bf16 | 64,64,32 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_bf16_m64_n64_k48_s16` | bf16 | 64,64,48 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_bf16_m64_n64_k64_s16` | bf16 | 64,64,64 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_f16_m16_n64_k32_s16` | f16 | 16,64,32 | splitk=16 | B | as generated | 0/5 | 9.09375 | 1.29482 | 1022/1024 | 1 |
| `splitk_f16_m16_n64_k48_s16` | f16 | 16,64,48 | splitk=16 | B | as generated | 0/5 | 9.46875 | 1.12049 | 1021/1024 | 1 |
| `splitk_f16_m16_n64_k64_s16` | f16 | 16,64,64 | splitk=16 | B | as generated | 0/5 | 10.5 | 1.1762 | 1022/1024 | 1 |
| `splitk_f16_m64_n64_k128_s16` | f16 | 64,64,128 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_f16_m64_n64_k32_s16` | f16 | 64,64,32 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_f16_m64_n64_k48_s16` | f16 | 64,64,48 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_f16_m64_n64_k64_s16` | f16 | 64,64,64 | splitk=16 | B | as generated | 5/5 | 0 | 0 | 0/4096 | 1 |
| `splitk_bf16_m32_n64_k128_s16` | bf16 | 32,64,128 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_bf16_m32_n64_k32_s16` | bf16 | 32,64,32 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_bf16_m32_n64_k48_s16` | bf16 | 32,64,48 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_bf16_m32_n64_k64_s16` | bf16 | 32,64,64 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_f16_m32_n64_k128_s16` | f16 | 32,64,128 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_f16_m32_n64_k32_s16` | f16 | 32,64,32 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_f16_m32_n64_k48_s16` | f16 | 32,64,48 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_f16_m32_n64_k64_s16` | f16 | 32,64,64 | splitk=16 | B | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_bf16_m32_n64_k128_s16` | bf16 | 32,64,128 | splitk=16 | C | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_bf16_m32_n64_k32_s16` | bf16 | 32,64,32 | splitk=16 | C | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_f16_m32_n64_k32_s16` | f16 | 32,64,32 | splitk=16 | C | as generated | 0/0 (第 1 次即 AI Core 异常 507015) | — | — | — | — |
| `splitk_bf16_m16_n64_k128_s16` | bf16 | 16,64,128 | splitk=16 | C | sham | 0/5 | 26.5405 | 2.0201 | 1024/1024 | 5 |
| `splitk_bf16_m16_n64_k32_s16` | bf16 | 16,64,32 | splitk=16 | C | sham | 0/5 | 5.23447 | 0.743402 | 1024/1024 | 5 |
| `splitk_f16_m16_n64_k128_s16` | f16 | 16,64,128 | splitk=16 | C | sham | 0/5 | 9.34053 | 0.699424 | 1024/1024 | 5 |
| `splitk_f16_m16_n64_k32_s16` | f16 | 16,64,32 | splitk=16 | C | sham | 0/5 | 4.26158 | 0.725587 | 1024/1024 | 5 |
| `splitk_bf16_m16_n64_k128_s16` | bf16 | 16,64,128 | splitk=16 | C | settle | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_bf16_m16_n64_k32_s16` | bf16 | 16,64,32 | splitk=16 | C | settle | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_f16_m16_n64_k128_s16` | f16 | 16,64,128 | splitk=16 | C | settle | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_f16_m16_n64_k32_s16` | f16 | 16,64,32 | splitk=16 | C | settle | 5/5 | 0 | 0 | 0/1024 | 1 |
| `splitk_bf16_m32_n64_k32_s16` | bf16 | 32,64,32 | splitk=16 | C | settle | 2/2 | 0 | 0 | 0/2048 | 1 |
| `splitk_f16_m32_n64_k32_s16` | f16 | 32,64,32 | splitk=16 | C | settle | 2/2 | 0 | 0 | 0/2048 | 1 |

**手写累加链**：每格是 5 次里逐位正确的次数，所有格 max_abs_diff 都是 0。N=16 的行在卡 A，`16,64` 这一行（split-K 的手写孪生）在卡 B：

| dtype | M,N | t1 | t2_nobar | t2_bar | t3_nobar | t3_bar | t8_nobar | t8_bar |
|---|---|---|---|---|---|---|---|---|
| f32 | 16,16 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| f32 | 32,16 | — | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| f32 | 64,16 | — | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| f32 | 16,64 | — | — | — | — | — | 5/5 | 5/5 |
| bf16 | 16,16 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| bf16 | 32,16 | — | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| bf16 | 64,16 | — | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| bf16 | 16,64 | — | — | — | — | — | 5/5 | 5/5 |
| f16 | 16,16 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| f16 | 32,16 | — | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| f16 | 64,16 | — | 5/5 | 5/5 | 5/5 | 5/5 | — | — |
| f16 | 16,64 | — | — | — | — | — | 5/5 | 5/5 |

读法：

- **FP32 split-K 全部逐位**，包括 M10-081 原 case、隔离形状与 KDA `intra.py` 形状。pin 的 settle 规则在这台卡上有效，
  本机导入的 `desugar.py` 就是回执里那一版。
- **BF16/FP16 split-K：M16 全错，M32 每次触发 AI Core 异常，M64 全对。** 两种失效都在卡 B、C 上复现，M16 在卡 A 上也复现。
  M32 的异常见 `evidence/device/card-*_m32_bf16_k32_harness_run.log`（`aclrtSynchronizeStream … 507015`）
  与 `m32_bf16_k32_cann_runtime_errors.log`（`retCode=0x26, [aicore exception]`）。
  2 个 K 分片（K32）也错，而 K32 走的是普通 autosync 事件，不是
  M10-083 的配对双槽环（生成代码见 `evidence/generated/`），所以错误不依赖那个环。
- **错误不是"丢了某个分片"**：把错误输出最小二乘分解到 8 个分片乘积上，残差约 0.99。分解到 64 个交叉分片乘积
  A_i·B_j^T 上，残差仍约 0.97。输出范数约为参考的 5 倍（`evidence/device/cross_fragment_fit.log`）。
  所以错误输出不是本 kernel 操作数的任何组合，符合"读到未落定或未写入的数据"，与 M10-081 的现象同型：
  短 MMAD、第二个分片起错、M32/M64 不错。
- **手写孪生不出错**：`chain_{bf16,f16,f32}_m16_n64_k16_t8_nobar` 与 split-K 算同一件事（8 次 K16 MMAD 写同一 L0C、不加 barrier），
  真机 5/5 逐位。区别在发射节奏：split-K 循环里下一片的 L0 装载与上一片 MMAD 重叠，MMAD 背靠背发出；
  手写链每次 `matmul` 之间有完整的 MTE1→M 事件往返。这说明"没踩到"取决于时序，**不能当作手写链安全的证据**。
  lint 对 FP32 手写链照样报 trap。
- **同一张卡上两次结果可以不同**：卡 A 上 BF16 K128 五次输出完全相同，卡 B 上五次各不相同。错误的具体数值取决于时序和残留数据，
  只有"对/错"这个判定是稳定的。

### 5.4 对照实验：只在生成代码里补一行 `PipeBarrier<PIPE_M>()`

`--patch-generated settle` 在本次运行的**生成 CCE 源码**里、每个 split-K MMAD 分支之后、计数器递增之前插入一行
`PipeBarrier<PIPE_M>();`。位置与 pin 的 FP32 规则生成的完全一致，ascriptor checkout 不动。
`--patch-generated sham` 在同一位置只插一行注释，用来控制"改生成代码再重编"这条路径本身的影响。结果见 §5.3 表里 `variant` 为
`sham` / `settle` 的行。

结果：

- **settle 版**：BF16/FP16 的 M16（K32、K128）5/5 逐位。M32（K32）2/2 逐位，而且不再触发 AI Core 异常。
- **sham 版**：M16 的四个 case 全部 0/5 出错。

两版只差那一行 `PipeBarrier<PIPE_M>()`，`evidence/generated/` 下有 settle 前后的生成源码，可以直接 diff。由此，BF16/FP16 split-K 在 M16 出错、在 M32 触发异常的直接原因是 **MMAD 之后缺少 M 管线 settle**，也就是 M10-081 的同一个构造。pin 的修复按 dtype 把 BF16/FP16 排除在外，所以它们没有被修到。

这个对照改的是一次运行的生成代码，只用于定位原因，不是修复（§10）。

## 6. 为什么 FP32 手写链在真机上也"没出错"

`chain_f32_m16_n16_k16_t2/t3_nobar` 是 KDA/GDN inverse 的构造，`chain_f32_m16_n64_k16_t8_nobar` 是 M10-081 原形状的手写版。
两者真机 5/5 逐位。原因同 §5.3：手写链的 MMAD 不是背靠背发出的。

这**不改变**命中表的判定。M10-081 记录的是硬件不互锁，库自己的 lint 对这些链报 trap，而真实 kernel 的发射节奏
（多核、流水交叠、更长的循环）与探针不同。按 `AGENTS.md` §6 的"时序没踩到不等于安全"，派生单元必须显式 settle。

## 7. 25 个 kernel 的命中表

由 `repro_splitk_fp32.py --hit-table` 生成，完整 JSON 见 `evidence/host/hit_table.json`。每行做两道独立检查：

- **AST**：列出 kernel 函数及其同文件 helper 里的每个 `matmul`/`mmad` 调用，含行号、A/B dtype（按 buffer 声明解析）、m/n/k、split、`is_init`。
- **IR**：在 pin 下按 a5 profile lower，与源码的 authoring 目标一致。之后找出所有"前面某条路径上有一次写同一 L0C 根的 MMAD、
  中间没有 M/ALL barrier"的 `is_init=False` MMAD。列 `lib fp32 rule` 表示 pin 自带的 `_a2_family_unsettled_fp32_mmad` 是否会对它报 trap。

判定：

- **HIT-A**：FP32 split-K。a2 上已由 pin 的 desugar 规则 settle。
- **HIT-B**：FP32 手写累加链，没有 settle。a2 上 lint 报 trap，但不自动修。
- **EXPOSED-C**：BF16/FP16 累加链，不在任何规则内。
- **MISS**：每个 MMAD 都初始化自己的 L0C（`splitn` 写的是不相交的子块），或累加前已有显式 M/ALL barrier，或根本没有 cube matmul。

"dispatched"是 `ascend_fla` 现在实际加载的 kernel：`ops/kda/chunk.py`、`chunk_bwd.py`、`fused_recurrent.py`，GDN 按上游单元的 `execute.py`。
"replaced"是被本仓派生单元取代的上游原版，一并扫描供 A2-03 参考。

| # | 组 | 阶段 | kernel | 文件 | 角色 | 累加 MMAD（IR 报出的行：A,B dtype；库 FP32 规则是否报） | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | KDA fwd | gate | `kda_sub1_gate_stable_kernel` | `kda_fwd_stable/kernels/gate.py` | dispatched | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 2 | KDA fwd | scores | `kda_sub2_score_stable_kernel` | `kda_fwd_stable/kernels/intra.py` | dispatched | matmul/mmad 调用 2 处，IR MMAD 4 个；246: f32,f32, 是; 247: f32,f32, 是 | HIT-A fp32 split-K (a2: settled by the pinned desugar rule) |
| 3 | KDA fwd | inverse | `tril_inverse64_v2_strict_bf16_kernel` | `kda_fwd/kernels/triangular_inverse.py` | dispatched | matmul/mmad 调用 11 处，IR MMAD 16 个；273: f32,f32, 是; 279: f32,f32, 是; 284: f32,f32, 是; 285: f32,f32, 是 | HIT-B fp32 accumulate chain, NOT settled (a2 lint trap) |
| 4 | KDA fwd | wy | `kda_sub3_wy_stable_kernel` | `kda_fwd_stable/kernels/wy.py` | dispatched | matmul/mmad 调用 2 处，IR MMAD 2 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 5 | KDA fwd | recurrent | `kda_sub45_aqk_repaired_kernel` | `kda_fwd_stable/kernels/recurrent.py` | dispatched | matmul/mmad 调用 4 处，IR MMAD 4 个；401: bf16,bf16, 否 | EXPOSED-C bf16 accumulate, outside the fp32 rule (device probe) |
| 6 | KDA bwd | scan_fused | `scan_fused_kernel` | `kda_bwd/kernels/scan_fused.py` | dispatched | matmul/mmad 调用 1 处，IR MMAD 8 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 7 | KDA bwd | inverse_mm | `inverse_mm_bounded_kernel` | `kda_bwd_stable/kernels/inverse_mm.py` | dispatched | matmul/mmad 调用 5 处，IR MMAD 5 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 8 | KDA bwd | inverse_epilogue | `inverse_epilogue_kernel` | `kda_bwd/kernels/inverse_epilogue.py` | dispatched | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 9 | KDA bwd | inverse_dainv | `inverse_dainv_kernel` | `kda_bwd/kernels/inverse_dainv.py` | dispatched | matmul/mmad 调用 2 处，IR MMAD 2 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 10 | KDA bwd | inverse_dakk_fused | `inverse_dakk_fused_kernel` | `kda_bwd/kernels/inverse_dakk_fused.py` | dispatched | matmul/mmad 调用 2 处，IR MMAD 2 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 11 | KDA bwd | finalize_pre | `finalize_pre_stable_kernel` | `kda_bwd_stable/kernels/finalize_pre.py` | dispatched | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 12 | KDA bwd | finalize_pair | `finalize_pair_kernel` | `kda_bwd/kernels/finalize_pair.py` | dispatched | matmul/mmad 调用 4 处，IR MMAD 4 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 13 | KDA bwd | finalize_post | `finalize_post_stable_kernel` | `kda_bwd_stable/kernels/finalize_post.py` | dispatched | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 14 | KDA bwd | finalize_reduce | `finalize_reduce_kernel` | `kda_bwd/kernels/finalize_reduce.py` | dispatched | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 15 | KDA decode | step | `kda_fused_recurrent_kernel` | `kda_fused_recurrent/kernels/step.py` | dispatched | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 16 | GDN fwd | preprocess | `gdn_preprocess_v2_kernel` | `gdn_fwd/kernels/preprocess.py` | dispatched | matmul/mmad 调用 2 处，IR MMAD 2 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 17 | GDN fwd | inverse | `tril_inverse64_v2_strict_bf16_kernel` | `gdn_fwd/kernels/inverse.py` | dispatched | matmul/mmad 调用 11 处，IR MMAD 16 个；306: f32,f32, 是; 312: f32,f32, 是; 317: f32,f32, 是; 318: f32,f32, 是 | HIT-B fp32 accumulate chain, NOT settled (a2 lint trap) |
| 18 | GDN fwd | recompute | `gdn_recompute_wu_v2_kernel` | `gdn_fwd/kernels/recompute.py` | dispatched | matmul/mmad 调用 2 处，IR MMAD 2 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 19 | GDN fwd | scores | `sub1_kernel` | `gdn_fwd/kernels/scores.py` | dispatched | matmul/mmad 调用 1 处，IR MMAD 1 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 20 | GDN fwd | recurrent | `gdn_recurrent_plain` | `gdn_fwd/kernels/recurrent.py` | dispatched | matmul/mmad 调用 4 处，IR MMAD 4 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 21 | GDN bwd | scan_local | `scan_local_bwd_kernel` | `gdn_bwd/kernels/scan_local.py` | dispatched | matmul/mmad 调用 3 处，IR MMAD 3 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 22 | GDN bwd | scan_state | `scan_state_bwd_kernel` | `gdn_bwd/kernels/scan_state.py` | dispatched | matmul/mmad 调用 1 处，IR MMAD 11 个；— | MISS every accumulate MMAD follows an explicit M/ALL barrier |
| 23 | GDN bwd | wu | `wu_bwd_kernel` | `gdn_bwd/kernels/wu.py` | dispatched | matmul/mmad 调用 4 处，IR MMAD 4 个；206: bf16,bf16, 否 | EXPOSED-C bf16 accumulate, outside the fp32 rule (device probe) |
| 24 | GDN bwd | inverse_preprocess | `inverse_preprocess_bwd_kernel` | `gdn_bwd/kernels/inverse_preprocess.py` | dispatched | matmul/mmad 调用 5 处，IR MMAD 5 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 25 | GDN bwd | finalize | `finalize_bwd_kernel` | `gdn_bwd/kernels/finalize.py` | dispatched | matmul/mmad 调用 7 处，IR MMAD 7 个；205: f32,f32, 是; 217: f32,f32, 是; 226: f32,f32, 是; 235: f32,f32, 是; 244: f32,f32, 是 | HIT-B fp32 accumulate chain, NOT settled (a2 lint trap) |
| 26 | GDN fwd | recurrent_saved | `gdn_recurrent_saved` | `gdn_fwd/kernels/recurrent_saved.py` | alternative | matmul/mmad 调用 4 处，IR MMAD 4 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 27 | KDA fwd | gate | `kda_sub1_gate_kernel` | `kda_fwd/kernels/gate.py` | replaced | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 28 | KDA fwd | scores | `kda_sub2_score_kernel` | `kda_fwd/kernels/intra.py` | replaced | matmul/mmad 调用 2 处，IR MMAD 4 个；179: f32,f32, 是; 180: f32,f32, 是 | HIT-A fp32 split-K (a2: settled by the pinned desugar rule) |
| 29 | KDA fwd | wy | `kda_sub3_wy_kernel` | `kda_fwd/kernels/wy.py` | replaced | matmul/mmad 调用 2 处，IR MMAD 2 个；— | MISS every MMAD initialises its L0C (no accumulate) |
| 30 | KDA fwd | recurrent | `kda_sub45_fused_kernel` | `kda_fwd/kernels/recurrent.py` | replaced | matmul/mmad 调用 4 处，IR MMAD 4 个；401: bf16,bf16, 否 | EXPOSED-C bf16 accumulate, outside the fp32 rule (device probe) |
| 31 | KDA bwd | inverse_mm | `inverse_mm_kernel` | `kda_bwd/kernels/inverse_mm.py` | replaced | matmul/mmad 调用 5 处，IR MMAD n/a 个；— | MISS every MMAD initialises its L0C (no accumulate); IR: PassError: local_mutex: cube needs 34 mutex IDs (maximum 32) at #18 loc("inverse_mm.py:65:15"); %_l0a: 2 slots; %_l0b: 2 slots; %l1_do: 2 slots; %l1_vnew: 2 slots; %l1_dv: 2 slots; %l1_h: 2 slots; %l1 |
| 32 | KDA bwd | finalize_pre | `finalize_pre_kernel` | `kda_bwd/kernels/finalize_pre.py` | replaced | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |
| 33 | KDA bwd | finalize_post | `finalize_post_kernel` | `kda_bwd/kernels/finalize_post.py` | replaced | matmul/mmad 调用 0 处，IR MMAD 0 个；— | MISS no cube matmul |

要点：

- 行 3、17（inverse）：FP32、M=N=K=16 的累加链，对应 `triangular_inverse.py:273/279/284-285` 与 `inverse.py:306/312/317-318`。
  它们正是 M10-081 的失效形状，而且是手写链，**pin 的修复够不到**。
- 行 25（GDN bwd finalize）：FP32 链，M 为 `rows_cube`（动态），行号 `finalize.py:205/217/226/235/244`。
- 行 5（KDA recurrent）与行 23（GDN bwd wu）：BF16 链，M=64（`recurrent.py:401`、`wu.py:206`）。
  本机探针在 M64 的 BF16 split-K 与 BF16 链上都没出错，但它们不在任何规则保护之内，时序也和探针不同。
- 行 22（GDN bwd scan_state）：`scan_state.py:578` 有一处累加，但 helper `_state_mmad` 在每次 MMAD 前后都 `bar_all()`，已经 settle。
- 行 31：上游原版 `inverse_mm_kernel` 在 pin 下 lower 失败（34 个 mutex ID 超过上限 32），这就是 D-PM-24 批准派生 bounded 版的原因。
  它的判定只来自 AST：5 个 `splitn` 调用，全部 `is_init=True`。

## 8. 绕行方案与验证计划

所有验证一律在 FP32 下判定，报 rel-L2 与 max_abs_diff，对两个独立参考。A2 上的数只作观测，直到 A2-11。

| # | 方案 | 适用 | 代价 / 风险 | FP32 验证计划 |
|---|---|---|---|---|
| W1 | 保留 pin 的 FP32 split-K settle 规则，不动 | HIT-A（KDA scores `intra.py`） | 无新改动。每个分片多一次 M 管线等待 | ① 本脚本 `splitk_f32_*` 真机逐位（已做）。② A2-03 派生 intra 单元在真实形状上对 CPU FP32 参考逐位或 rel-L2，`bd1` 与 `bd4` 逐位相同。③ lowered IR 里 settle barrier 数等于 split-K 循环数 |
| W2 | **派生单元对每条累加链显式 `barrier(Pipe.M)`**，放在每次产生 L0C 的 MMAD 之后、下一次 `is_init=False` 之前。这正是 lint 给出的修法 | HIT-B（KDA/GDN inverse、GDN bwd finalize）。按 W4 也用于 EXPOSED-C | 多一次 M 管线串行化，性能要测 | ① lint 0 trap。② `chain_*_bar` 真机逐位（已做）。③ 派生 inverse 单元：输出与 CPU FP32 逐位比对（输入同样用有界二进分数），并对真实形状测 `‖(I+A)·inv − I‖`。④ 奇偶 C × 多头的边界格（§6 的做法）。⑤ 同卡前后三明治测时延 |
| W3 | 把手写链改写成单个 `matmul(..., splitk=16)`（沿 K 拼接操作数），让 pin 的 FP32 规则覆盖 | HIT-B 中能拼 K 的情形 | 要重排 L1 布局，改变 kernel 结构，属于 kernel 批次 | 同 W2 的 ②③④。另加生成代码检查：每个 split-K 循环有 settle |
| W4 | BF16/FP16 累加：派生单元同样显式 settle，不依赖"真机没踩到" | EXPOSED-C（KDA recurrent `:401`、GDN bwd wu `:206`） | 同 W2 | ① `chain_bf16_*_bar` 真机逐位。② 派生单元真实形状 FP32 判定，bf16 只作质量检查。③ 与无 barrier 版在同形状下对比时延 |
| W5 | **BF16/FP16 split-K：库侧把 settle 规则从 FP32 扩到所有 A2 系 MMAD 操作数 dtype**。这是 ascriptor 所有者的修改，本仓不改 ascriptor。修好之前，本仓 a2 路径在 M<64 时不用 BF16/FP16 `splitk`，入口按 `AGENTS.md` §7 报错 | 目前 25 个 kernel 都不用 BF16/FP16 split-K，但 A2-03 派生时 L0 容量可能迫使使用 | 库外依赖 | ① §5.4 的 `settle` 对照先行预演。② 库修复后本脚本 `splitk_{bf16,f16}_m{16,32,64}_*` 全部真机逐位。③ M10-081 原 21 格扫描按 BF16/FP16 各跑一遍 |
| W6 | 换 dtype（FP32→BF16）绕开 FP32 规则 | — | **不推荐**。BF16 split-K 在 M16 上本身就错。inverse 需要 FP32 精度 | — |

**推荐**：A2-03 派生单元采用 **W1 + W2 + W4**，即所有 L0C 累加链显式 settle、不分 dtype。W5 以 `RISK` 报给 PM，转 ascriptor 所有者。
W3 只在 W2 的性能代价不可接受时考虑，走 kernel 批次。

## 9. 证据与复算

`benchmarks/a2/evidence/` 下全部是脱敏的原始回执：主机名、账号与绝对路径替换成 `<repo>`、`<ascriptor>`、`<cann>`、`<task-tmp>` 占位符，其余字节不动，也没有按行过滤。

| 路径 | 内容 |
|---|---|
| `device/env.json` | 每个真机作业的环境：python/torch/numpy、ascriptor commit 与两个修复文件的 sha256、CANN 版本、内置算子包、SoC、核数 |
| `device/device_props.log` | `torch_npu` 查到的设备属性（§3） |
| `device/card-*.log` | 每个真机作业未过滤的原始输出，`CASE …` 一行一次运行，`DECOMP …` 是错误输出的分片分解 |
| `device/receipts/card-*.json` | 每个作业一份：`summary` 加 `case_receipts`。每个 case 有定义、参考输出 sha256、kernel 源码 sha256、IR 事实、每次运行的逐位判定、max_abs_diff、rel_l2、错元素数、输出 sha256、分片分解。`--patch-generated` 的运行另记 `patch_generated` 与 `patch_insertions` |
| `device/card-*_m32_bf16_k32_harness_run.log`、`device/m32_bf16_k32_cann_runtime_errors.log` | M32 异常的 harness 输出与 CANN runtime 的 `[ERROR]` 行 |
| `device/cross_fragment_fit.log` | §5.3 的 64 项交叉分片拟合 |
| `generated/*_cube.h` | 真机上实际编译的 CCE cube 源码：FP32 与 BF16 split-K、K32 与 K128、settle 补丁前后、FP32 手写链、BF16 手写孪生 |
| `host/final_<profile>_<launcher>.log`、`host/receipts/<profile>_<launcher>.json` | reference / sim / pipesim 在 a2、a5 两个 profile 下的原始输出与回执（§5.1） |
| `host/hit_table.log`、`host/hit_table.json` | §7 |
| `host/pytest_host.log` | 本分支主机侧全量测试（屏蔽 torch_npu，挂 pin 版 ascriptor） |

复算：`python benchmarks/a2/repro_splitk_fp32.py --profile a2 --launcher reference` 会重算每个 case 的 CPU 参考。
它的 `reference_sha256` 应与回执里的一致。输入由 case id 与种子确定性生成（`make_inputs`），所以任何人都能在 CPU 上重建输入和参考，
再用回执里的 `output_sha256` 核对真机输出。

## 10. 没覆盖的

- **只测了 `block_dim=1`、单 cube 核**。多核、`bd1` 与 `bd4` 逐位对比、真实 KDA/GDN kernel 在 A2 上的整体结果都没测。
  那需要 A2-03 的派生单元，以及 A2-10/A2-11。
- **没有 A3 真机**。M10-081 记录里 A3 同样失效，本任务没有复测。
- **§5.4 的 settle 对照是诊断，不是修复**。它改的是一次运行的生成代码。真正的修复要在 ascriptor 侧落地并重跑全部矩阵。
- **BF16/FP16 split-K 的原因**已定位到"MMAD 之后缺 M 管线 settle"这一层：§5.4 的对照只差这一行，结果从错变对。
  更细的硬件机制，包括为什么 M32 表现为异常而不是错值，要由 ascriptor 所有者用有界探针确定。
