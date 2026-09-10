#!/usr/bin/env python3
"""从 docs/matrix/*.json 生成 docs/matrix/README.md。

json 是单一事实源，markdown 是产物 —— 不要手写 README.md。

    python tools/gen_matrix.py          # 生成
    python tools/gen_matrix.py --check  # 校验已生成的内容是最新的（CI 用）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "docs" / "matrix"
OUT = MATRIX / "README.md"

HEADER = "<!-- 由 tools/gen_matrix.py 生成，请勿手改。改 docs/matrix/*.json 后重新运行。 -->"

MARK = {"match": "✅", "passed": "✅", "gap": "❌", "failed": "❌", "untested": "⬜", "partial": "🟡"}
STATUS = {"not-started": "⬜ 未开始", "in-progress": "🟡 进行中", "done": "✅ 完成", "blocked": "❌ 受阻"}


def load(name: str) -> dict:
    return json.loads((MATRIX / name).read_text(encoding="utf-8"))


def mark(value: str) -> str:
    """状态值 → 带符号的展示文本。未知值原样返回。"""
    return f"{MARK[value]} {value}" if value in MARK else str(value)


def model_section(models: dict) -> list[str]:
    abi = models["ascriptor_a5_abi"]
    out = [
        "## 目标模型形状",
        "",
        f"ascriptor A5 定尺 ABI：`{abi['layout']}`，L={abi['L']}，D={abi['D']}，"
        f"q/k/v `{abi['dtype_qkv']}`，beta/g `{abi['dtype_beta_g']}`。",
        "",
        "| 模型 | 算子族 | 优先级 | 目标期 | H | HV | head_k | head_v | dtype | 定尺匹配 | 阻塞缺口 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models["models"]:
        s, fit = m["shape"], m["ascriptor_fit"]
        blocking = ", ".join(f"`{g}`" for g in fit["blocking_gaps"]) or "—"
        fit_marks = " ".join(
            MARK.get(fit.get(k, ""), "") for k in ("head_k_dim", "head_v_dim", "head_grouping")
        )
        phase = m.get("target_phase")
        out.append(
            f"| {m['label']} | {m['op_family']} | {m['priority']} | "
            f"{('第 %d 期' % phase) if phase else '—'} | "
            f"{s.get('num_key_heads', '—')} | {s.get('num_value_heads', '—')} | "
            f"{s.get('head_k_dim', '—')} | {s.get('head_v_dim', '—')} | "
            f"{s.get('dtype', '—')} | {fit_marks} | {blocking} |"
        )
    out += ["", "定尺匹配三格依次为 head_k / head_v / head 分组。", ""]

    out += ["### 算子测试应覆盖的形状", ""]
    for key, case in models["test_case_shapes"].items():
        if key == "note":
            continue
        dims = ", ".join(f"{k}={v}" for k, v in case.items() if k != "comment")
        out.append(f"- **{key}** — {dims} · {case.get('comment', '')}")
    out += ["", f"> {models['test_case_shapes']['note']}", ""]

    if models.get("internal_shapes_pending"):
        out += ["### 待核实的内部规格", ""]
        for p in models["internal_shapes_pending"]:
            out.append(f"- **{p['id']}**（{p['op_family']}）：{p['known']} — {p['action']}")
        out.append("")
    return out


def ops_section(ops: dict) -> list[str]:
    pin = ops["ascriptor_pin"]
    scope = pin["release_scope"]
    out = [
        "## 算子支持状态",
        "",
        f"ascriptor pin：`{pin['version']}` · library `{pin['library_commit'][:12]}` · "
        f"支持硬件 {', '.join(scope['supported_hardware'])} · "
        f"deferred {', '.join(scope['deferred_hardware'])}",
        "",
        "| 算子 | 族 | 方向 | reference | sim | pipesim | emit | **compile** | board(cce) | 本仓接线 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for op in ops["linear_attention_ops"]:
        v, st = op["validation"], op["our_status"]
        name = f"**`{op['id']}`** ★" if op.get("is_first_target") else f"`{op['id']}`"
        out.append(
            f"| {name} | {op['op_family']} | {op['direction']} | "
            + " | ".join(
                MARK.get(v.get(k, "untested"), "?")
                for k in ("reference", "sim", "pipesim", "emit", "compile", "board_cce")
            )
            + f" | {STATUS.get(st['wiring'], st['wiring'])} |"
        )
    out += [
        "",
        "★ 标记第一期的首个目标。",
        "",
        "> **compile** 一列指 `ascriptor compile` CLI（纯源码发射+编译、不执行），"
        "**不是** aclnn launcher —— unit runner 把 board / aclnn / pypto 都记为 `board` stage。"
        "本仓的 aclnn 本地编译与零拷贝调用已独立实测通过，见下方 `our_runtime_bridge`。",
        "",
    ]

    out += ["### 缺失的算子", ""]
    for m in ops["missing_ops"]:
        out.append(f"- **{m['id']}**（{m['op_family']}）— {m['purpose']}。{m.get('note', '')}")
    out.append("")

    out += ["### 可复用原语", "", "| 原语 | 对应 fla | 本仓状态 |", "|---|---|---|"]
    for p in ops["reusable_primitives"]:
        out.append(f"| `{p['id']}` | {p['maps_to']} | {STATUS.get(p['our_status'], p['our_status'])} |")
    out.append("")

    if bl := ops.get("performance_baselines"):
        c = bl["conditions"]
        out += [
            "### 性能基线", "",
            f"> {bl['note']}", "",
            f"机器：{bl['machine']} · 记录于 {bl['recorded_at']}",
            "",
            f"条件：dtype={c['dtype']} K=V={c['K']} chunk={c['chunk']} "
            f"warmup={c['warmup']} iters={c['iters']} "
            f"synchronized={'yes' if c['synchronized'] else 'no'} "
            f"forward_only={'yes' if c['forward_only'] else 'no'}",
            "",
        ]
        tn = bl["torch_npu_vectorized"]
        out += [f"> {tn['note']}", "",
                "| 形状 | B/H/HV/T | 四次测量 (ms) | 中位数 | o relL2 vs CPU |",
                "|---|---|---|---|---|"]
        for r in tn["ms_per_run"]:
            runs = " / ".join(f"{x:.3f}" for x in r["runs"])
            out.append(f"| {r['shape']} | {r['dims']} | {runs} | {r['median']:.3f} | "
                       f"{r['rel_l2_vs_cpu']:.3e} |")
        out += ["", f"**波动**：{tn['variance_note']}", ""]
        if asc := bl.get("ascriptor_self_compiled"):
            out += ["", f"> {asc['note']}", "",
                    "| 形状 | B/H/HV/T | bd=1 | bd=2 | bd=3 | bd=4 | bd1→4 | o relL2 |",
                    "|---|---|---|---|---|---|---|---|"]
            for r in asc["ms_by_block_dim"]:
                gain = f"{r['bd1'] / r['bd4']:.2f}x" if r["bd4"] else "—"
                out.append(f"| {r['shape']} | {r['dims']} | {r['bd1']:.3f} | {r['bd2']:.3f} | "
                           f"{r['bd3']:.3f} | {r['bd4']:.3f} | {gain} | {r['rel_l2_o_vs_cpu']:.3e} |")
            sp = asc["speedup_vs_torch_npu_at_bd4"]
            out += ["", "**block_dim=4 下 vs torch_npu 基线**："
                    + " · ".join(f"{k} {v}" for k, v in sp.items()), ""]
            kb = asc["kernel_breakdown_kimi_linear_layer"]
            out += [f"> {kb['note']}", "",
                    "| 段 | bd=1 (ms) | bd=4 (ms) |", "|---|---|---|"]
            for seg in kb["bd1"]:
                out.append(f"| `{seg}` | {kb['bd1'][seg]:.3f} | {kb['bd4'][seg]:.3f} |")
            out += ["", f"**怎么读**：{kb['reading']}", ""]

        out += ["", f"**观察**：{bl['observation']}", ""]
        if cc := bl.get("cross_cann_consistency"):
            out += [f"**跨 CANN 版本一致性**：{cc}", ""]
        if cv := bl.get("caveats"):
            out += [f"**测量注意**：{cv}", ""]
        if nx := bl.get("next"):
            out += [f"**下一步**：{nx}", ""]
        if nm := bl.get("not_yet_measured"):
            out += [f"**尚未测得**：{nm}", ""]

    stack = ops["stack_layers"]
    out += ["### 全链路三层", "", f"> {stack['note']}", ""]
    for layer in ("modules", "layers", "models"):
        out += [f"**{layer}**", ""]
        for item in stack[layer]:
            extra = item.get("ascriptor_asset") or item.get("strategy") or ""
            extra = f" · {extra}" if extra else ""
            out.append(f"- `{item['id']}` — {STATUS.get(item['our_status'], item['our_status'])}{extra}")
        out.append("")
    return out


FAMILY_LABEL = {"all": "全部", "kda": "KDA", "gated_delta_rule": "GDN", "delta_rule": "DeltaNet"}


def gaps_section(gaps: dict) -> list[str]:
    s = gaps["summary"]
    out = [
        "## 缺口",
        "",
        f"P0 {s['P0']} 项 · P1 {s['P1']} 项 · P2 {s['P2']} 项 · 共 {s['total']} 项",
        "",
        f"**首个里程碑**：{s['first_milestone']}",
        "",
        f"**建议的首个目标**：{s['recommended_first_target']}",
        "",
    ]

    if cmp_ := s.get("kda_vs_gdn"):
        out += [
            "### 为什么首个目标是 KDA",
            "",
            f"> {cmp_['note']}",
            "",
            "| 仅 KDA 具备 | 仅 GDN 具备 |",
            "|---|---|",
        ]
        kda_only, gdn_only = cmp_["kda_only"], cmp_["gdn_only"]
        for i in range(max(len(kda_only), len(gdn_only))):
            left = kda_only[i] if i < len(kda_only) else ""
            right = gdn_only[i] if i < len(gdn_only) else ""
            out.append(f"| {left} | {right} |")
        out.append("")

    # 按算子族速查：每族受哪些缺口影响
    out += ["### 按算子族速查", "", "| 算子族 | P0 | P1 | P2 |", "|---|---|---|---|"]
    for fam in ("kda", "gated_delta_rule", "delta_rule"):
        row = [FAMILY_LABEL[fam]]
        for sev in ("P0", "P1", "P2"):
            hit = [
                f"`{g['id']}`"
                for g in gaps["gaps"]
                if g["severity"] == sev and (fam in g["applies_to"] or "all" in g["applies_to"])
            ]
            row.append("<br>".join(hit) or "—")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    for sev in ("P0", "P1", "P2"):
        items = [g for g in gaps["gaps"] if g["severity"] == sev]
        if not items:
            continue
        out += [f"### {sev}", ""]
        for g in items:
            blocks = ", ".join(f"`{b}`" for b in g["blocks"]) or "—"
            applies = " / ".join(FAMILY_LABEL.get(f, f) for f in g["applies_to"])
            out += [
                f"#### `{g['id']}` — {g['title']}",
                "",
                f"- **类别** {g['category']} · **适用于** {applies} · **阻塞** {blocks}",
                f"- **依据** {g['evidence']}",
                f"- **影响** {g['impact']}",
                f"- **建议** {g['proposed_action']}",
                "",
            ]
    return out


def validate(models: dict, ops: dict, gaps: dict) -> list[str]:
    """三份 json 之间的交叉一致性。返回问题清单，空表示通过。"""
    problems: list[str] = []
    gap_ids = {g["id"] for g in gaps["gaps"]}

    # 模型引用的阻塞缺口必须有对应条目
    for m in models["models"]:
        for gid in m["ascriptor_fit"]["blocking_gaps"]:
            if gid not in gap_ids:
                problems.append(f"models.json: {m['id']} 引用了不存在的缺口 {gid!r}")

    # summary 的计数必须与实际条目一致
    actual = {sev: sum(g["severity"] == sev for g in gaps["gaps"]) for sev in ("P0", "P1", "P2")}
    for sev, n in actual.items():
        if gaps["summary"].get(sev) != n:
            problems.append(f"gaps.json: summary.{sev}={gaps['summary'].get(sev)}，实际 {n}")
    if gaps["summary"].get("total") != len(gaps["gaps"]):
        problems.append(f"gaps.json: summary.total={gaps['summary'].get('total')}，实际 {len(gaps['gaps'])}")

    # 缺口 id 不得重复
    dupes = {gid for gid in gap_ids if sum(g["id"] == gid for g in gaps["gaps"]) > 1}
    problems += [f"gaps.json: 缺口 id 重复 {gid!r}" for gid in sorted(dupes)]

    # 算子的 validation 必须用已知词汇
    known_stages = set(ops["stage_vocabulary"]["statuses"])
    for op in ops["linear_attention_ops"]:
        for stage, value in op["validation"].items():
            if stage.startswith("evidence") or stage == "note":
                continue
            if value not in known_stages:
                problems.append(f"ops.json: {op['id']}.{stage} 的状态 {value!r} 不在 stage_vocabulary 中")

    # 缺口的 applies_to 必须存在且用已知算子族
    for g in gaps["gaps"]:
        if not g.get("applies_to"):
            problems.append(f"gaps.json: {g['id']} 缺少 applies_to")
            continue
        for fam in g["applies_to"]:
            if fam not in FAMILY_LABEL:
                problems.append(f"gaps.json: {g['id']} 的 applies_to 含未知算子族 {fam!r}")

    # 首个目标必须在三份 json 里一致
    primary_families = {m["op_family"] for m in models["models"] if m["priority"] == "primary"}
    target_ops = {op["op_family"] for op in ops["linear_attention_ops"] if op.get("is_first_target")}
    if primary_families != target_ops:
        problems.append(
            f"首个目标不一致：models.json 的 primary 模型属于 {sorted(primary_families)}，"
            f"而 ops.json 标记 is_first_target 的算子属于 {sorted(target_ops)}"
        )
    target_layers = {L["id"] for L in ops["stack_layers"]["layers"] if L.get("is_first_target")}
    if len(target_layers) != 1:
        problems.append(f"ops.json: stack_layers.layers 应恰有一个 is_first_target，实际 {sorted(target_layers)}")
    return problems


def render() -> str:
    models, ops, gaps = load("models.json"), load("ops.json"), load("gaps.json")
    if problems := validate(models, ops, gaps):
        raise SystemExit("矩阵 json 不一致：\n" + "\n".join(f"  - {p}" for p in problems))
    lines = [
        HEADER,
        "",
        "# 支持矩阵",
        "",
        f"记录于 {models['recorded_at']}。本文件由 `docs/matrix/*.json` 生成。"
        "状态词汇沿用 ascriptor：`passed` / `untested` / `gap` / `failed`。",
        "",
    ]
    lines += model_section(models) + ops_section(ops) + gaps_section(gaps)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只校验产物是最新的，不写文件")
    args = ap.parse_args()

    content = render()
    if args.check:
        if not OUT.exists():
            print(f"{OUT.relative_to(ROOT)} 不存在，请运行 python tools/gen_matrix.py", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != content:
            print(f"{OUT.relative_to(ROOT)} 与 json 不一致，请重新运行 python tools/gen_matrix.py", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} 是最新的")
        return 0

    OUT.write_text(content, encoding="utf-8")
    print(f"已写入 {OUT.relative_to(ROOT)}（{len(content.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
