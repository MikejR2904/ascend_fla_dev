#!/usr/bin/env python3
"""多 agent 任务看板：校验、渲染、派单候选。

``docs/pm/board.json`` 是唯一事实源，**只有 PM 写**；本工具只读。协议见
``docs/pm/PROTOCOL.md``，与 GitHub 的同步见 ``tools/pm_github.py``。

    python tools/pm_board.py --check                          # 校验（CI 用）
    python tools/pm_board.py --render                         # 按波次打印进度表
    python tools/pm_board.py --tree                           # 按 模型→kernel→dtype→机器 打印，标出起点
    python tools/pm_board.py --next --socs a2 --ascriptor --fla   # 某个 agent 能接的任务

只依赖标准库 —— 看板校验不该因为机器上没装 torch 而跑不了。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "docs" / "pm" / "board.json"

STATUSES = ("gated", "open", "assigned", "in_progress", "blocked", "review", "rework", "done", "cancelled")
# 占着写集的状态。blocked 也算：暂停中的任务仍持有分支。
IN_FLIGHT = frozenset({"assigned", "in_progress", "blocked", "review", "rework"})
SOCS = ("any", "a2", "a3", "a5")
PRIORITIES = ("P0", "P1", "P2", "-")
# 任务是怎么来的：用户提的 / PM 拆的 / 外部需求提案（GitHub issue，见 PROTOCOL §3.9）
ORIGIN_KINDS = ("user", "pm", "request")
REQUIRED = ("id", "title", "wave", "soc", "priority", "status", "deps", "write_set", "needs", "spec")
# 仓库是公开的，看板会被提交，绝不能混进机器信息（AGENTS.md §5）。这里只能拦最明显的一类。
_IP = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")


def load(path: Path = BOARD) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm(path: str) -> str:
    """``a/b/**`` → ``a/b``；``AGENTS.md#§2`` → ``AGENTS.md``（按文件算冲突，宁严勿松）。"""
    path = path.split("#", 1)[0].strip()
    if path.endswith("/**"):
        path = path[:-3]
    return path.rstrip("/")


def paths_overlap(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def write_conflicts(task: dict, others: list[dict]) -> list[str]:
    """``others`` 里与 ``task`` 写集有交集的任务 id。"""
    return sorted({
        o["id"] for o in others if o["id"] != task["id"]
        for x in task["write_set"] for y in o["write_set"] if paths_overlap(x, y)
    })


WAVE_ORDER = ("W0", "W-A2", "W-A3", "W-A5")


def _closure(by_id: dict[str, dict], tid: str) -> set[str]:
    """一个任务的全部传递依赖。"""
    seen, stack = set(), list(by_id[tid]["deps"])
    while stack:
        d = stack.pop()
        if d in seen or d not in by_id:
            continue
        seen.add(d)
        stack += by_id[d]["deps"]
    return seen


def _start_rank(task: dict) -> tuple:
    """起点排序：能马上做的排前面，然后按波次、优先级。"""
    status = 0 if task["status"] == "open" else (1 if task["status"] != "gated" else 2)
    wave = WAVE_ORDER.index(task["wave"]) if task["wave"] in WAVE_ORDER else len(WAVE_ORDER)
    prio = PRIORITIES.index(task["priority"]) if task["priority"] in PRIORITIES else len(PRIORITIES)
    return (status, wave, prio, task["id"])


def kernel_entries(board: dict) -> dict[str, list[str]]:
    """每个 kernel 的**起点**：该 kernel 组里，传递依赖中不含同组任务的那些。

    也就是"要让这个 kernel 动起来，先做哪一条"。起点是从依赖图算出来的，不手写 ——
    手写的标记迟早和 deps 漂移。

    用**传递**依赖而不是直接依赖，是因为同一个算子族的任务会跨 kernel 相互依赖：
    A2-15（层）直接依赖 A2-13/A2-14，它俩不在 kda_fwd_stable 组里，但顺着往上是 A2-03，
    而 A2-03 在组里 —— 所以 A2-15 不是 kda_fwd_stable 的起点。只看直接依赖会把它误判成起点。

    一个 kernel 可以有多个起点（互不依赖的几条路），按"能不能马上做"排序：
    open 排在 gated 前面，然后按波次与优先级。第一个就是下一步该做的那条。
    """
    by_id = {t["id"]: t for t in board["tasks"]}
    out: dict[str, list[str]] = {}
    for kernel in sorted({k for t in board["tasks"] for k in t.get("kernel") or []}):
        group = [t for t in board["tasks"] if kernel in (t.get("kernel") or [])]
        ids = {t["id"] for t in group}
        roots = [t for t in group if not (_closure(by_id, t["id"]) & ids)]
        out[kernel] = [t["id"] for t in sorted(roots, key=_start_rank)]
    return out


def chain_of(board: dict, kernel: str) -> list[dict]:
    """一个 kernel 组内部按依赖排好序的任务链。

    用**传递**依赖分层，和 :func:`kernel_entries` 保持一致 —— 只看直接依赖的话，
    A2-15 会排在 A2-12 前面（它对 fwd 组没有直接依赖），可它顺着 A2-13 往上要经过 A2-03。
    """
    by_id = {t["id"]: t for t in board["tasks"]}
    group = [t for t in board["tasks"] if kernel in (t.get("kernel") or [])]
    ids = {t["id"] for t in group}
    done, out = set(), []
    while len(out) < len(group):
        layer = [t for t in group if t["id"] not in done
                 and not (_closure(by_id, t["id"]) & ids - done)]
        if not layer:                      # 成环时兜底，别死循环（全局 --check 已经会报环）
            layer = [t for t in group if t["id"] not in done]
        out += layer
        done |= {t["id"] for t in layer}
    return out


def _check_axes(task: dict, axes: dict) -> list[str]:
    problems = []
    for axis in ("model", "kernel", "dtype"):
        value = task.get(axis)
        if value is None:
            problems.append(f"{task['id']}: 缺 {axis} 轴（没有就写空列表）")
            continue
        if not isinstance(value, list):
            problems.append(f"{task['id']}: {axis} 必须是列表，实际 {value!r}")
            continue
        vocab = set(axes.get(axis, []))
        if axis == "dtype":
            vocab |= set(axes.get("dtype_gated", {}).get("values", []))
        if unknown := [v for v in value if v not in vocab]:
            problems.append(f"{task['id']}: {axis} 里有未知取值 {unknown}，词汇表见 board.axes.{axis}")
    return problems


def _check_origin(task: dict) -> list[str]:
    """任务来源。外部需求（GitHub issue）**不会自己变成可派的任务** —— 放行是用户的事
    （`AGENTS.md` §1：改变范围要显式决策），所以 request 来源的任务在用户批准前只能是 gated。
    """
    origin = task.get("origin")
    if origin is None:
        return []
    tid = task["id"]
    if not isinstance(origin, dict) or origin.get("kind") not in ORIGIN_KINDS:
        return [f"{tid}: origin.kind 必须是 {ORIGIN_KINDS} 之一，实际 {origin!r}"]
    if origin["kind"] != "request":
        return []
    problems = []
    num = origin.get("issue")
    if not isinstance(num, int) or isinstance(num, bool) or num <= 0:
        problems.append(f"{tid}: request 来源必须记下提案 issue 号，实际 {num!r}")
    if not origin.get("by"):
        problems.append(f"{tid}: request 来源必须记下提案人 GitHub 账号")
    if not origin.get("approved_by_user") and task["status"] not in ("gated", "cancelled"):
        problems.append(f"{tid}: 来自外部需求且用户尚未批准，只能是 gated（当前 {task['status']}）")
    return problems


def _find_cycle(by_id: dict[str, dict]) -> list[str] | None:
    state: dict[str, int] = {}  # 1 = 在栈上，2 = 已完成
    stack: list[str] = []

    def visit(tid: str) -> list[str] | None:
        state[tid] = 1
        stack.append(tid)
        for dep in by_id[tid]["deps"]:
            if dep not in by_id:
                continue
            if state.get(dep) == 1:
                return stack[stack.index(dep):] + [dep]
            if dep not in state and (found := visit(dep)):
                return found
        stack.pop()
        state[tid] = 2
        return None

    for tid in by_id:
        if tid not in state and (found := visit(tid)):
            return found
    return None


def check(board: dict, root: Path = ROOT) -> list[str]:
    """返回问题清单，空表示通过。"""
    problems: list[str] = []
    tasks = board.get("tasks", [])
    ids = [t.get("id") for t in tasks]
    problems += [f"任务 id 重复 {tid!r}" for tid in sorted({i for i in ids if ids.count(i) > 1})]

    valid = []
    for t in tasks:
        if missing := [k for k in REQUIRED if k not in t]:
            problems.append(f"{t.get('id', '?')}: 缺字段 {missing}")
        else:
            valid.append(t)
    by_id = {t["id"]: t for t in valid}

    for t in valid:
        tid, status, needs = t["id"], t["status"], t["needs"]
        if status not in STATUSES:
            problems.append(f"{tid}: 状态 {status!r} 不在词汇表 {STATUSES} 中")
        if t["soc"] not in SOCS:
            problems.append(f"{tid}: soc {t['soc']!r} 不在 {SOCS} 中")
        if t["priority"] not in PRIORITIES:
            problems.append(f"{tid}: priority {t['priority']!r} 不在 {PRIORITIES} 中")
        problems += [f"{tid}: 依赖了不存在的任务 {d!r}" for d in t["deps"] if d not in by_id]
        if needs.get("npu") and t["soc"] == "any":
            problems.append(f"{tid}: 需要 NPU 却没指定 SoC —— 结论不能跨 SoC 搬")
        if status == "gated" and not t.get("gate"):
            problems.append(f"{tid}: gated 必须写明 gate（等什么）")
        if status not in ("gated", "cancelled"):
            if not t["spec"]:
                problems.append(f"{tid}: 状态 {status} 的任务必须有 spec")
            elif not (Path(root) / t["spec"]).is_file():
                problems.append(f"{tid}: spec 文件不存在 {t['spec']}")
        for key in ("issue", "pr"):
            value = t.get(key)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                problems.append(f"{tid}: {key} 必须是正整数或 null，实际 {value!r}")
        problems += _check_origin(t)
        problems += _check_axes(t, board.get("axes", {}))
        if status in IN_FLIGHT:
            if not t.get("assignee") or not t.get("branch"):
                problems.append(f"{tid}: {status} 必须有 assignee 与 branch")
            undone = [d for d in t["deps"] if by_id.get(d, {}).get("status") != "done"]
            if undone:
                problems.append(f"{tid}: 依赖 {undone} 未完成却已在进行中")
            # 机器归 agent 管；PM 只核对 agent 声明过这个 SoC 的真机能力。
            if needs.get("npu"):
                declared = (t.get("assignee_caps") or {}).get("socs") or []
                if t["soc"] not in declared:
                    problems.append(f"{tid}: 需要 {t['soc']} 真机，assignee 声明的 SoC 是 {declared}")
        if status == "done" and not (t.get("result") or {}).get("commits"):
            problems.append(f"{tid}: done 必须在 result.commits 记下合入的提交")

    if cycle := _find_cycle(by_id):
        problems.append(f"依赖成环: {' -> '.join(cycle)}")

    flying = [t for t in valid if t["status"] in IN_FLIGHT]
    for i, t in enumerate(flying):
        for other in write_conflicts(t, flying[i + 1:]):
            problems.append(f"写集冲突: {t['id']} 与 {other} 同时在进行中")

    holders: dict[str, str] = {}
    for t in flying:
        who = t.get("assignee")
        if who in holders:
            problems.append(f"{who} 同时持有 {holders[who]} 与 {t['id']} —— 一个 agent 同时只接一个任务")
        elif who:
            holders[who] = t["id"]

    numbers = [t["issue"] for t in valid if isinstance(t.get("issue"), int)]
    problems += [f"issue #{n} 被多个任务引用" for n in sorted({n for n in numbers if numbers.count(n) > 1})]

    # 仓主并行轨道的文件不能落进任何任务的写集 —— 两条轨道共用仓库，撞车只能靠边界预防
    reserved = (board.get("reserved_paths") or {}).get("paths", [])
    for t in valid:
        hits = sorted({r for r in reserved for w in t["write_set"] if paths_overlap(r, w)})
        if hits:
            problems.append(f"{t['id']}: 写集碰到仓主并行轨道的路径 {hits}（见 board.reserved_paths）")

    # 每个 kernel 都要有起点，否则"从哪条开始"这个问题没有答案
    for kernel, entries in kernel_entries(board).items():
        if not entries:
            problems.append(f"kernel {kernel!r} 的任务互相依赖成环，找不到起点")

    if m := _IP.search(json.dumps(board, ensure_ascii=False)):
        problems.append(f"看板里出现了疑似 IP 地址 {m.group(0)!r} —— 机器信息只能写在 machine_specs.md")
    return problems


def next_candidates(board: dict, socs: set[str], ascriptor: bool, fla: bool) -> list[dict]:
    """某个 agent 按声明的能力可以接的任务，按看板顺序（即 PM 定的派单顺序）。"""
    tasks = board["tasks"]
    by_id = {t["id"]: t for t in tasks}
    flying = [t for t in tasks if t["status"] in IN_FLIGHT]
    out = []
    for t in tasks:
        needs = t["needs"]
        if t["status"] != "open":
            continue
        if any(by_id.get(d, {}).get("status") != "done" for d in t["deps"]):
            continue
        if needs.get("npu") and t["soc"] not in socs:
            continue
        if needs.get("ascriptor") and not ascriptor:
            continue
        if needs.get("fla") and not fla:
            continue
        if write_conflicts(t, flying):
            continue
        out.append(t)
    return out


def render(board: dict) -> str:
    lines = [f"看板更新于 {board.get('updated_at', '?')}", ""]
    waves = [w["id"] for w in board.get("waves", [])]
    waves += sorted({t["wave"] for t in board["tasks"]} - set(waves))
    for wave in waves:
        rows = [t for t in board["tasks"] if t["wave"] == wave]
        if not rows:
            continue
        title = next((w.get("title", "") for w in board.get("waves", []) if w["id"] == wave), "")
        counts = {s: sum(t["status"] == s for t in rows) for s in STATUSES}
        summary = " ".join(f"{s}={n}" for s, n in counts.items() if n)
        lines += [f"## {wave} {title}（{summary}）", "",
                  "| id | issue | soc | pri | status | assignee | deps | title |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in rows:
            status = t["status"] + (f" ({t['gate']})" if t["status"] == "gated" else "")
            issue = f"#{t['issue']}" if t.get("issue") else ""
            lines.append(f"| {t['id']} | {issue} | {t['soc']} | {t['priority']} | {status} | {t.get('assignee') or ''} "
                         f"| {', '.join(t['deps'])} | {t['title']} |")
        lines.append("")
    return "\n".join(lines)


def _row(t: dict, indent: str, entries: set[str]) -> str:
    mark = " ★起点" if t["id"] in entries else ""
    issue = f"#{t['issue']} " if t.get("issue") else ""
    gate = f"（等 {t['gate']}）" if t["status"] == "gated" else ""
    return f"{indent}{issue}{t['id']} [{t['status']}{gate}] soc={t['soc']} {t['title']}{mark}"


def render_tree(board: dict) -> str:
    """按 模型 → kernel → 数据类型 → 机器 的层次打印，并标出每个 kernel 的起点。"""
    tasks = board["tasks"]
    all_entries = kernel_entries(board)
    lines = ["任务按 模型 → kernel → 数据类型 → 机器 归类；★起点 = 该 kernel 组里依赖全在组外的任务", ""]

    for model in board.get("axes", {}).get("model", []):
        owned = [t for t in tasks if model in (t.get("model") or [])]
        if not owned:
            continue
        lines.append(f"## 模型 {model}（{len(owned)} 条）")
        kernels = sorted({k for t in owned for k in t.get("kernel") or []})
        for kernel in kernels:
            entries = set(all_entries.get(kernel, []))
            chain = [t for t in chain_of(board, kernel) if model in (t.get("model") or [])]
            # 起点按**整条 kernel 链**算，不按本模型这一片算 —— 否则会把"本模型里最早的一条"
            # 误报成起点，而真正要先做的那条可能挂在别的模型下面
            ordered = all_entries.get(kernel, [])
            head = f"下一步 {ordered[0]}" if ordered else "无起点"
            rest = f"；其他入口 {', '.join(ordered[1:])}" if len(ordered) > 1 else ""
            lines.append(f"  ### kernel {kernel} —— {head}{rest}")
            for dtype in sorted({d for t in chain for d in t.get("dtype") or []}) or ["(未标注)"]:
                inner = [t for t in chain if dtype in (t.get("dtype") or [])]
                if not inner:
                    continue
                lines.append(f"    dtype {dtype}")
                for soc in sorted({t["soc"] for t in inner}):
                    lines += [_row(t, "      ", entries) for t in inner if t["soc"] == soc]
        loose = [t for t in owned if not t.get("kernel")]
        if loose:
            lines.append("  ### 不绑定 kernel（模型级 / 模块级）")
            lines += [_row(t, "    ", set()) for t in loose]
        lines.append("")

    lines.append("## 每个 kernel 的完整链（跨模型）")
    for kernel, entries in all_entries.items():
        head = f"下一步 {entries[0]}" if entries else "无起点"
        rest = f"；其他入口 {', '.join(entries[1:])}" if len(entries) > 1 else ""
        lines.append(f"  {kernel}：{head}{rest}")
        lines += [_row(t, "    ", set(entries)) for t in chain_of(board, kernel)]
    lines.append("")

    infra = [t for t in tasks if not t.get("model") and not t.get("kernel")]
    if infra:
        lines.append(f"## 基础设施 / 不属于任何模型或 kernel（{len(infra)} 条）")
        lines += [_row(t, "  ", set()) for t in infra]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, default=BOARD)
    ap.add_argument("--check", action="store_true", help="校验看板")
    ap.add_argument("--render", action="store_true", help="按波次打印进度表")
    ap.add_argument("--tree", action="store_true",
                    help="按 模型 → kernel → 数据类型 → 机器 打印，并标出每个 kernel 的起点")
    ap.add_argument("--next", action="store_true", help="列出某个 agent 能接的任务")
    ap.add_argument("--socs", default="", help="agent 声明可用的 SoC，逗号分隔，如 a2,a3；纯主机侧留空")
    ap.add_argument("--ascriptor", action="store_true", help="agent 有 ascriptor workspace")
    ap.add_argument("--fla", action="store_true", help="agent 装了 fla（oracle）")
    args = ap.parse_args()

    board = load(args.board)
    if not (args.check or args.render or args.next or args.tree):
        args.check = True
    rc = 0
    if args.check:
        problems = check(board)
        if problems:
            print("看板不一致：\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
            rc = 1
        else:
            print(f"{args.board.name}: {len(board['tasks'])} 个任务，校验通过")
    if args.render:
        print(render(board))
    if args.tree:
        print(render_tree(board))
    if args.next:
        socs = {s for s in args.socs.split(",") if s}
        for t in next_candidates(board, socs, args.ascriptor, args.fla):
            print(f"{t['id']}\t{t['priority']}\t{t['soc']}\t{t['title']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
