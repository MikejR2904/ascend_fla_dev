#!/usr/bin/env python3
"""多 agent 任务看板：校验、渲染、派单候选。

``docs/pm/board.json`` 是唯一事实源，**只有 PM 写**；本工具只读。协议见
``docs/pm/PROTOCOL.md``。

    python tools/pm_board.py --check                          # 校验（CI 用）
    python tools/pm_board.py --render                         # 按波次打印进度表
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
# 占着写集与卡的状态。blocked 也算：暂停中的任务仍持有分支与租约。
IN_FLIGHT = frozenset({"assigned", "in_progress", "blocked", "review", "rework"})
SOCS = ("any", "a2", "a3", "a5")
PRIORITIES = ("P0", "P1", "P2", "-")
REQUIRED = ("id", "title", "wave", "soc", "priority", "status", "deps", "write_set", "needs", "spec")
# 看板会被提交，绝不能混进机器信息（AGENTS.md §5）。这里只能拦最明显的一类。
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
        if status in IN_FLIGHT:
            if not t.get("assignee") or not t.get("branch"):
                problems.append(f"{tid}: {status} 必须有 assignee 与 branch")
            undone = [d for d in t["deps"] if by_id.get(d, {}).get("status") != "done"]
            if undone:
                problems.append(f"{tid}: 依赖 {undone} 未完成却已在进行中")
            if needs.get("npu"):
                lease = t.get("lease") or {}
                if lease.get("soc") != t["soc"] or not isinstance(lease.get("card"), int):
                    problems.append(f"{tid}: 需要 NPU 的进行中任务必须持有同 SoC 的卡租约，实际 {lease}")
        if status == "done" and not (t.get("result") or {}).get("commits"):
            problems.append(f"{tid}: done 必须在 result.commits 记下合入的提交")

    if cycle := _find_cycle(by_id):
        problems.append(f"依赖成环: {' -> '.join(cycle)}")

    flying = [t for t in valid if t["status"] in IN_FLIGHT]
    for i, t in enumerate(flying):
        for other in write_conflicts(t, flying[i + 1:]):
            problems.append(f"写集冲突: {t['id']} 与 {other} 同时在进行中")

    leased: dict[tuple, str] = {}
    for t in flying:
        lease = t.get("lease") or {}
        if lease.get("card") is None:
            continue
        key = (lease.get("soc"), lease.get("card"))
        if key in leased:
            problems.append(f"卡重复租出: {key} 同时给了 {leased[key]} 与 {t['id']}")
        leased[key] = t["id"]

    if m := _IP.search(json.dumps(board, ensure_ascii=False)):
        problems.append(f"看板里出现了疑似 IP 地址 {m.group(0)!r} —— 机器信息只能写在 machine_specs.md")
    return problems


def next_candidates(board: dict, socs: set[str], ascriptor: bool, fla: bool) -> list[dict]:
    """某个 agent 按能力可以接的任务，按看板顺序（即 PM 定的派单顺序）。

    不看卡是否空闲 —— 那由 PM 按 ``tmp/pm/leases.json`` 与 agent 报的 ``npu-smi`` 读数决定。
    """
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
                  "| id | soc | pri | status | assignee | deps | title |", "|---|---|---|---|---|---|---|"]
        for t in rows:
            status = t["status"] + (f" ({t['gate']})" if t["status"] == "gated" else "")
            lines.append(f"| {t['id']} | {t['soc']} | {t['priority']} | {status} | {t.get('assignee') or ''} "
                         f"| {', '.join(t['deps'])} | {t['title']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, default=BOARD)
    ap.add_argument("--check", action="store_true", help="校验看板")
    ap.add_argument("--render", action="store_true", help="按波次打印进度表")
    ap.add_argument("--next", action="store_true", help="列出某个 agent 能接的任务")
    ap.add_argument("--socs", default="", help="agent 可用的 SoC，逗号分隔，如 a2,a3；纯主机侧留空")
    ap.add_argument("--ascriptor", action="store_true", help="agent 有 ascriptor workspace")
    ap.add_argument("--fla", action="store_true", help="agent 装了 fla（oracle）")
    args = ap.parse_args()

    board = load(args.board)
    if not (args.check or args.render or args.next):
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
    if args.next:
        socs = {s for s in args.socs.split(",") if s}
        for t in next_candidates(board, socs, args.ascriptor, args.fla):
            print(f"{t['id']}\t{t['priority']}\t{t['soc']}\t{t['title']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
