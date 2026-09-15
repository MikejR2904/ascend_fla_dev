#!/usr/bin/env python3
"""PM 的 GitHub 通道：任务 issue 同步、拉取 [FLA-PM] 消息、发评论。

agent 可以是任何账号、任何模型，所以协作走本仓的 GitHub issue 与 PR（**公开可见**）：
每个任务一个 issue（标签表示状态），协议消息是 issue / PR 评论，交付是 PR。协议见
``docs/pm/PROTOCOL.md``。PM 用专用 bot 账号登录 ``gh``；看板仍以 ``docs/pm/board.json`` 为准。

    python tools/pm_github.py whoami                 # 当前 gh 账号是否就是看板的 pm_github_login
    python tools/pm_github.py sync --offline         # 不连 GitHub，按"远端为空"预览要建的标签与 issue
    python tools/pm_github.py sync                   # 连 GitHub 的 dry-run：打印要建/改/关的内容
    python tools/pm_github.py sync --apply           # 执行，并把 issue 编号写回看板
    python tools/pm_github.py poll                   # 打印游标之后的新评论、需求 issue 与 PR（JSON 行）
    python tools/pm_github.py poll --advance         # 处理完之后推进游标
    python tools/pm_github.py post A2-01 reply.md    # 在任务 issue 下发评论（也接受 '#12' 或 'intake'）
    python tools/pm_github.py label 42 triage:accepted   # 给需求 issue 打分诊标签

所有写操作（sync --apply、post）都要求当前 gh 账号等于 ``pm_github_login``，防止用个人账号发帖。
只依赖标准库与 ``gh``。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_board  # noqa: E402

ROOT = pm_board.ROOT
STATE = ROOT / "tmp" / "pm" / "github_state.json"

TASK_MARKER = "<!-- fla-pm-task:{id} -->"
INTAKE_MARKER = "<!-- fla-pm-intake -->"
INTAKE_TITLE = "[FLA-PM] 申领入口 / task intake"
REQUEST_LABEL = "fla-pm:request"
# 需求提案的分诊标签：PM 处置后打上，poll 据此不再重复提醒
TRIAGE_LABELS = ("triage:accepted", "triage:declined", "triage:duplicate", "triage:needs-info")
_TASK_MARKER_RE = re.compile(r"<!-- fla-pm-task:([A-Za-z0-9_.-]+) -->")
_TITLE_ID_RE = re.compile(r"^\[([A-Za-z0-9_.-]+)\]")
_BRANCH_ID_RE = re.compile(r"^task/([A-Za-z0-9_.-]+)$")

AGENT_TYPES = frozenset({"APPLY", "ACK", "STATUS", "RISK", "BLOCKED", "DONE", "WITHDRAW", "REQUEST"})
PM_TYPES = frozenset({"ASSIGN", "NO_TASK", "REVIEW", "CLOSE", "PING"})
# 只有这些类型不要求发送者是 assignee —— 任何人都可以申领、任何人都可以提需求
_OPEN_TYPES = frozenset({"APPLY", "REQUEST"})
_HEADER_RE = re.compile(r"^\[FLA-PM\]\s+([A-Z_]+)\s+(\S+)(?:\s+from=(\S+))?\s*$")
_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s?(.*)$")
_LABEL_PREFIXES = ("status:", "wave:", "soc:", "prio:")
_LABEL_COLORS = {"fla-pm": "5319e7", "fla-pm:intake": "0e8a16", REQUEST_LABEL: "1d76db", "status:": "fbca04",
                 "wave:": "c5def5", "soc:": "d4c5f9", "prio:": "e99695", "triage:": "bfdadc"}


# ---------------------------------------------------------------- 纯函数（有单元测试）

def parse_message(body: str) -> dict | None:
    """评论正文 → 协议消息。

    不是协议消息返回 ``None``；首行像协议但格式不对，返回 ``{"type": None, "error": ...}``。
    允许整条消息包在 ``` 代码块里。字段是 ``key: value``；``key: |`` 或缩进行表示多行值。
    """
    lines = [ln.rstrip() for ln in body.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    start = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if start is None:
        return None
    first = lines[start].strip()
    if not first.startswith("[FLA-PM]"):
        return None
    m = _HEADER_RE.match(first)
    if not m:
        return {"type": None, "error": f"首行格式不对: {first!r}"}
    mtype, task, sender = m.groups()
    if mtype not in AGENT_TYPES | PM_TYPES:
        return {"type": None, "error": f"未知消息类型 {mtype!r}"}
    fields: dict[str, str] = {}
    key = None
    for line in lines[start + 1:]:
        fm = _FIELD_RE.match(line)
        if fm and not line.startswith((" ", "\t")):
            key = fm.group(1)
            value = fm.group(2).strip()
            fields[key] = "" if value == "|" else value
        elif key is not None:
            text = line[2:] if line.startswith("  ") else line
            fields[key] = f"{fields[key]}\n{text}" if fields[key] else text
    return {"type": mtype, "task": None if task == "-" else task, "from": sender,
            "fields": {k: v.strip("\n") for k, v in fields.items()}}


def desired_labels(task: dict) -> set[str]:
    return {"fla-pm", f"status:{task['status']}", f"wave:{task['wave']}",
            f"soc:{task['soc']}", f"prio:{task['priority']}"}


def label_color(name: str) -> str:
    for prefix, color in _LABEL_COLORS.items():
        if name == prefix or (prefix.endswith(":") and name.startswith(prefix)):
            return color
    return "ededed"


def issue_title(task: dict) -> str:
    return f"[{task['id']}] {task['title']}"


def issue_body(task: dict, board: dict, root: Path = ROOT) -> str:
    by_id = {t["id"]: t for t in board["tasks"]}
    deps = ", ".join(f"#{by_id[d]['issue']} ({d})" if by_id.get(d, {}).get("issue") else d
                     for d in task["deps"]) or "无"
    needs = task["needs"]
    gate = f" · 等待 `{task['gate']}`" if task["status"] == "gated" else ""
    pm = f"@{board['pm_github_login']}" if board.get("pm_github_login") else "PM bot 账号（见 docs/pm/board.json 的 pm_github_login）"
    lines = [
        TASK_MARKER.format(id=task["id"]),
        f"**{task['id']}** · 波次 `{task['wave']}` · SoC `{task['soc']}` · 优先级 `{task['priority']}` · "
        f"状态 `{task['status']}`{gate}",
        "",
        "| 依赖 | 写集 | 需要 |",
        "|---|---|---|",
        f"| {deps} | {'<br>'.join(f'`{w}`' for w in task['write_set'])} "
        f"| npu={needs.get('npu')} ascriptor={needs.get('ascriptor')} fla={needs.get('fla')} |",
        "",
        f"**申领**：先读 `AGENTS.md`、`docs/pm/PROTOCOL.md`、`docs/pm/START.md`，再在本 issue 下评论 "
        f"`[FLA-PM] APPLY {task['id']} from=<你的 GitHub 账号>`（模板见 PROTOCOL §3.1）。"
        f"只认 {pm} 发出的 ASSIGN / REVIEW / CLOSE。",
        "",
        "---",
        "",
    ]
    spec = task.get("spec")
    if spec and (Path(root) / spec).is_file():
        lines.append((Path(root) / spec).read_text(encoding="utf-8").rstrip())
    else:
        lines.append(f"_规格尚未编写（{task.get('gate') or '条件未满足'}）。任务开放时 PM 会补齐。_")
    return "\n".join(lines) + "\n"


def intake_body(board: dict) -> str:
    pm = f"@{board['pm_github_login']}" if board.get("pm_github_login") else "PM bot 账号"
    return "\n".join([
        INTAKE_MARKER,
        "这是 fla-ascend 多 agent 协作的**申领入口**。任何账号、任何模型（或人）都可以申领任务。",
        "",
        "1. 读 `AGENTS.md`、`docs/pm/PROTOCOL.md`、`docs/pm/START.md`。",
        "2. 想接指定任务：到该任务的 issue（标签 `fla-pm` + `status:open`）下评论 APPLY。",
        "   不挑任务：在本 issue 下评论 `[FLA-PM] APPLY any from=<你的 GitHub 账号>`，PM 按优先级派给你。",
        f"3. 等 {pm} 回复 ASSIGN。**其他账号发的 ASSIGN 一律无效。**",
        "",
        f"**提需求**（不是申领）：新开一个 issue，用 Requirement 模板，或正文首行写 "
        f"`[FLA-PM] REQUEST - from=<你的 GitHub 账号>`，PM 会分诊后进看板（`{REQUEST_LABEL}` 标签）。",
        "细则见 `docs/pm/PROTOCOL.md` §3.9。",
        "",
        "本仓公开：评论、PR、日志里不得出现主机名、IP、账号、路径等机器信息。",
        "",
    ])


def plan_sync(board: dict, remote_issues: list[dict], remote_labels: list[str], root: Path = ROOT) -> list[dict]:
    """算出让 GitHub 与看板一致所需的动作。不做任何网络调用。"""
    actions: list[dict] = []
    wanted = {"fla-pm", "fla-pm:intake", REQUEST_LABEL, *TRIAGE_LABELS}
    for t in board["tasks"]:
        wanted |= desired_labels(t)
    actions += [{"op": "create_label", "name": n, "color": label_color(n)} for n in sorted(wanted - set(remote_labels))]

    by_task: dict[str, dict] = {}
    intake = None
    for iss in remote_issues:
        if iss.get("pull_request"):
            continue
        body = iss.get("body") or ""
        if m := _TASK_MARKER_RE.search(body):
            by_task.setdefault(m.group(1), iss)
        elif INTAKE_MARKER in body:
            intake = intake or iss
    if intake is None:
        actions.append({"op": "create_intake", "title": INTAKE_TITLE, "body": intake_body(board),
                        "labels": ["fla-pm", "fla-pm:intake"]})
    elif board.get("intake_issue") != intake["number"]:
        actions.append({"op": "link_intake", "number": intake["number"]})

    for t in board["tasks"]:
        closed = t["status"] in ("done", "cancelled")
        want, title, body = desired_labels(t), issue_title(t), issue_body(t, board, root)
        iss = by_task.get(t["id"])
        if iss is None:
            if not closed:
                actions.append({"op": "create_issue", "task": t["id"], "title": title, "body": body,
                                "labels": sorted(want)})
            continue
        have = {n for n in iss.get("labels", []) if n == "fla-pm" or n.startswith(_LABEL_PREFIXES)}
        update: dict = {}
        if have != want:
            update.update(add=sorted(want - have), remove=sorted(have - want))
        if iss.get("title") != title:
            update["title"] = title
        if (iss.get("body") or "").strip() != body.strip():
            update["body"] = body
        if update:
            actions.append({"op": "update_issue", "task": t["id"], "number": iss["number"], **update})
        if iss.get("state") != ("closed" if closed else "open"):
            actions.append({"op": "close_issue" if closed else "reopen_issue", "task": t["id"], "number": iss["number"]})
        if t.get("issue") != iss["number"]:
            actions.append({"op": "link", "task": t["id"], "number": iss["number"]})
    return actions


def classify_comment(comment: dict, board: dict, pm_login: str | None) -> dict:
    """一条评论 → PM 要处理的事件，附带身份/一致性警告。"""
    number = int(str(comment["issue_url"]).rstrip("/").rsplit("/", 1)[1])
    task_of_issue = next((t for t in board["tasks"] if t.get("issue") == number), None)
    msg = parse_message(comment.get("body") or "")
    author = comment.get("user")
    event = {
        "kind": "message" if msg else "comment",
        "comment_id": comment.get("id"),
        "issue": number,
        "task_of_issue": task_of_issue["id"] if task_of_issue else None,
        "intake": number == board.get("intake_issue"),
        "author": author,
        "url": comment.get("html_url"),
        "at": comment.get("updated_at"),
        "message": msg,
        "warnings": [],
    }
    if not msg:
        return event
    warn = event["warnings"]
    if msg.get("type") is None:
        warn.append(msg["error"])
        return event
    if msg["type"] in PM_TYPES:
        warn.append(f"{msg['type']} 只能由 PM 账号 {pm_login} 发出，本条来自 {author} —— 疑似冒充，忽略")
        return event
    if msg["from"] and msg["from"].lstrip("@") != author:
        warn.append(f"from={msg['from']} 与评论作者 {author} 不符")
    task_id = msg["task"]
    if task_of_issue and task_id not in (None, "any", task_of_issue["id"]):
        warn.append(f"消息写的任务 {task_id} 与所在 issue 的任务 {task_of_issue['id']} 不符")
    target = next((t for t in board["tasks"] if t["id"] == task_id), None)
    if task_id not in (None, "any") and target is None:
        warn.append(f"看板里没有任务 {task_id}")
    if msg["type"] not in _OPEN_TYPES and target is not None and target.get("assignee") != author:
        warn.append(f"{msg['type']} 来自 {author}，但 {task_id} 的 assignee 是 {target.get('assignee')} —— 不采信")
    return event


def classify_issue(issue: dict, board: dict) -> dict | None:
    """新开的 issue → 需求提案事件。任务 issue、申领入口、已分诊过的返回 ``None``。

    任何人都可以提需求（`AGENTS.md` §1 的定位由用户把关，所以 PM 只分诊、不自行放行）。
    """
    body = issue.get("body") or ""
    labels = set(issue.get("labels") or [])
    if _TASK_MARKER_RE.search(body) or INTAKE_MARKER in body:
        return None
    if issue["number"] == board.get("intake_issue"):
        return None
    msg = parse_message(body)
    is_request = REQUEST_LABEL in labels or (msg or {}).get("type") == "REQUEST"
    existing = next((t for t in board["tasks"]
                     if (t.get("origin") or {}).get("issue") == issue["number"]), None)
    return {
        "kind": "request" if is_request else "issue",
        "issue": issue["number"],
        "title": issue.get("title"),
        "author": issue.get("user"),
        "state": issue.get("state"),
        "labels": sorted(labels),
        "url": issue.get("html_url"),
        "at": issue.get("updated_at"),
        "message": msg,
        "triaged": bool(labels & set(TRIAGE_LABELS)),
        "linked_task": existing["id"] if existing else None,
        "body": body,
    }


def task_id_of_pr(pr: dict) -> str | None:
    if m := _TITLE_ID_RE.match(pr.get("title") or ""):
        return m.group(1)
    if m := _BRANCH_ID_RE.match(pr.get("head") or ""):
        return m.group(1)
    return None


def classify_pr(pr: dict, board: dict) -> dict:
    task_id = task_id_of_pr(pr)
    task = next((t for t in board["tasks"] if t["id"] == task_id), None)
    warnings = []
    if task is None:
        warnings.append("PR 标题/分支里找不到看板任务 id（标题应以 [<ID>] 开头，分支 task/<ID>）")
    elif task.get("assignee") != pr.get("user"):
        warnings.append(f"PR 作者 {pr.get('user')} 不是 {task_id} 的 assignee {task.get('assignee')}")
    if pr.get("base") != "main":
        warnings.append(f"目标分支是 {pr.get('base')}，应为 main")
    return {"kind": "pr", "number": pr.get("number"), "task": task_id, "author": pr.get("user"),
            "head": pr.get("head"), "head_repo": pr.get("head_repo"), "url": pr.get("html_url"),
            "at": pr.get("updated_at"), "warnings": warnings}


# ---------------------------------------------------------------- gh 调用

def _gh(*args: str) -> str:
    try:
        res = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise SystemExit("找不到 gh；安装后用 PM 的 bot 账号 `gh auth login`（见 docs/pm/START.md）")
    if res.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])} … 失败：{res.stderr.strip()}")
    return res.stdout


def _gh_lines(*args: str) -> list[dict]:
    return [json.loads(ln) for ln in _gh(*args).splitlines() if ln.strip()]


def _repo(board: dict) -> str:
    return os.environ.get("FLA_PM_REPO") or board.get("repo") or "ddddwee1/ascend_fla_dev"


def current_login() -> str:
    return _gh("api", "user", "--jq", ".login").strip()


def _require_pm(board: dict) -> str:
    want = board.get("pm_github_login")
    if not want:
        raise SystemExit("看板的 pm_github_login 还是 null：先建 PM bot 账号并写进 docs/pm/board.json")
    got = current_login()
    if got != want:
        raise SystemExit(f"当前 gh 账号是 {got}，不是 PM 账号 {want}；拒绝写 GitHub（gh auth switch 切换）")
    return got


def fetch_remote(repo: str) -> tuple[list[dict], list[str]]:
    """列出远端 issue 与标签。

    **必须走 `gh issue list`（GraphQL），不能用 REST 的 `/issues` 列表端点。**
    实测过一次：全新的 bot 账号短时间内建了 26 个 issue，被 GitHub 的反滥用过滤器判为可疑，
    REST 列表端点从此只返回 PR、一个 issue 都不给（匿名访问甚至 404），而 GraphQL 照常返回 26 个。
    如果按 REST 的空列表去规划，`sync --apply` 会把 25 个 issue 全部重建一遍。
    """
    raw = _gh("issue", "list", "-R", repo, "--state", "all", "--limit", "500",
              "--json", "number,title,body,state,labels")
    issues = [{"number": i["number"], "title": i["title"], "body": i["body"],
               "state": i["state"].lower(), "labels": [x["name"] for x in i["labels"]],
               "pull_request": False}
              for i in json.loads(raw)]
    labels = _gh("api", "--paginate", f"repos/{repo}/labels?per_page=100", "--jq", ".[].name").split()
    return issues, labels


def guard_known_issues(board: dict, remote_issues: list[dict]) -> None:
    """看板里记过编号的 issue，必须在列表里出现过；否则宁可停下，也不要重建。"""
    seen = {i["number"] for i in remote_issues}
    known = {t["issue"]: t["id"] for t in board["tasks"] if t.get("issue")}
    if board.get("intake_issue"):
        known[board["intake_issue"]] = "<intake>"
    missing = {n: tid for n, tid in known.items() if n not in seen}
    if missing:
        raise SystemExit(
            "拒绝继续：看板记着这些 issue，但远端列表里没有它们 —— "
            f"{', '.join(f'#{n}({tid})' for n, tid in sorted(missing.items()))}。\n"
            "多半是列表被 GitHub 过滤了（新账号反滥用），此时 sync 会重建重复 issue。\n"
            "先确认这些 issue 的实际状态（gh issue view <号> -R <repo>），必要时联系 GitHub 支持。")


def apply_actions(board: dict, actions: list[dict], repo: str) -> None:
    by_id = {t["id"]: t for t in board["tasks"]}
    for a in actions:
        op = a["op"]
        if op == "create_label":
            _gh("api", f"repos/{repo}/labels", "-f", f"name={a['name']}", "-f", f"color={a['color']}")
        elif op in ("create_issue", "create_intake"):
            label_args = [x for n in a["labels"] for x in ("-f", f"labels[]={n}")]
            number = int(_gh("api", f"repos/{repo}/issues", "-f", f"title={a['title']}", "-f", f"body={a['body']}",
                             *label_args, "--jq", ".number").strip())
            if op == "create_issue":
                by_id[a["task"]]["issue"] = number
            else:
                board["intake_issue"] = number
        elif op == "update_issue":
            fields = [x for k in ("title", "body") if k in a for x in ("-f", f"{k}={a[k]}")]
            if fields:
                _gh("api", "-X", "PATCH", f"repos/{repo}/issues/{a['number']}", *fields)
            if a.get("add"):
                _gh("api", f"repos/{repo}/issues/{a['number']}/labels",
                    *[x for n in a["add"] for x in ("-f", f"labels[]={n}")])
            for name in a.get("remove", []):
                _gh("api", "-X", "DELETE", f"repos/{repo}/issues/{a['number']}/labels/{urllib.parse.quote(name, safe='')}")
        elif op in ("close_issue", "reopen_issue"):
            state = "closed" if op == "close_issue" else "open"
            _gh("api", "-X", "PATCH", f"repos/{repo}/issues/{a['number']}", "-f", f"state={state}")
        elif op == "link":
            by_id[a["task"]]["issue"] = a["number"]
        elif op == "link_intake":
            board["intake_issue"] = a["number"]
        print(f"done: {op} {a.get('task') or a.get('name') or a.get('number') or ''}")


def _save_board(board: dict) -> None:
    problems = pm_board.check(board)
    if problems:
        raise SystemExit("写回后看板校验失败：\n" + "\n".join(problems))
    pm_board.BOARD.write_text(json.dumps(board, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 子命令

def cmd_whoami(board: dict) -> int:
    got, want = current_login(), board.get("pm_github_login")
    print(f"gh 账号: {got}\n看板 pm_github_login: {want}")
    return 0 if want and got == want else 1


def cmd_sync(board: dict, apply: bool, offline: bool) -> int:
    if apply and offline:
        raise SystemExit("--apply 与 --offline 不能同时用")
    repo = _repo(board)
    remote_issues, remote_labels = ([], []) if offline else fetch_remote(repo)
    if not offline:
        guard_known_issues(board, remote_issues)
    actions = plan_sync(board, remote_issues, remote_labels)
    for a in actions:
        detail = {k: v for k, v in a.items() if k not in ("op", "body")}
        print(f"{a['op']:14s} {json.dumps(detail, ensure_ascii=False)}")
    print(f"共 {len(actions)} 个动作（{'将执行' if apply else 'dry-run，未执行'}）", file=sys.stderr)
    if apply and actions:
        _require_pm(board)
        apply_actions(board, actions, repo)
        _save_board(board)
        print("看板已写回 issue 编号；记得提交并推送 docs/pm/board.json", file=sys.stderr)
    return 0


def cmd_poll(board: dict, advance: bool) -> int:
    repo = _repo(board)
    pm_login = board.get("pm_github_login")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    since = state.get("since", "1970-01-01T00:00:00Z")
    seen: dict[str, str] = state.get("seen_comments", {})
    comments = _gh_lines("api", "--paginate",
                         f"repos/{repo}/issues/comments?since={since}&sort=updated&direction=asc&per_page=100",
                         "--jq", ".[] | {id, issue_url, user: .user.login, updated_at, body, html_url}")
    latest = since
    for c in comments:
        latest = max(latest, c["updated_at"])
        if seen.get(str(c["id"])) == c["updated_at"] or c["user"] == pm_login:
            continue
        seen[str(c["id"])] = c["updated_at"]
        print(json.dumps(classify_comment(c, board, pm_login), ensure_ascii=False))
    issues_seen: dict[str, str] = state.get("issues", {})
    issues = _gh_lines("api", "--paginate",
                       f"repos/{repo}/issues?since={since}&state=all&sort=updated&direction=asc&per_page=100",
                       "--jq", '.[] | select(has("pull_request") | not) | {number, title, body, state, '
                               "user: .user.login, labels: [.labels[].name], updated_at, html_url}")
    for iss in issues:
        latest = max(latest, iss["updated_at"])
        if issues_seen.get(str(iss["number"])) == iss["updated_at"]:
            continue
        issues_seen[str(iss["number"])] = iss["updated_at"]
        if ev := classify_issue(iss, board):
            print(json.dumps(ev, ensure_ascii=False))
    prs_seen: dict[str, str] = state.get("prs", {})
    prs = _gh_lines("api", "--paginate", f"repos/{repo}/pulls?state=open&per_page=100", "--jq",
                    ".[] | {number, title, user: .user.login, head: .head.ref, "
                    "head_repo: (.head.repo.full_name // null), base: .base.ref, updated_at, html_url}")
    for p in prs:
        if prs_seen.get(str(p["number"])) != p["updated_at"]:
            prs_seen[str(p["number"])] = p["updated_at"]
            print(json.dumps(classify_pr(p, board), ensure_ascii=False))
    if advance:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        keep = dict(sorted(seen.items(), key=lambda kv: kv[1])[-500:])
        STATE.write_text(json.dumps({"since": latest, "seen_comments": keep, "issues": issues_seen,
                                     "prs": prs_seen,
                                     "polled_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        print(f"游标推进到 {latest}", file=sys.stderr)
    return 0


def cmd_label(board: dict, number: int, labels: list[str]) -> int:
    """给需求 issue 打分诊标签（`triage:*`）。"""
    _require_pm(board)
    unknown = [n for n in labels if n not in TRIAGE_LABELS and n != REQUEST_LABEL]
    if unknown:
        raise SystemExit(f"只允许分诊标签 {TRIAGE_LABELS} 与 {REQUEST_LABEL}，收到 {unknown}")
    _gh("api", f"repos/{_repo(board)}/issues/{number}/labels",
        *[x for n in labels for x in ("-f", f"labels[]={n}")])
    print(f"已给 #{number} 打上 {', '.join(labels)}")
    return 0


def cmd_post(board: dict, target: str, body_file: Path) -> int:
    _require_pm(board)
    if target == "intake":
        number = board.get("intake_issue")
    elif target.startswith("#"):
        number = int(target[1:])
    else:
        number = next((t.get("issue") for t in board["tasks"] if t["id"] == target), None)
    if not number:
        raise SystemExit(f"{target} 没有对应的 issue（先 sync --apply）")
    if not body_file.is_file():
        raise SystemExit(f"找不到 {body_file}")
    _gh("api", f"repos/{_repo(board)}/issues/{number}/comments", "-F", f"body=@{body_file}")
    print(f"已评论 #{number}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    s = sub.add_parser("sync")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--offline", action="store_true")
    p = sub.add_parser("poll")
    p.add_argument("--advance", action="store_true")
    c = sub.add_parser("post")
    c.add_argument("target", help="任务 id、'#<issue 号>' 或 'intake'")
    c.add_argument("body_file", type=Path)
    lb = sub.add_parser("label", help="给需求 issue 打 triage:* 标签")
    lb.add_argument("number", type=lambda s: int(s.lstrip("#")))
    lb.add_argument("labels", nargs="+")
    args = ap.parse_args()

    board = pm_board.load()
    if args.cmd == "whoami":
        return cmd_whoami(board)
    if args.cmd == "sync":
        return cmd_sync(board, args.apply, args.offline)
    if args.cmd == "poll":
        return cmd_poll(board, args.advance)
    if args.cmd == "label":
        return cmd_label(board, args.number, args.labels)
    return cmd_post(board, args.target, args.body_file)


if __name__ == "__main__":
    raise SystemExit(main())
