#!/usr/bin/env bash
# 启动（或重启）PM 会话。必须在本仓主 checkout 的 main 分支、干净工作区上运行。
#
#   tools/pm_start.sh                      # 默认开 Remote Control，其他机器上的 agent 与用户手机可达
#   PM_REMOTE_CONTROL=0 tools/pm_start.sh  # 只在本机用
#
# 同名会话只能有一个：已有 PM 在跑时不要再起，否则名字会被加后缀、agent 找错人。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NAME="${PM_NAME:-ascend-fla-dev-management-team}"
PY="$ROOT/.venv/bin/python"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" ]]; then
  echo "PM 必须在 main 上启动（当前 ${branch}）" >&2; exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "工作区不干净，先提交或清理：" >&2; git status --short >&2; exit 1
fi
if [[ "$(git rev-parse --git-dir)" != "$(git rev-parse --git-common-dir)" ]]; then
  echo "这是一个 worktree；PM 要在主 checkout 上运行" >&2; exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "没有 .venv，先运行 tools/dev_env.sh" >&2; exit 1
fi
command -v claude >/dev/null 2>&1 || { echo "找不到 claude 命令" >&2; exit 1; }

"$PY" tools/pm_board.py --check
"$PY" tools/gen_matrix.py --check
"$PY" -m pytest tests -q

args=(--name "$NAME" --append-system-prompt-file docs/pm/prompts/pm.md)
if [[ "${PM_REMOTE_CONTROL:-1}" == "1" ]]; then
  args+=(--remote-control "$NAME")
fi
exec claude "${args[@]}" "开始 PM 值班：按系统提示里的开机流程执行，然后等待 agent 的 APPLY。"
