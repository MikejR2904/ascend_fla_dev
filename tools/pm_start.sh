#!/usr/bin/env bash
# 启动（或重启）PM 会话。必须在本仓主 checkout 的 main 分支、干净工作区上运行，且 gh 已登录 PM bot 账号。
#
#   tools/pm_start.sh                      # 默认开 Remote Control，用户可从别的设备看 PM
#   PM_REMOTE_CONTROL=0 tools/pm_start.sh  # 不开 Remote Control
#
# 同一时间只跑一个 PM 会话：两个 PM 会重复派单、互相覆盖看板。部署步骤见 docs/pm/START.md §2。
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
[[ -x "$PY" ]] || { echo "没有 .venv，先运行 tools/dev_env.sh" >&2; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "找不到 claude 命令" >&2; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "找不到 gh 命令（见 docs/pm/START.md §2）" >&2; exit 1; }

# gh 账号必须是看板里的 PM bot 账号 —— 防止用个人账号发帖与合入
"$PY" tools/pm_github.py whoami || { echo "gh 账号与看板的 pm_github_login 不一致（gh auth switch）" >&2; exit 1; }

git pull --ff-only
"$PY" tools/pm_board.py --check
"$PY" tools/gen_matrix.py --check
"$PY" -m pytest tests -q

args=(--name "$NAME" --append-system-prompt-file docs/pm/prompts/pm.md)
if [[ "${PM_REMOTE_CONTROL:-1}" == "1" ]]; then
  args+=(--remote-control "$NAME")
fi
exec claude "${args[@]}" "开始 PM 值班：按系统提示的开机流程执行，然后每 15 分钟执行一次轮询流程。"
