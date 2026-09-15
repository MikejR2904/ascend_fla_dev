#!/usr/bin/env bash
# 启动一个执行 agent 会话：读文档 → 找到 PM → 发 APPLY。在本仓主 checkout 上运行。
#
#   tools/agent_start.sh agent-a --ascriptor
#   tools/agent_start.sh agent-b --fla
#   tools/agent_start.sh agent-c --socs a2 --ascriptor --fla
#
# 选项：
#   --socs a2,a3   能用的 SoC 机器（按 machine_specs.md）；纯主机侧不写
#   --ascriptor    有 ascriptor workspace
#   --fla          装了 fla（tools/dev_env.sh --with-fla）
#   --no-remote    不开 Remote Control（只有 PM 在同一台机器上时才可以）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 || "$1" == -* ]]; then
  sed -n '2,12p' "$0" >&2; exit 2
fi
NAME="$1"; shift
SOCS="none"; ASCRIPTOR="no"; FLA="no"; REMOTE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --socs) SOCS="${2:?--socs 需要参数}"; shift 2 ;;
    --ascriptor) ASCRIPTOR="yes"; shift ;;
    --fla) FLA="yes"; shift ;;
    --no-remote) REMOTE=0; shift ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [[ "$(git rev-parse --git-dir)" != "$(git rev-parse --git-common-dir)" ]]; then
  echo "请在主 checkout 上启动 agent（任务 worktree 由 agent 拿到 ASSIGN 后自己建）" >&2; exit 1
fi
[[ -x "$ROOT/.venv/bin/python" ]] || { echo "没有 .venv，先运行 tools/dev_env.sh" >&2; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "找不到 claude 命令" >&2; exit 1; }
git fetch -q origin 2>/dev/null || true

args=(--name "$NAME" --append-system-prompt-file docs/pm/prompts/agent.md)
if (( REMOTE )); then
  args+=(--remote-control "$NAME")
fi
exec claude "${args[@]}" \
  "你是执行 agent ${NAME}。能力：socs=${SOCS} ascriptor=${ASCRIPTOR} fla=${FLA}。按系统提示的开机流程：读文档、确认环境、找到 PM，然后发 APPLY。"
