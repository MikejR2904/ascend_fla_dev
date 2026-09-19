#!/usr/bin/env bash
# agent 上线助手：检查 clone 与环境，按声明的能力生成 APPLY 评论（任何模型 / 任何账号通用）。
#
#   tools/agent_setup.sh --login alice --agent "some-model" --task any
#   tools/agent_setup.sh --login bob --agent human --task A2-05 --fla
#   tools/agent_setup.sh --login carol --agent "some-model" --socs a2 \
#       --soc-details "a2: CANN 9.x，内置算子包含 ascend910b" --ascriptor "<library 修订号>" --fla
#
# 选项：
#   --login L         你的 GitHub 账号（必填）
#   --agent A         你是什么：模型 / 工具名，或 human（必填）
#   --task ID|any     申领的任务；any 表示在申领入口 issue 由 PM 派
#   --socs a2,a3      你能用的 SoC 真机。所有任务都要真机验证（D-PM-34），不写就接不到任务
#   --soc-details S   CANN 版本、内置算子包覆盖（不要写主机信息）
#   --ascriptor REV   有 ascriptor workspace 时写 library 修订号
#   --fla             装了 fla
#   --availability S  在线方式，例如 "异步，每天约 4 小时"
#   --post            用 gh（你自己的账号）直接把 APPLY 发出去
#
# 只打印、不改仓库。仓库公开：不要在任何参数里写主机名 / IP / 路径。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
usage() { sed -n '2,23p' "$0" >&2; exit 2; }

LOGIN=""; AGENT=""; TASK="any"; SOCS="none"; SOC_DETAILS="none"; ASCRIPTOR="no"; FLA="no"
AVAIL="未说明"; POST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --login) LOGIN="${2:?}"; shift 2 ;;
    --agent) AGENT="${2:?}"; shift 2 ;;
    --task) TASK="${2:?}"; shift 2 ;;
    --socs) SOCS="${2:?}"; shift 2 ;;
    --soc-details) SOC_DETAILS="${2:?}"; shift 2 ;;
    --ascriptor) ASCRIPTOR="yes ${2:?}"; shift 2 ;;
    --fla) FLA="yes"; shift ;;
    --availability) AVAIL="${2:?}"; shift 2 ;;
    --post) POST=1; shift ;;
    -h|--help) usage ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done
[[ -n "$LOGIN" && -n "$AGENT" ]] || usage

PY="$ROOT/.venv/bin/python"
if [[ -x "$PY" ]]; then
  PYINFO="$("$PY" -c 'import sys, torch; print(f"{sys.version.split()[0]} / torch {torch.__version__}")' 2>/dev/null || echo "venv 不完整")"
  "$PY" -c 'import pytest' 2>/dev/null && PYINFO="$PYINFO / pytest 可用"
else
  echo "提示：还没有 .venv，先运行 tools/dev_env.sh" >&2
  PYINFO="未就绪"
fi

UPSTREAM="ddddwee1/ascend_fla_dev"
# 交付用的 fork：先看名为 fork 的 remote，再退回 origin。
# （origin 可能指向上游本仓 —— 本机就是这样，只看 origin 会把 fork 字段填成上游。）
slug() { git remote get-url "$1" 2>/dev/null | sed -E 's#^(https://github.com/|git@github.com:)##; s#\.git$##'; }
FORK="$(slug fork)"
[[ -n "$FORK" ]] || FORK="$(slug origin)"
if [[ -z "$FORK" || "$FORK" == "$UPSTREAM" ]]; then
  echo "提示：没找到你的 fork（fork 与 origin 两个 remote 都指向上游或不存在）。" >&2
  echo "      先 fork 本仓，然后 git remote add fork https://github.com/<你>/ascend_fla_dev.git" >&2
  FORK="未设置"
fi
git remote get-url upstream >/dev/null 2>&1 || echo "提示：git remote add upstream https://github.com/${UPSTREAM}.git" >&2

BOARD_INFO="$("$PY" - <<'EOF' 2>/dev/null || true
import json; b = json.load(open("docs/pm/board.json", encoding="utf-8"))
print(b.get("pm_github_login") or "", b.get("intake_issue") or "")
EOF
)"
read -r PM_LOGIN INTAKE <<<"${BOARD_INFO:-}" || true

MSG="$(cat <<EOF
[FLA-PM] APPLY ${TASK} from=${LOGIN}
agent: ${AGENT}
socs: ${SOCS}
soc_details: ${SOC_DETAILS}
ascriptor: ${ASCRIPTOR}
fla: ${FLA}
python: ${PYINFO}
fork: ${FORK:-未设置}
availability: ${AVAIL}
EOF
)"
echo "$MSG"
echo >&2
echo "PM 账号：${PM_LOGIN:-（看板尚未填写 pm_github_login）}；只采信它发的 ASSIGN。" >&2
if [[ "$TASK" == "any" ]]; then
  echo "贴到申领入口 issue：${INTAKE:+#$INTAKE}（标签 fla-pm:intake）" >&2
else
  echo "贴到任务 ${TASK} 的 issue 下（标题以 [${TASK}] 开头）" >&2
fi

if (( POST )); then
  command -v gh >/dev/null 2>&1 || { echo "--post 需要 gh" >&2; exit 1; }
  who="$(gh api user --jq .login)"
  [[ "$who" == "$LOGIN" ]] || { echo "gh 当前账号是 ${who}，与 --login ${LOGIN} 不一致" >&2; exit 1; }
  if [[ "$TASK" == "any" ]]; then
    number="$INTAKE"
  else
    number="$(gh issue list -R "$UPSTREAM" --label fla-pm --state open --search "[${TASK}] in:title" --json number,title \
      --jq "map(select(.title | startswith(\"[${TASK}]\")))[0].number")"
  fi
  [[ -n "$number" && "$number" != "null" ]] || { echo "找不到目标 issue" >&2; exit 1; }
  gh issue comment "$number" -R "$UPSTREAM" --body "$MSG"
fi
