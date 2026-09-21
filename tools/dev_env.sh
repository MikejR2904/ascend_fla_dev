#!/usr/bin/env bash
# 主机侧开发环境：建 .venv（Python 3.11 + torch + pytest），可选装 fla oracle，最后跑一遍主机侧测试。
#
#   tools/dev_env.sh              # 基础环境
#   tools/dev_env.sh --with-fla   # 另装 flash-linear-attention（oracle；macOS 上没有 triton，见 docs/pm/START.md §8）
#
# pyproject 要求 Python >= 3.10，而 macOS 自带的是 3.9 —— 所以用 uv 下载独立的 3.11，不动系统 Python。
# 没装 uv 时，在 $TMPDIR 下用系统 python3 临时引导一个。环境变量 ASCEND_FLA_VENV 可改 venv 位置。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ASCEND_FLA_VENV:-$ROOT/.venv}"
WITH_FLA=0
for arg in "$@"; do
  case "$arg" in
    --with-fla) WITH_FLA=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  UV=uv
else
  BOOT="${TMPDIR:-/tmp}/ascend_fla_uv_boot"
  if [[ ! -x "$BOOT/bin/uv" ]]; then
    python3 -m venv "$BOOT"
    "$BOOT/bin/pip" -q install uv
  fi
  UV="$BOOT/bin/uv"
fi

[[ -x "$VENV/bin/python" ]] || "$UV" venv --python 3.11 "$VENV"
"$UV" pip install --python "$VENV/bin/python" "torch>=2.9.0" pytest pytest-xdist "numpy>=2.2"
if (( WITH_FLA )); then
  "$UV" pip install --python "$VENV/bin/python" flash-linear-attention
fi

"$VENV/bin/python" -m pytest "$ROOT/tests" -q -rs
echo "环境就绪：$VENV"
