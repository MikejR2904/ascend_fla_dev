"""KDA 的门控测试 —— 纯 host 侧，**不需要 NPU**。

这些检查必须在没有硬件的机器上也能跑：它们守的正是"不满足就报错"这条规则
（AGENTS.md §7），而其中 block_dim 那一条如果漏掉，后果是在硬件 barrier 上
死锁而不是报错 —— 那是最不该靠有卡的机器才能发现的问题。

    pytest tests/test_kda_gating.py -v
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.ops.kda.chunk import SUPPORTED_BLOCK_DIM, _check  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("bad", [0, -1, 5, 8, 28, 32])
def test_block_dim_outside_contract_is_rejected(bad):
    """block_dim 门控在碰任何张量之前就报错。"""
    with pytest.raises(ValueError, match="block_dim 只支持"):
        _check(None, None, None, None, None, None, bad)


@pytest.mark.parametrize("ok", SUPPORTED_BLOCK_DIM)
def test_block_dim_inside_contract_passes_that_check(ok):
    """契约内的值不能被 block_dim 这一条挡住（后面的张量检查报错是预期的）。"""
    with pytest.raises((ValueError, AttributeError)) as e:
        _check(None, None, None, None, None, None, ok)
    assert "block_dim" not in str(e.value)


def test_supported_block_dim_matches_ascriptor_contract():
    """别让这个常量和 ascriptor 契约各自漂移。

    契约不在本仓内（只读引用，见 AGENTS.md §3），所以本机没有 kernels 树时 skip；
    有的时候则必须一致 —— 这是"矩阵是唯一真相源"的同款约束。
    """
    from ascend_fla.ops.kda.chunk import _kernels_root

    try:
        contract = _kernels_root() / "contract.json"
    except FileNotFoundError:
        pytest.skip("本机没有 ascriptor kernels 树")
    if not contract.is_file():
        pytest.skip(f"{contract} 不存在")
    declared = json.loads(contract.read_text(encoding="utf-8"))["domain"]["block_dim"]
    assert tuple(declared) == SUPPORTED_BLOCK_DIM, (
        f"contract.json 声明 {declared}，chunk.py 写的是 {list(SUPPORTED_BLOCK_DIM)}"
    )
