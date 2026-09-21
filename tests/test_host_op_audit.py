"""FMT-01 审计工具自身的 CPU 单测：分类表、拷贝判定、脱敏与 JSON 可解析性。

**只测工具，不声称任何 kernel 执行**：这里不碰 NPU，也不调用任何公共算子入口。
真机清单在 `benchmarks/evidence/host_op_audit/`，由 `python benchmarks/host_op_audit.py --all` 产出。

    pytest tests/test_host_op_audit.py -q
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("host_op_audit", REPO / "benchmarks/host_op_audit.py")
audit_tool = importlib.util.module_from_spec(_spec)
sys.modules["host_op_audit"] = audit_tool  # dataclasses 需要模块已在 sys.modules 里
_spec.loader.exec_module(audit_tool)


@pytest.fixture(autouse=True)
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _run(fn, roots=("tests",)):
    audit = audit_tool.HostOpAudit(roots=roots)
    with audit:
        fn()
    return audit


def _ops(audit, category):
    return {row["op"] for row in audit.rows() if row["category"] == category}


# ----------------------------------------------------------------- 分类表本身

def test_classification_table_is_explicit_and_consistent():
    """表里每个值都是已定义的类别，键不重复，且没有"名字里带 copy 就算拷贝"这类启发式。"""
    categories = {audit_tool.ALLOC, audit_tool.META, audit_tool.CHECK, audit_tool.FORBIDDEN_DTYPE,
                  audit_tool.FORBIDDEN_FORMAT, audit_tool.FORBIDDEN_ARITH}
    assert set(audit_tool.CATEGORY_BY_OP.values()) <= categories
    assert all(key.count(".") == 1 for key in audit_tool.CATEGORY_BY_OP), "键应是 op.overload"
    # 未登记的算子必须落 unclassified，而不是被猜成某一类
    assert audit_tool.classify("aten.some_new_op.default") == audit_tool.UNCLASSIFIED
    assert audit_tool.classify("aten.copy_like_but_not_registered.default") == audit_tool.UNCLASSIFIED
    # torch_npu 自定义算子单列
    assert audit_tool.classify("npu.npu_format_cast.default") == audit_tool.NPU_CUSTOM


@pytest.mark.parametrize("op,expected", [
    ("aten.empty.memory_format", audit_tool.ALLOC),
    ("aten.view.default", audit_tool.META),
    ("aten.isfinite.default", audit_tool.CHECK),
    ("aten._to_copy.default", audit_tool.FORBIDDEN_DTYPE),
    ("aten.clone.default", audit_tool.FORBIDDEN_FORMAT),
    ("aten.cat.default", audit_tool.FORBIDDEN_FORMAT),
    ("aten.mul.Tensor", audit_tool.FORBIDDEN_ARITH),
    ("aten.cumsum.default", audit_tool.FORBIDDEN_ARITH),
])
def test_known_operators_map_to_their_category(op, expected):
    assert audit_tool.classify(op) == expected


# ----------------------------------------------------------------- 正例 / 反例

def test_dtype_cast_is_reported_as_forbidden():
    audit = _run(lambda: torch.ones(4, 8).to(torch.bfloat16))
    assert _ops(audit, audit_tool.FORBIDDEN_DTYPE), audit.rows()
    assert any(row["dtype_changed"] for row in audit.rows())
    assert audit.summary()["verdict"] == "violations"


def test_same_dtype_to_is_not_reported_as_cast():
    """`x.float()` 在已经是 FP32 时是 no-op，不该被报成 dtype 转换。"""
    audit = _run(lambda: torch.ones(4, 8).float())
    assert not _ops(audit, audit_tool.FORBIDDEN_DTYPE), audit.rows()


def test_layout_copy_is_reported_and_metadata_view_is_not():
    def body():
        x = torch.ones(4, 8)
        x.t().contiguous()      # 真的拷贝（aten 层分解成 clone）
        x.view(2, 16)           # 纯元数据
        x.permute(1, 0)         # 纯元数据

    audit = _run(body)
    assert _ops(audit, audit_tool.FORBIDDEN_FORMAT), audit.rows()
    copied = [row for row in audit.rows() if row["copy"]]
    assert copied, audit.rows()
    assert all(row["op"] not in ("aten.view.default", "aten.t.default", "aten.permute.default")
               for row in copied), copied
    assert audit.summary()["verdict"] == "violations"


def test_contiguous_on_contiguous_tensor_is_a_noop():
    """已连续时 `contiguous` 不分配新 storage —— 按 storage 判，不按算子名判。"""
    audit = _run(lambda: torch.ones(4, 8).contiguous())
    assert not [row for row in audit.rows() if row["copy"]], audit.rows()
    assert audit.summary()["verdict"] == "clean"


def test_arithmetic_is_reported():
    audit = _run(lambda: torch.ones(4, 8) * 2)
    assert _ops(audit, audit_tool.FORBIDDEN_ARITH), audit.rows()


def test_allocation_and_metadata_only_function_is_clean():
    def body():
        buffer = torch.empty(4, 8)
        zeros = torch.zeros(4, 8)
        buffer.view(8, 4).unsqueeze(0)
        zeros.detach()

    audit = _run(body)
    summary = audit.summary()
    assert summary["verdict"] == "clean", summary
    assert summary["forbidden_calls"] == 0
    assert set(summary["by_category"]) <= {audit_tool.ALLOC, audit_tool.META}


def test_readonly_check_is_listed_but_not_a_violation():
    """只读校验按**调用点**认：`isfinite(...).all()` 在 aten 层分解成 abs/ne/mul/all，
    单看算子名会误报成算术，所以登记的校验函数名里发出的调用归 readonly_check。"""
    def _validate():                       # 名字在 CHECK_SITE_FUNCTIONS 里
        return bool(torch.isfinite(torch.ones(4, 8)).all())

    audit = _run(_validate)
    assert _ops(audit, audit_tool.CHECK), audit.rows()
    assert audit.summary()["verdict"] == "clean", audit.summary()
    assert audit.summary()["by_category"].get(audit_tool.CHECK)


def test_arithmetic_outside_a_check_site_is_still_a_violation():
    """同样的算式不在校验函数里就照报不误——调用点规则不能变成万能豁免。"""
    audit = _run(lambda: bool(torch.isfinite(torch.ones(4, 8)).all()))
    assert _ops(audit, audit_tool.FORBIDDEN_ARITH), audit.rows()
    assert audit.summary()["verdict"] == "violations"


# ----------------------------------------------------------------- 归因、脱敏、JSON

def test_call_site_is_attributed_to_the_audited_root():
    audit = _run(lambda: torch.ones(4, 8).to(torch.bfloat16), roots=("tests",))
    sites = {row["site_file"] for row in audit.rows()}
    assert any(site.startswith("tests/") for site in sites), sites
    assert all(not pathlib.Path(site).is_absolute() for site in sites), sites


def test_calls_outside_the_audited_root_are_marked():
    audit = _run(lambda: torch.ones(4, 8).to(torch.bfloat16), roots=("ascend_fla/ops",))
    assert {row["site_file"] for row in audit.rows()} == {"<outside audited roots>"}


def test_report_is_json_serialisable_and_carries_no_machine_identity():
    audit = _run(lambda: torch.ones(4, 8).to(torch.bfloat16))
    payload = json.dumps({"rows": audit.rows(), "summary": audit.summary(),
                          "environment": audit_tool.environment()}, ensure_ascii=False)
    json.loads(payload)
    assert "/home/" not in payload and "/root/" not in payload, "路径要脱敏"
    for token in ("ssh", "@", "password", "token"):
        assert token not in payload.lower() or token == "@", payload[:200]


def test_environment_line_has_the_required_fields():
    info = audit_tool.environment()
    for field in ("python", "torch", "audited_commit", "audit_tool_sha256", "timestamp_utc"):
        assert field in info, info
    assert len(info["audit_tool_sha256"]) == 64


def test_every_registered_case_is_addressable():
    """注册表的 id 唯一、可按 id 重跑；构造函数与备注齐全。"""
    assert audit_tool.CASES, "至少要注册一个入口"
    for case_id, (builder, kwargs, _note) in audit_tool.CASES.items():
        assert callable(builder), case_id
        assert isinstance(kwargs, dict), case_id
        assert case_id.count(".") >= 2, f"id 形如 family.entry.path: {case_id}"


def test_verdict_is_not_clean_when_the_entry_failed():
    """入口报错时 verdict 不能读成 clean：没记录 → did-not-run，有记录 → incomplete-*。"""
    audit_tool.CASES["__test_fail__"] = (lambda device, **_: ((lambda: (_ for _ in ()).throw(RuntimeError("boom"))), []), {}, "test")
    audit_tool.CASES["__test_partial__"] = (
        lambda device, **_: ((lambda: (torch.empty(2, 2), (_ for _ in ()).throw(RuntimeError("boom")))), []), {}, "test")
    try:
        empty = audit_tool.audit_case("__test_fail__", device="cpu", roots=("tests",))
        assert empty["error"], empty
        assert empty["summary"]["verdict"] == "did-not-run", empty["summary"]
        partial = audit_tool.audit_case("__test_partial__", device="cpu", roots=("tests",))
        assert partial["error"], partial
        assert partial["summary"]["verdict"].startswith("incomplete-"), partial["summary"]
    finally:
        audit_tool.CASES.pop("__test_fail__", None)
        audit_tool.CASES.pop("__test_partial__", None)


def test_redaction_removes_absolute_paths():
    """证据里不许出现绝对路径：报错文本统一过一遍 redact()。"""
    text = ('PassError: local_mutex: cube needs 34 mutex IDs (maximum 32) at #18 '
            'loc("/root/work/someone-a5/ascriptor/kernels/projects/a5/kda_bwd/kernels/inverse_mm.py:65:15")')
    out = audit_tool.redact(text)
    assert "/root/" not in out and "someone-a5" not in out, out
    assert "inverse_mm.py:65:15" in out, "文件名与行号要保留，只去掉路径前缀"
    assert audit_tool.redact("") == ""
    assert audit_tool.redact(None) is None


def test_site_reclassification_is_recorded_for_review():
    """被"按调用点"改判为只读校验的行要留痕：行上 check_by_site=True，汇总里列出 函数 × 算子 × 次数。"""
    def _validate():                       # 名字在 CHECK_SITE_FUNCTIONS 里
        return bool(torch.isfinite(torch.ones(4, 8)).all())

    audit = _run(_validate)
    changed = [row for row in audit.rows() if row["check_by_site"]]
    assert changed, audit.rows()
    assert all(row["category"] == audit_tool.CHECK for row in changed)
    listed = audit.summary()["reclassified_by_site"]
    assert listed and {item["op"] for item in listed} == {row["op"] for row in changed}
    assert all(item["site_func"] == "_validate" for item in listed), listed
    # 名单里的每个名字都要有理由，供读者审这条机制
    assert all(reason.strip() for reason in audit_tool.CHECK_SITE_FUNCTIONS.values())
    assert audit.summary()["check_site_functions"] == dict(audit_tool.CHECK_SITE_FUNCTIONS)


def test_clean_run_lists_no_reclassification():
    audit = _run(lambda: torch.empty(4, 8).view(8, 4))
    assert audit.summary()["reclassified_by_site"] == []


def test_self_check_passes_on_cpu():
    assert audit_tool.self_check() == 0
