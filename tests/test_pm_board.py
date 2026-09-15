"""任务看板校验器的测试 —— 纯标准库，**不需要 NPU，也不需要 torch**。

看板的用处全在"不会把同一个文件或同一张卡同时给两个 agent"，所以这里每条冲突
都造一个最小反例，确认 ``check`` 真的会报，而不只是验证正常看板能通过。

    pytest tests/test_pm_board.py -v
    python -m unittest tests.test_pm_board -v
"""
from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import pm_board  # noqa: E402


def _task(tid, status="open", deps=(), write_set=("x.py",), npu=False, soc="any", **extra):
    t = {"id": tid, "title": tid, "wave": "W0", "soc": soc, "priority": "P1", "status": status,
         "deps": list(deps), "write_set": list(write_set),
         "needs": {"npu": npu, "ascriptor": False, "fla": False}, "spec": f"{tid}.md"}
    if status in pm_board.IN_FLIGHT:
        t.update(assignee="agent-" + tid, branch="task/" + tid)
    if status == "gated":
        t["gate"] = "user-decision"
    if status == "done":
        t["result"] = {"commits": ["abc1234"]}
    t.update(extra)
    return t


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def problems(self, *tasks):
        for t in tasks:
            if t.get("spec"):
                (self.root / t["spec"]).write_text("spec", encoding="utf-8")
        return pm_board.check({"tasks": list(tasks)}, root=self.root)


class TestRepoBoard(unittest.TestCase):
    def test_committed_board_is_valid(self):
        self.assertEqual(pm_board.check(pm_board.load()), [])

    def test_render_mentions_every_task(self):
        board = pm_board.load()
        text = pm_board.render(board)
        for t in board["tasks"]:
            self.assertIn(f"| {t['id']} |", text)


class TestCheck(_Tmp):
    def test_minimal_board_passes(self):
        self.assertEqual(self.problems(_task("A"), _task("B", deps=["A"])), [])

    def test_write_set_overlap_between_in_flight_tasks(self):
        found = self.problems(_task("A", "in_progress", write_set=["pkg/mod.py"]),
                              _task("B", "assigned", write_set=["pkg/**"]))
        self.assertTrue(any("写集冲突" in p for p in found), found)

    def test_overlap_ignored_when_one_task_not_in_flight(self):
        self.assertEqual(self.problems(_task("A", "in_progress", write_set=["pkg/mod.py"]),
                                       _task("B", "open", write_set=["pkg/**"])), [])

    def test_section_fragment_still_counts_as_same_file(self):
        self.assertTrue(pm_board.paths_overlap("AGENTS.md#§2", "AGENTS.md#§9"))
        self.assertFalse(pm_board.paths_overlap("pkg/a.py", "pkg/ab.py"))

    def test_double_leased_card(self):
        lease = {"soc": "a2", "card": 4}
        found = self.problems(_task("A", "in_progress", write_set=["a.py"], npu=True, soc="a2", lease=lease),
                              _task("B", "in_progress", write_set=["b.py"], npu=True, soc="a2", lease=dict(lease)))
        self.assertTrue(any("卡重复租出" in p for p in found), found)

    def test_npu_task_without_lease(self):
        found = self.problems(_task("A", "in_progress", npu=True, soc="a2"))
        self.assertTrue(any("租约" in p for p in found), found)

    def test_npu_task_must_name_soc(self):
        found = self.problems(_task("A", npu=True, soc="any"))
        self.assertTrue(any("SoC" in p for p in found), found)

    def test_dependency_cycle(self):
        found = self.problems(_task("A", deps=["B"]), _task("B", deps=["A"]))
        self.assertTrue(any("成环" in p for p in found), found)

    def test_unknown_dependency(self):
        found = self.problems(_task("A", deps=["nope"]))
        self.assertTrue(any("不存在的任务" in p for p in found), found)

    def test_in_flight_with_undone_dependency(self):
        found = self.problems(_task("A"), _task("B", "in_progress", deps=["A"], write_set=["b.py"]))
        self.assertTrue(any("未完成" in p for p in found), found)

    def test_gated_requires_gate_reason(self):
        t = _task("A", "gated", spec=None)
        del t["gate"]
        found = self.problems(t)
        self.assertTrue(any("gate" in p for p in found), found)

    def test_open_task_requires_existing_spec(self):
        found = pm_board.check({"tasks": [_task("A")]}, root=self.root)  # 故意不建 spec 文件
        self.assertTrue(any("spec 文件不存在" in p for p in found), found)

    def test_done_requires_commits(self):
        t = _task("A", "done")
        t["result"] = {}
        found = self.problems(t)
        self.assertTrue(any("commits" in p for p in found), found)

    def test_unknown_status(self):
        found = self.problems(_task("A", "wip"))
        self.assertTrue(any("词汇表" in p for p in found), found)

    def test_ip_address_rejected(self):
        found = self.problems(_task("A", title="run on 10.0.0.12"))
        self.assertTrue(any("IP" in p for p in found), found)


class TestNextCandidates(unittest.TestCase):
    def setUp(self):
        self.board = {"tasks": [
            _task("host", write_set=["h.py"]),
            _task("dev", write_set=["d.py"], npu=True, soc="a2"),
            _task("waiting", deps=["host"], write_set=["w.py"]),
            _task("busy", "in_progress", write_set=["shared/**"]),
            _task("clash", write_set=["shared/x.py"]),
            _task("gated", "gated", write_set=["g.py"]),
        ]}
        self.board["tasks"][1]["needs"]["ascriptor"] = True

    def ids(self, **kw):
        return [t["id"] for t in pm_board.next_candidates(self.board, **kw)]

    def test_host_only_agent_gets_no_device_task(self):
        self.assertEqual(self.ids(socs=set(), ascriptor=True, fla=False), ["host"])

    def test_device_agent_needs_matching_soc_and_ascriptor(self):
        self.assertEqual(self.ids(socs={"a2"}, ascriptor=True, fla=False), ["host", "dev"])
        self.assertEqual(self.ids(socs={"a5"}, ascriptor=True, fla=False), ["host"])
        self.assertEqual(self.ids(socs={"a2"}, ascriptor=False, fla=False), ["host"])

    def test_dependency_done_unlocks(self):
        board = copy.deepcopy(self.board)
        board["tasks"][0].update(status="done", result={"commits": ["abc"]})
        self.assertIn("waiting", [t["id"] for t in pm_board.next_candidates(board, set(), False, False)])


if __name__ == "__main__":
    unittest.main()
