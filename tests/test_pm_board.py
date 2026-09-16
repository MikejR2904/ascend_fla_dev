"""任务看板校验器的测试 —— 纯标准库，**不需要 NPU，也不需要 torch**。

看板的用处全在"不会把同一个文件同时给两个 agent、不会派出依赖没完成的任务"，所以这里
每条冲突都造一个最小反例，确认 ``check`` 真的会报，而不只是验证正常看板能通过。

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
         "needs": {"npu": npu, "ascriptor": False, "fla": False}, "spec": f"{tid}.md",
         "issue": None, "pr": None, "model": [], "kernel": [], "dtype": []}
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

    def test_no_card_leases_left(self):
        """机器归 agent 管（D-PM-4）：看板不再有租约字段。"""
        self.assertFalse(any("lease" in t for t in pm_board.load()["tasks"]))


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

    def test_npu_task_requires_declared_soc(self):
        found = self.problems(_task("A", "in_progress", npu=True, soc="a2"))
        self.assertTrue(any("声明的 SoC" in p for p in found), found)
        ok = self.problems(_task("B", "in_progress", npu=True, soc="a2", assignee_caps={"socs": ["a2"]}))
        self.assertEqual(ok, [])

    def test_one_task_per_agent(self):
        found = self.problems(_task("A", "in_progress", write_set=["a.py"], assignee="bob"),
                              _task("B", "assigned", write_set=["b.py"], assignee="bob"))
        self.assertTrue(any("同时只接一个任务" in p for p in found), found)

    def test_duplicate_issue_number(self):
        found = self.problems(_task("A", issue=7, write_set=["a.py"]), _task("B", issue=7, write_set=["b.py"]))
        self.assertTrue(any("#7" in p for p in found), found)

    def test_issue_must_be_positive_int(self):
        found = self.problems(_task("A", issue="7"))
        self.assertTrue(any("正整数" in p for p in found), found)

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

    def test_external_request_cannot_be_dispatchable_without_user_approval(self):
        """外部需求不会自己变成可派任务 —— 放行是用户的决定（PROTOCOL §3.9）。"""
        origin = {"kind": "request", "issue": 42, "by": "stranger"}
        found = self.problems(_task("A", "open", origin=origin))
        self.assertTrue(any("只能是 gated" in p for p in found), found)
        gated = self.problems(_task("B", "gated", spec=None, origin=origin))
        self.assertEqual(gated, [])
        approved = self.problems(_task("C", "open", origin={**origin, "approved_by_user": True}))
        self.assertEqual(approved, [])

    def test_request_origin_needs_issue_and_proposer(self):
        found = self.problems(_task("A", "gated", spec=None, origin={"kind": "request"}))
        self.assertEqual(len(found), 2, found)
        self.assertTrue(any("提案 issue 号" in p for p in found), found)
        self.assertTrue(any("提案人" in p for p in found), found)

    def test_unknown_origin_kind(self):
        found = self.problems(_task("A", origin={"kind": "whoever"}))
        self.assertTrue(any("origin.kind" in p for p in found), found)

    def test_ip_address_rejected(self):
        found = self.problems(_task("A", title="run on 10.0.0.12"))
        self.assertTrue(any("IP" in p for p in found), found)


class TestKernelEntries(unittest.TestCase):
    """"这个 kernel 从哪条开始" 必须从依赖图算出来，而且要看传递依赖。"""

    def board(self):
        # K 组：a（无依赖）、b（依赖 a）、c（依赖 x，x 依赖 a —— 传递上仍在 a 之后）、d（独立）
        tasks = [_task("a", kernel=["K"]), _task("b", deps=["a"], kernel=["K"]),
                 _task("x", deps=["a"], kernel=[]), _task("c", deps=["x"], kernel=["K"]),
                 _task("d", kernel=["K"], wave="W-A5", status="gated", spec=None, gate="wave:W-A5")]
        for t in tasks:
            t.setdefault("model", []); t.setdefault("dtype", [])
        return {"tasks": tasks, "axes": {"kernel": ["K"], "model": [], "dtype": []}}

    def test_transitive_dependency_excludes_from_entry(self):
        entries = pm_board.kernel_entries(self.board())["K"]
        self.assertNotIn("c", entries, "c 顺着 x 依赖 a，不该算起点")
        self.assertNotIn("b", entries)
        self.assertEqual(entries[0], "a", "能马上做的 open 任务排第一")
        self.assertIn("d", entries, "d 与 a 互不依赖，是另一个入口")

    def test_entries_rank_actionable_first(self):
        """gated 的入口排在 open 的后面 —— 第一个就是下一步该做的。"""
        self.assertEqual(pm_board.kernel_entries(self.board())["K"], ["a", "d"])

    def test_chain_respects_transitive_order(self):
        chain = [t["id"] for t in pm_board.chain_of(self.board(), "K")]
        self.assertLess(chain.index("a"), chain.index("c"), "c 必须排在 a 之后")
        self.assertLess(chain.index("a"), chain.index("b"))

    def test_unknown_axis_value_rejected(self):
        board = self.board()
        board["tasks"][0]["kernel"] = ["typo_kernel"]
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for t in board["tasks"]:
                if t.get("spec"):
                    (root / t["spec"]).write_text("s", encoding="utf-8")
            found = pm_board.check(board, root=root)
        self.assertTrue(any("未知取值" in p for p in found), found)


class TestReadmeBlock(_Tmp):
    """首页那张 kernel 进展表是生成的；手改会被 --check 拦住（AGENTS.md §8：手写的矩阵必然腐烂）。"""

    def board(self):
        t = _task("A", kernel=["K"], model=[], dtype=["bf16"])
        return {"repo": "o/r", "tasks": [t], "axes": {"kernel": ["K"], "model": [], "dtype": ["bf16"]},
                "kernel_inventory": {"kernels": [
                    {"id": "K", "family": "kda", "track": "agent", "note_zh": "有任务", "note_en": "has tasks"},
                    {"id": "Q", "family": "delta_rule", "track": "none",
                     "note_zh": "尚未排任务", "note_en": "no task scheduled yet"},
                ]}}

    def test_block_lists_kernel_and_links_issue(self):
        board = self.board()
        board["tasks"][0]["issue"] = 7
        block = pm_board.render_readme_block(board)
        self.assertIn("`K`", block)
        self.assertIn("https://github.com/o/r/issues/7", block)
        self.assertIn("★ 起点", block)

    def test_kernels_without_tasks_are_listed_with_reason(self):
        """没有任务的 kernel 也要出现在表里 —— 不列出来，读者会以为它不存在。"""
        block = pm_board.render_readme_block(self.board())
        self.assertIn("`Q`", block)
        self.assertIn("尚未排任务", block)
        self.assertIn("| — |", block, "没有任务的那行进度应当是 —")

    def test_english_variant(self):
        en = pm_board.render_readme_block(self.board(), "en")
        self.assertIn("no task yet", en)
        self.assertIn("★ start", en)
        self.assertNotIn("起点", en)

    def test_both_languages_cover_the_same_kernels(self):
        zh, en = (pm_board.render_readme_block(self.board(), x) for x in ("zh", "en"))
        for kernel in ("`K`", "`Q`"):
            self.assertIn(kernel, zh)
            self.assertIn(kernel, en)

    def test_drift_is_detected_and_fixable(self):
        board = self.board()
        (self.root / "README.md").write_text(
            "# x\n\n" + pm_board.render_readme_block(board) + "\n", encoding="utf-8")
        self.assertIsNone(pm_board.readme_is_current(board, self.root))

        board["tasks"][0]["status"] = "done"          # 看板变了，README 没跟上
        board["tasks"][0]["result"] = {"commits": ["a"]}
        self.assertIn("不一致", pm_board.readme_is_current(board, self.root))

        pm_board.write_readme(board, self.root)        # 重新生成之后就一致了
        self.assertIsNone(pm_board.readme_is_current(board, self.root))

    def test_missing_readme_is_not_an_error(self):
        self.assertIsNone(pm_board.readme_is_current(self.board(), self.root))


class TestReservedPaths(_Tmp):
    """仓主的并行轨道有专属路径；任何任务的写集碰到它都要报。"""

    def problems_with_reserved(self, task, paths):
        (self.root / task["spec"]).write_text("s", encoding="utf-8")
        return pm_board.check({"tasks": [task], "reserved_paths": {"paths": paths}}, root=self.root)

    def test_write_set_touching_reserved_path_is_flagged(self):
        found = self.problems_with_reserved(_task("A", write_set=["ascend_fla/models/**"]),
                                            ["ascend_fla/models/gdn2.py"])
        self.assertTrue(any("并行轨道" in p for p in found), found)

    def test_narrowed_write_set_passes(self):
        self.assertEqual(self.problems_with_reserved(_task("A", write_set=["ascend_fla/models/kimi_linear.py"]),
                                                     ["ascend_fla/models/gdn2.py"]), [])


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
