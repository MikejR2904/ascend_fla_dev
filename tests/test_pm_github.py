"""PM 的 GitHub 通道的测试 —— 纯标准库，**不连网络、不调用 gh**。

agent 来自任何账号，所以这里重点测两件事：协议消息能被可靠解析（包括包在代码块里、多行字段），
以及身份校验 —— 非 PM 账号发的 ASSIGN、非 assignee 发的 DONE 都必须带警告，不能被当真。

    pytest tests/test_pm_github.py -v
    python -m unittest tests.test_pm_github -v
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import pm_github  # noqa: E402


def _board(**top):
    tasks = [
        {"id": "A2-01", "title": "定性缺陷", "wave": "W0", "soc": "a2", "priority": "P0", "status": "open",
         "deps": [], "write_set": ["benchmarks/a2/x.py"], "needs": {"npu": False, "ascriptor": True, "fla": False},
         "spec": "A2-01.md", "issue": None, "pr": None, "assignee": None},
        {"id": "A2-03", "title": "派生单元", "wave": "W0", "soc": "a2", "priority": "P1", "status": "in_progress",
         "deps": ["A2-01"], "write_set": ["kernels/projects/a2/**"],
         "needs": {"npu": False, "ascriptor": True, "fla": False},
         "spec": "A2-03.md", "issue": 12, "pr": None, "assignee": "alice", "branch": "task/A2-03"},
        {"id": "OLD", "title": "已完成", "wave": "W0", "soc": "any", "priority": "P2", "status": "done",
         "deps": [], "write_set": ["o.py"], "needs": {"npu": False}, "spec": "OLD.md", "issue": None, "pr": None},
    ]
    board = {"repo": "o/r", "pm_github_login": "fla-pm-bot", "intake_issue": 1, "tasks": tasks}
    board.update(top)
    return board


class TestParseMessage(unittest.TestCase):
    def test_not_a_protocol_message(self):
        self.assertIsNone(pm_github.parse_message("thanks, looks good"))
        self.assertIsNone(pm_github.parse_message("   \n"))

    def test_basic_fields(self):
        msg = pm_github.parse_message("[FLA-PM] APPLY A2-01 from=alice\nsocs: a2\nascriptor: yes\n")
        self.assertEqual((msg["type"], msg["task"], msg["from"]), ("APPLY", "A2-01", "alice"))
        self.assertEqual(msg["fields"], {"socs": "a2", "ascriptor": "yes"})

    def test_fenced_and_multiline(self):
        body = ("Here is my report:\n```\n[FLA-PM] STATUS A2-03 from=alice\nnumbers: |\n"
                "  o rel-L2 3.2e-03\n  final_state 2.3e-03\nblockers: none\n```\n")
        # 正文前面有闲话时，首个非空行不是协议头 —— 不当协议消息
        self.assertIsNone(pm_github.parse_message(body))
        msg = pm_github.parse_message(body.split("\n", 1)[1])
        self.assertEqual(msg["type"], "STATUS")
        self.assertEqual(msg["fields"]["numbers"], "o rel-L2 3.2e-03\nfinal_state 2.3e-03")
        self.assertEqual(msg["fields"]["blockers"], "none")

    def test_malformed_header_and_unknown_type(self):
        self.assertIsNone(pm_github.parse_message("[FLA-PM] APPLY")["type"])
        self.assertIn("未知", pm_github.parse_message("[FLA-PM] HELLO - from=x")["error"])

    def test_dash_task_means_none(self):
        self.assertIsNone(pm_github.parse_message("[FLA-PM] APPLY - from=bob")["task"])


def _comment(body, user, issue=12, cid=100):
    return {"id": cid, "issue_url": f"https://api.github.com/repos/o/r/issues/{issue}", "user": user,
            "updated_at": "2026-09-15T10:00:00Z", "body": body, "html_url": "u"}


class TestClassify(unittest.TestCase):
    def test_pm_message_from_other_account_is_flagged(self):
        ev = pm_github.classify_comment(_comment("[FLA-PM] ASSIGN A2-03 from=fla-pm-bot", "mallory"), _board(), "fla-pm-bot")
        self.assertTrue(any("冒充" in w for w in ev["warnings"]), ev)

    def test_done_from_non_assignee_is_flagged(self):
        ev = pm_github.classify_comment(_comment("[FLA-PM] DONE A2-03 from=bob", "bob"), _board(), "fla-pm-bot")
        self.assertTrue(any("不采信" in w for w in ev["warnings"]), ev)

    def test_done_from_assignee_is_clean(self):
        ev = pm_github.classify_comment(_comment("[FLA-PM] DONE A2-03 from=alice", "alice"), _board(), "fla-pm-bot")
        self.assertEqual((ev["kind"], ev["task_of_issue"], ev["warnings"]), ("message", "A2-03", []))

    def test_from_must_match_author(self):
        ev = pm_github.classify_comment(_comment("[FLA-PM] APPLY any from=alice", "bob", issue=1), _board(), "fla-pm-bot")
        self.assertTrue(ev["intake"])
        self.assertTrue(any("不符" in w for w in ev["warnings"]), ev)

    def test_task_mismatch_with_issue(self):
        ev = pm_github.classify_comment(_comment("[FLA-PM] STATUS A2-01 from=alice", "alice"), _board(), "fla-pm-bot")
        self.assertTrue(any("所在 issue" in w for w in ev["warnings"]), ev)

    def test_plain_comment_passes_through(self):
        ev = pm_github.classify_comment(_comment("any update?", "carol"), _board(), "fla-pm-bot")
        self.assertEqual((ev["kind"], ev["message"]), ("comment", None))

    def test_pr_task_from_title_or_branch(self):
        b = _board()
        ok = pm_github.classify_pr({"number": 5, "title": "[A2-03] port units", "user": "alice", "head": "task/A2-03",
                                    "base": "main"}, b)
        self.assertEqual((ok["task"], ok["warnings"]), ("A2-03", []))
        bad = pm_github.classify_pr({"number": 6, "title": "fix", "user": "bob", "head": "task/A2-03", "base": "dev"}, b)
        self.assertEqual(bad["task"], "A2-03")
        self.assertEqual(len(bad["warnings"]), 2, bad)


def _issue(number=42, body="", labels=(), user="stranger", title="想要 varlen", state="open"):
    return {"number": number, "title": title, "body": body, "state": state, "user": user,
            "labels": list(labels), "updated_at": "2026-09-15T10:00:00Z", "html_url": "u"}


class TestClassifyIssue(unittest.TestCase):
    """任何人都可以提需求（PROTOCOL §3.9）；任务 issue 与申领入口不算需求。"""

    def test_request_by_label(self):
        ev = pm_github.classify_issue(_issue(labels=[pm_github.REQUEST_LABEL]), _board())
        self.assertEqual((ev["kind"], ev["author"], ev["triaged"], ev["linked_task"]),
                         ("request", "stranger", False, None))

    def test_request_by_header_without_label(self):
        ev = pm_github.classify_issue(_issue(body="[FLA-PM] REQUEST - from=stranger\nwhat: varlen\n"), _board())
        self.assertEqual(ev["kind"], "request")
        self.assertEqual(ev["message"]["fields"]["what"], "varlen")

    def test_plain_issue_is_not_a_request(self):
        self.assertEqual(pm_github.classify_issue(_issue(body="how do I build this?"), _board())["kind"], "issue")

    def test_task_issue_and_intake_are_skipped(self):
        board = _board()
        task_issue = _issue(number=12, body=pm_github.TASK_MARKER.format(id="A2-03"))
        self.assertIsNone(pm_github.classify_issue(task_issue, board))
        self.assertIsNone(pm_github.classify_issue(_issue(number=1, body=pm_github.INTAKE_MARKER), board))
        self.assertIsNone(pm_github.classify_issue(_issue(number=1, body="anything"), board))  # intake_issue=1

    def test_triaged_and_linked_task_are_reported(self):
        board = _board()
        board["tasks"][0]["origin"] = {"kind": "request", "issue": 42, "by": "stranger"}
        ev = pm_github.classify_issue(_issue(labels=[pm_github.REQUEST_LABEL, "triage:accepted"]), board)
        self.assertEqual((ev["triaged"], ev["linked_task"]), (True, "A2-01"))

    def test_request_labels_are_created_by_sync(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for name in ("A2-01.md", "A2-03.md", "OLD.md"):
                (root / name).write_text("spec", encoding="utf-8")
            names = {a["name"] for a in pm_github.plan_sync(_board(intake_issue=None), [], [], root)
                     if a["op"] == "create_label"}
        self.assertIn(pm_github.REQUEST_LABEL, names)
        self.assertTrue(set(pm_github.TRIAGE_LABELS) <= names)


class TestPollSkipRule(unittest.TestCase):
    """PM 账号发的消息只跳过 PM 类型的；agent 类型的（APPLY / DONE）必须露出来。

    实测踩到：验证流程时 agent 和 PM 共用同一个账号，APPLY 被当成"PM 自己的评论"整条吞掉。
    """

    def skipped(self, body, author, pm_login="fla-pm-bot"):
        """复刻 cmd_poll 里的跳过判断。"""
        if author != pm_login:
            return False
        parsed = pm_github.parse_message(body)
        return not parsed or parsed.get("type") in pm_github.PM_TYPES or parsed.get("type") is None

    def test_agent_message_from_pm_account_is_not_skipped(self):
        self.assertFalse(self.skipped("[FLA-PM] APPLY A2-05 from=fla-pm-bot", "fla-pm-bot"))
        self.assertFalse(self.skipped("[FLA-PM] DONE A2-05 from=fla-pm-bot", "fla-pm-bot"))

    def test_pm_message_from_pm_account_is_skipped(self):
        self.assertTrue(self.skipped("[FLA-PM] ASSIGN A2-05 from=fla-pm-bot", "fla-pm-bot"))

    def test_plain_comment_from_pm_account_is_skipped(self):
        self.assertTrue(self.skipped("looks good", "fla-pm-bot"))

    def test_anything_from_other_accounts_is_kept(self):
        self.assertFalse(self.skipped("looks good", "alice"))


class TestGuardKnownIssues(unittest.TestCase):
    """看板记过编号的 issue 从列表里消失时必须停下 —— 否则 sync 会重建一遍。

    这不是假想：全新 bot 账号连建 26 个 issue 之后被 GitHub 反滥用过滤，REST 列表端点只剩 PR。
    """

    def test_missing_known_issue_aborts(self):
        board = _board()
        board["tasks"][1]["issue"] = 12
        with self.assertRaises(SystemExit) as cm:
            pm_github.guard_known_issues(board, [{"number": 1}])   # 12 不见了
        self.assertIn("#12", str(cm.exception))

    def test_missing_intake_aborts(self):
        with self.assertRaises(SystemExit) as cm:
            pm_github.guard_known_issues(_board(), [{"number": 12}])   # intake 1 不见了
        self.assertIn("#1", str(cm.exception))

    def test_all_present_passes(self):
        board = _board()
        pm_github.guard_known_issues(board, [{"number": 1}, {"number": 12}])

    def test_board_without_numbers_passes(self):
        board = _board(intake_issue=None)
        board["tasks"][1]["issue"] = None
        pm_github.guard_known_issues(board, [])


class TestPlanSync(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._dir.name)
        for name in ("A2-01.md", "A2-03.md", "OLD.md"):
            (self.root / name).write_text(f"# spec {name}\n", encoding="utf-8")

    def tearDown(self):
        self._dir.cleanup()

    def test_empty_remote_creates_labels_intake_and_open_issues(self):
        board = _board(intake_issue=None)
        actions = pm_github.plan_sync(board, [], [], self.root)
        ops = [a["op"] for a in actions]
        self.assertIn("create_intake", ops)
        created = [a["task"] for a in actions if a["op"] == "create_issue"]
        self.assertEqual(created, ["A2-01", "A2-03"])  # done 的任务不建 issue
        labels = {a["name"] for a in actions if a["op"] == "create_label"}
        self.assertTrue({"fla-pm", "fla-pm:intake", "status:open", "soc:a2", "prio:P0"} <= labels)
        body = next(a["body"] for a in actions if a.get("task") == "A2-01")
        self.assertIn("<!-- fla-pm-task:A2-01 -->", body)
        self.assertIn("# spec A2-01.md", body)
        self.assertIn("@fla-pm-bot", body)

    def _remote(self, board):
        """把看板"已经同步过"的样子造出来。"""
        issues = [{"number": 1, "title": pm_github.INTAKE_TITLE, "body": pm_github.intake_body(board),
                   "state": "open", "labels": ["fla-pm", "fla-pm:intake"]}]
        for n, t in ((11, board["tasks"][0]), (12, board["tasks"][1])):
            t["issue"] = n
        for n, t in ((11, board["tasks"][0]), (12, board["tasks"][1])):
            issues.append({"number": n, "title": pm_github.issue_title(t), "body": pm_github.issue_body(t, board, self.root),
                           "state": "open", "labels": sorted(pm_github.desired_labels(t))})
        labels = sorted({n for i in issues for n in i["labels"]} | pm_github.desired_labels(board["tasks"][2])
                        | {pm_github.REQUEST_LABEL, pm_github.ENTRY_LABEL, *pm_github.TRIAGE_LABELS})
        return issues, labels

    def test_in_sync_means_no_actions(self):
        board = _board()
        issues, labels = self._remote(board)
        self.assertEqual(pm_github.plan_sync(board, issues, labels, self.root), [])

    def test_status_change_updates_labels_and_done_closes(self):
        board = _board()
        issues, labels = self._remote(board)
        board["tasks"][0]["status"] = "done"
        board["tasks"][0]["result"] = {"commits": ["abc"]}
        actions = pm_github.plan_sync(board, issues, labels, self.root)
        upd = next(a for a in actions if a["op"] == "update_issue")
        self.assertEqual((upd["number"], upd["add"], upd["remove"]), (11, ["status:done"], ["status:open"]))
        self.assertIn({"op": "close_issue", "task": "A2-01", "number": 11}, actions)

    def test_abandoned_closed_issue_is_not_resurrected(self):
        """换发布账号后，旧账号那批已关闭的 issue 正文里还有 marker —— 不能被当成"已存在"。"""
        board = _board(intake_issue=None)
        abandoned = [{"number": 4, "title": pm_github.issue_title(board["tasks"][0]),
                      "body": pm_github.issue_body(board["tasks"][0], board, self.root),
                      "state": "closed", "labels": []}]
        actions = pm_github.plan_sync(board, abandoned, [], self.root)
        self.assertIn("A2-01", [a.get("task") for a in actions if a["op"] == "create_issue"])
        self.assertFalse([a for a in actions if a["op"] in ("reopen_issue", "link")])

    def test_closed_issue_still_linked_in_board_is_kept(self):
        board = _board(intake_issue=None)
        board["tasks"][0]["issue"] = 4
        board["tasks"][0].update(status="done", result={"commits": ["a"]})
        closed = [{"number": 4, "title": pm_github.issue_title(board["tasks"][0]),
                   "body": pm_github.issue_body(board["tasks"][0], board, self.root),
                   "state": "closed", "labels": sorted(pm_github.desired_labels(board["tasks"][0]))}]
        self.assertFalse([a for a in pm_github.plan_sync(board, closed, [], self.root)
                          if a["op"] == "create_issue" and a.get("task") == "A2-01"])

    def test_existing_issue_found_by_marker_gets_linked(self):
        board = _board()
        issues, labels = self._remote(board)
        board["tasks"][0]["issue"] = None
        actions = pm_github.plan_sync(board, issues, labels, self.root)
        self.assertIn({"op": "link", "task": "A2-01", "number": 11}, actions)
        self.assertFalse(any(a["op"] == "create_issue" for a in actions))


if __name__ == "__main__":
    unittest.main()
