"""核验并发保护的回归测试（可控制交错顺序）。

场景：核验先读取数据快照（如 2 元票价），另一请求随后保存新数据
（5 元票价）并使账单失效；过期的核验提交时必须被版本检查拦截
（409 DATA_CHANGED），不得写回过期账单，也不得清掉新账单。

交错顺序通过「关卡」精确控制：把 app.main.verify 包一层，在快照
读取之后、计价提交之前挂起核验线程，主线程完成并发保存再放行。
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import db
from app import main

STATIONS = "科技园, 1\n中心公园, 2"
FARES_2 = "1, 2, 2.00\n2, 1, 2.00"
FARES_5 = "1, 2, 5.00\n2, 1, 5.00"
RECORDS_GOOD = ("A001, 2026-09-18 08:00:00, 科技园, 进\n"
                "A001, 2026-09-18 08:30:00, 中心公园, 出")
# 只有进站没有出站：核验必失败（MISSING_EXIT）
RECORDS_BAD = "A001, 2026-09-18 08:00:00, 科技园, 进"


class _GatedVerify:
    """可控制交错顺序的关卡：快照已读取、提交前挂起核验线程。

    只拦截第一次调用（在途核验线程）；之后的调用直接放行，
    测试主线程可继续执行并发保存与新的核验。
    """

    def __init__(self) -> None:
        self.snapshot_taken = threading.Event()
        self.release = threading.Event()
        self._real = main.verify
        self._lock = threading.Lock()
        self._intercepted = False

    def __call__(self, *args, **kwargs):
        with self._lock:
            intercept = not self._intercepted
            self._intercepted = True
        if intercept:
            # 此时核验已完成一致快照读取（输入 + 数据版本），
            # 尚未计价、更未提交账单 —— 正是并发保存插入的窗口。
            self.snapshot_taken.set()
            if not self.release.wait(timeout=10):
                raise AssertionError("测试放行超时")
        return self._real(*args, **kwargs)


class _VerifyThread(threading.Thread):
    """在独立线程中调用 POST /api/verify 并记录响应。"""

    def __init__(self) -> None:
        super().__init__()
        self.status: int | None = None
        self.payload: dict | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            with TestClient(main.app) as client:
                resp = client.post("/api/verify")
            self.status = resp.status_code
            self.payload = resp.json()
        except BaseException as exc:  # pragma: no cover - 便于诊断
            self.error = exc


class VerifyConcurrencyTest(unittest.TestCase):
    """/api/verify 乐观并发控制（数据版本）的回归用例。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self._tmp.name, "test.db")
        db.init_db()
        self.client_ctx = TestClient(main.app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        db.DB_PATH = self._old_db_path
        self._tmp.cleanup()

    # -- 辅助 ----------------------------------------------------------

    def _put(self, path: str, text: str) -> None:
        resp = self.client.put(path, json={"text": text})
        self.assertEqual(resp.status_code, 200, resp.text)

    def _seed(self, fares: str = FARES_2, records: str = RECORDS_GOOD) -> None:
        self._put("/api/stations", STATIONS)
        self._put("/api/fares", fares)
        self._put("/api/records", records)

    def _bills(self) -> dict:
        resp = self.client.get("/api/bills")
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _version(self) -> int:
        with db.connect() as conn:
            return db.get_data_version(conn)

    def _start_gated_verify(self, gate: _GatedVerify) -> _VerifyThread:
        """以受关卡控制的 verify 启动核验线程，并等待快照读取完成。"""
        self._verify_patch = mock.patch.object(main, "verify", gate)
        self._verify_patch.start()
        self.addCleanup(self._verify_patch.stop)
        t = _VerifyThread()
        t.start()
        self.assertTrue(gate.snapshot_taken.wait(timeout=10),
                        "核验线程未在预期时间内完成快照读取")
        return t

    def _join(self, t: _VerifyThread) -> None:
        t.join(timeout=10)
        self.assertFalse(t.is_alive(), "核验线程未结束")
        if t.error is not None:
            raise t.error

    # -- 用例 ----------------------------------------------------------

    def test_stale_success_verify_rejected(self):
        """读入 2 元票价后另一请求保存 5 元并清空账单：
        旧核验不得写回 2 元账单，应返回 409 DATA_CHANGED。"""
        self._seed(fares=FARES_2)
        base_version = self._version()

        gate = _GatedVerify()
        t = self._start_gated_verify(gate)

        # 并发保存：5 元票价 + 版本递增 + 账单失效（同一事务）
        self._put("/api/fares", FARES_5)
        self.assertEqual(self._version(), base_version + 1)

        gate.release.set()
        self._join(t)

        self.assertEqual(t.status, 409, t.payload)
        self.assertEqual(t.payload["ok"], False)
        self.assertEqual(t.payload["code"], "DATA_CHANGED")

        # 过期核验未写回 2 元账单，核验时间也未被设置
        bills = self._bills()
        self.assertEqual(bills["count"], 0)
        self.assertIsNone(bills["last_verified_at"])

        # 重新核验则按 5 元新票价出账
        resp = self.client.post("/api/verify")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["bill_count"], 1)
        self.assertEqual(body["bills"][0]["base_fare"], 5.0)
        self.assertEqual(body["bills"][0]["charged_fare"], 5.0)

    def test_stale_failed_verify_does_not_clear_new_bills(self):
        """过期的失败核验不得清掉并发新核验写出的账单。"""
        # 先让数据处于「核验必失败」状态（缺出站）
        self._seed(records=RECORDS_BAD)

        gate = _GatedVerify()
        t = self._start_gated_verify(gate)

        # 并发修复记录并成功核验，写出新账单
        self._put("/api/records", RECORDS_GOOD)
        resp = self.client.post("/api/verify")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["bill_count"], 1)
        fresh = self._bills()
        self.assertEqual(fresh["count"], 1)
        self.assertIsNotNone(fresh["last_verified_at"])

        # 放行旧核验：其快照里的数据会核验失败，但版本已变，
        # 必须返回 409 且不得清掉新账单
        gate.release.set()
        self._join(t)

        self.assertEqual(t.status, 409, t.payload)
        self.assertEqual(t.payload["code"], "DATA_CHANGED")

        after = self._bills()
        self.assertEqual(after["count"], 1)
        self.assertEqual(after["items"][0]["charged_fare"], 2.0)
        self.assertEqual(after["last_verified_at"],
                         fresh["last_verified_at"])

    def test_conflict_detected_across_connections(self):
        """版本号持久化在数据库中：另一连接（模拟另一进程）递增版本，
        本进程在途核验同样被拦截（跨进程生效）。"""
        self._seed()
        gate = _GatedVerify()
        t = self._start_gated_verify(gate)

        # 模拟另一进程：不经由本应用连接，直接递增数据版本
        other = sqlite3.connect(db.DB_PATH)
        try:
            other.execute(
                "UPDATE meta SET value = CAST(value AS INTEGER) + 1 "
                "WHERE key = ?", (db.DATA_VERSION_KEY,))
            other.commit()
        finally:
            other.close()

        gate.release.set()
        self._join(t)

        self.assertEqual(t.status, 409, t.payload)
        self.assertEqual(t.payload["code"], "DATA_CHANGED")
        self.assertEqual(self._bills()["count"], 0)

    def test_success_and_failure_without_conflict(self):
        """无并发修改时保留原行为：成功写账单，失败 422 并清账单。"""
        self._seed()
        resp = self.client.post("/api/verify")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["bill_count"], 1)
        self.assertEqual(body["bills"][0]["charged_fare"], 2.0)
        self.assertIsNotNone(self._bills()["last_verified_at"])

        # 数据改坏后核验失败：422 + 定位错误，账单被清空
        self._put("/api/records", RECORDS_BAD)
        resp = self.client.post("/api/verify")
        self.assertEqual(resp.status_code, 422, resp.text)
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["bill_count"], 0)
        self.assertEqual(body["errors"][0]["code"], "MISSING_EXIT")
        bills = self._bills()
        self.assertEqual(bills["count"], 0)
        self.assertIsNone(bills["last_verified_at"])

    def test_saves_bump_version_and_invalidate_bills(self):
        """各保存操作在同一事务内递增数据版本并使账单失效。"""
        self._seed()
        resp = self.client.post("/api/verify")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._bills()["count"], 1)

        v0 = self._version()
        self._put("/api/fares", FARES_5)
        self.assertEqual(self._version(), v0 + 1)
        self.assertEqual(self._bills()["count"], 0)
        self.assertIsNone(self._bills()["last_verified_at"])

        self._put("/api/stations", STATIONS)
        self.assertEqual(self._version(), v0 + 2)
        self._put("/api/records", RECORDS_GOOD)
        self.assertEqual(self._version(), v0 + 3)

        resp = self.client.put("/api/config", json={
            "transfer_minutes": 90, "transfer_discount": 1.0,
            "daily_cap": 10.0, "day_split_hour": 4})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._version(), v0 + 4)

    def test_old_database_backfills_version(self):
        """旧数据库（meta 中无数据版本）启动时自动补齐版本号。"""
        # 模拟旧库：删掉版本号行
        with db.connect() as conn:
            conn.execute("DELETE FROM meta WHERE key = ?",
                         (db.DATA_VERSION_KEY,))
            self.assertIsNone(db.get_meta(conn, db.DATA_VERSION_KEY))

        # 启动初始化（lifespan 调用）应自动补齐
        db.init_db()
        self.assertEqual(self._version(), 0)

        # 补齐后保存与核验照常工作
        self._seed()
        self.assertEqual(self._version(), 3)
        resp = self.client.post("/api/verify")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["bill_count"], 1)


if __name__ == "__main__":
    unittest.main()
