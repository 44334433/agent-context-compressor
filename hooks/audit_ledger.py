#!/usr/bin/env python3
"""审计者自动化账本: 量化「系统自检发现 vs 用户发现」的比率, 目标让自检覆盖率趋近 1。

目标函数: 用户抓到的缺陷数 ÷ (自检抓到的 + 用户抓到的) → 趋近 0 = 用户真正不用盯着
两个事件源:
  - user_correction : 用户抓到 (纠错信号词扫描 / 用户显式指错)
  - self_catch      : 系统自检抓到 (验收闸/看门狗/生命周期湮灭/锚点覆写)
库: ~/.agent_ctx/state/audit_ledger.db  纯 SQLite, 零 LLM。
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state/audit_ledger.db")

_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,        -- user_correction / self_catch
    detector TEXT NOT NULL,      -- 谁抓到的: user / verify_gate / watchdog / lifecycle / anchor_overwrite / ...
    brief TEXT NOT NULL,         -- 一句话
    ref TEXT DEFAULT ''          -- 指针: session msg id / 日志路径
);
CREATE INDEX IF NOT EXISTS idx_ae_ts ON audit_events(ts);
"""


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.executescript(_DDL)
    return c


def log(source: str, detector: str, brief: str, ref: str = ""):
    """记账入口: source in {user_correction, self_catch}"""
    c = _conn()
    c.execute(
        "INSERT INTO audit_events (ts, source, detector, brief, ref) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), source, detector, brief[:300], ref[:200]),
    )
    c.commit()
    c.close()


def report(days: int = 7) -> dict:
    since = (datetime.now() - timedelta(days=days)).isoformat()
    c = _conn()
    rows = c.execute(
        "SELECT source, detector, count(*) FROM audit_events WHERE ts>=? GROUP BY source, detector",
        (since,),
    ).fetchall()
    total_user = sum(n for s, _, n in rows if s == "user_correction")
    total_self = sum(n for s, _, n in rows if s == "self_catch")
    c.close()
    denom = total_user + total_self
    # 自检覆盖率: 自检抓到的占比 (越高=系统越能自己发现; 用户抓到占比越低越好)
    coverage = (total_self / denom) if denom else None
    return {
        "window_days": days,
        "user_caught": total_user,
        "self_caught": total_self,
        "self_check_coverage": round(coverage, 4) if coverage is not None else None,
        "by_detector": {f"{s}/{d}": n for s, d, n in rows},
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        print(json.dumps(report(int(sys.argv[2]) if len(sys.argv) > 2 else 7), ensure_ascii=False, indent=2))
