#!/usr/bin/env python3
"""v2 分层采样 manifest 生成 (拍板③: 短5K + 中长1.5万 + 长5K = 2.5万)
源: state.db messages 全量 (排除系统/空/敏感样本) → candidates_manifest_v2.jsonl
敏感闸: 含密码/邮箱/key 样本排除 (沿用短批纪律)
"""
import json
import os
import random
import re
import sqlite3

DB = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state.db")
OUT = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/corpus/distill_pool/candidates_manifest_v2.jsonl")
SENSITIVE = re.compile(r"password|passwd|api[_-]?key|secret|token=|Bearer|@gmail|@qq\.com|@163\.com", re.I)
TARGETS = {"short": 5000, "mid": 15000, "long": 5000}  # <200 / 200-2000 / >2000 字符


def bucket(n):
    return "short" if n < 200 else ("mid" if n <= 2000 else "long")


def main():
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = db.execute(
        """SELECT id, session_id, role, content, length(content) FROM messages
           WHERE role IN ('user','assistant') AND length(content)>=8
           AND content NOT LIKE '%%[IMPORTANT%%' AND content NOT LIKE '%%[ARC%%'"""
    ).fetchall()
    db.close()
    pools = {"short": [], "mid": [], "long": []}
    for rid, sid, role, content, ln in rows:
        if SENSITIVE.search(content or ""):
            continue
        pools[bucket(ln)].append({"mid": rid, "sid": sid, "role": role, "text": content, "src": "v2_stratified"})
    random.seed(7)
    manifest = []
    for b, target in TARGETS.items():
        pool = pools[b]
        take = min(target, len(pool))
        manifest += random.sample(pool, take)
        print(f"{b}: 池 {len(pool)} → 取 {take}")
    with open(OUT, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"总 manifest: {len(manifest)} → {OUT}")


if __name__ == "__main__":
    main()
