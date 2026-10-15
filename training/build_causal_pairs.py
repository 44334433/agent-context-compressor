#!/usr/bin/env python3
"""因果判别头 · 自监督训练数据管线 (拍板②: 零 LLM 机械配对, 2026-10-15)

任务定义: 判别「用户指令 ↔ 动作(tool_call)」是否真实因果配对 (二分类)
正例: 同会话内指令 → 其后 5 条消息内的 assistant tool_call (真实先后)
负例: 指令 ↔ 同会话随机远距(>50条) tool_call (错配)
产出: corpus/causal_pairs/{train,val}.jsonl  每行 {text_a, text_b, label, sid}
底座无关 (text_a/text_b 原文, 训练时现场 tokenize) — bge-m3 与 mmBERT 同数据再 A/B。
"""
import json
import os
import random
import sqlite3
import re

DB = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state.db")
OUT = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state/model-foundation/corpus/causal_pairs")
MAX_PAIRS = 20000
random.seed(42)

INSTR_RE = re.compile(r"^(做|执行|跑|修|查|写|改|加|删|测试|验证|推进|继续|先|再|然后|帮|请|启动|停止|部署|拍板|按)")


def is_instruction(text: str) -> bool:
    t = (text or "").strip()
    return 8 <= len(t) <= 500 and not t.startswith(("[", "```", "{")) and (
        INSTR_RE.match(t) or t.endswith(("吗", "吧", "？", "?", "。")) or "拍板" in t
    )


def tool_brief(msg: dict) -> str:
    tc = msg.get("tool_calls") or "[]"
    try:
        calls = json.loads(tc) if isinstance(tc, str) else tc
        briefs = []
        for c in (calls or [])[:3]:
            fn = (c.get("function") or {}) if isinstance(c, dict) else {}
            args = fn.get("arguments") or ""
            try:
                a = json.loads(args)
                brief = " ".join(str(v) for v in list(a.values())[:2])[:180] if isinstance(a, dict) else str(a)[:180]
            except Exception:
                brief = str(args)[:180]
            briefs.append(f"{fn.get('name','tool')}: {brief}")
        return " | ".join(briefs)[:400]
    except Exception:
        return str(tc)[:200]


def main():
    os.makedirs(OUT, exist_ok=True)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = db.execute(
        """SELECT id, session_id, role, content, tool_calls FROM messages
           WHERE session_id IN (SELECT session_id FROM messages GROUP BY session_id HAVING count(*)>=40)
           ORDER BY id"""
    ).fetchall()
    db.close()
    by_sid = {}
    for r in rows:
        by_sid.setdefault(r[1], []).append(r)
    pairs = []
    for sid, msgs in by_sid.items():
        for i, m in enumerate(msgs):
            if m[2] != "user" or not is_instruction(m[3] or ""):
                continue
            # 正例: 其后 5 条内的 tool_call
            for j in range(i + 1, min(i + 6, len(msgs))):
                if msgs[j][2] == "assistant" and msgs[j][4]:
                    tb = tool_brief({"tool_calls": msgs[j][4]})
                    if tb:
                        pairs.append({"text_a": m[3][:400], "text_b": tb, "label": 1, "sid": sid})
                    break
            # 负例: 远距 tool_call
            if len(msgs) > i + 60:
                j = random.randrange(i + 60, len(msgs))
                if msgs[j][2] == "assistant" and msgs[j][4]:
                    tb = tool_brief({"tool_calls": msgs[j][4]})
                    if tb:
                        pairs.append({"text_a": m[3][:400], "text_b": tb, "label": 0, "sid": sid})
            if len(pairs) >= MAX_PAIRS * 1.2:
                break
        if len(pairs) >= MAX_PAIRS * 1.2:
            break
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    k = int(len(pairs) * 0.92)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs[:k]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs[k:]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pos = sum(1 for p in pairs if p["label"] == 1)
    print(f"扫描 {len(by_sid)} 会话/{len(rows)} 消息 → 配对 {len(pairs)} (正例 {pos}/负例 {len(pairs)-pos}) → train {k}/val {len(pairs)-k}")


if __name__ == "__main__":
    main()
