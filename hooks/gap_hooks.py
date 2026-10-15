#!/usr/bin/env python3
"""gap 检测钩子 · 零模型原型 (检索兜底层: 机械检测未闭环引用→关键词检索→证据回填)

链路: 压缩摘要产出 → 机械检测未闭环引用(gap, 纯正则四类) → 关键词抽提 →
state.db LIKE 预筛候选 → embedding service Qwen3-Embedding-0.6B (:8766 daemon) 相似度重排 →
证据回填块注入摘要顶层。零 LLM 参与; 0.6B 只干检索职能(复用embedding service现役)。
"""
import json
import math
import os
import re
import sqlite3
import urllib.request

STATE_DB = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state.db")
EMBED_URL = os.environ.get("EMBED_SERVICE_URL", "http://127.0.0.1:8766/v1/embed")

# ── gap 检测规则 (纯机械, 可持续扩充) ──
# 快照/系统行豁免 (精度: task_board 快照的数字是元数据不是叙事 gap)
_SNAPSHOT_RE = re.compile(r"task_board#|tasks\[[\w\-./]+\]|⟳v\d+|ack_note|task_board\.json#|HERMES_|session_digest")

GAP_PATTERNS = [
    ("悬空指代", re.compile(r"(该|此|这个|那个|上述|同一个)(文件|路径|方案|版本|问题|错误|脚本|模型|配置)")),
    ("孤儿引用", re.compile(r"(见上|如前述|同上|如上所述|沿用之前|按之前的)")),
    ("无锚点数字", re.compile(r"(提升|下降|达到|耗时|开销)[了到]?\s*[\d.]+\s*(%|倍|s|ms|分钟)")),
    ("未闭环动作", re.compile(r"(待拍板|待办|尚未|未完成|需要确认)")),
]


def extract_gaps(summary_text: str, max_gaps: int = 8) -> list:
    gaps = []
    seen = set()
    for line in summary_text.split("\n"):
        t = line.strip().lstrip("-•* ").strip()
        if len(t) < 12 or _SNAPSHOT_RE.search(t):
            continue
        for kind, pat in GAP_PATTERNS:
            m = pat.search(t)
            if m and t[:80] not in seen:
                seen.add(t[:80])
                gaps.append({"kind": kind, "text": t[:200], "hit": m.group(0)})
                break
        if len(gaps) >= max_gaps:
            break
    return gaps


def _keywords(text: str) -> list:
    words = re.findall(r"[a-zA-Z0-9_\-/\.]{4,}|[\u4e00-\u9fff]{2,6}", text)
    stop = {"这个", "那个", "文件", "路径", "方案", "问题", "上述", "之前", "然后",
            "可以", "已经", "需要", "确认", "提升", "下降", "达到", "耗时", "耗时"}
    return [w for w in words if w not in stop and not w.startswith("http")][:6]


def _embed(texts: list, query: bool = False) -> list:
    payload = json.dumps({"input": texts, "query": query}).encode()
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(EMBED_URL, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["embeddings"]
        except Exception as e:
            last_err = e
            import time
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"embed daemon 不可达: {last_err}")


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / max(na * nb, 1e-9)


def fill_gaps(session_id: str, gaps: list, per_gap: int = 3, min_sim: float = 0.35) -> list:
    db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    out = []
    for g in gaps:
        kws = _keywords(g["text"])
        if not kws:
            continue
        where = " OR ".join(["content LIKE ?"] * len(kws))
        rows = db.execute(
            f"SELECT id, content FROM messages WHERE session_id=? AND ({where}) ORDER BY id DESC LIMIT 15",
            (session_id, *[f"%{k}%" for k in kws]),
        ).fetchall()
        if not rows:
            continue
        try:
            qv = _embed([g["text"]], query=True)[0]
            cvs = _embed([r[1][:400] for r in rows])
            ranked = sorted(zip(rows, [_cos(qv, c) for c in cvs]), key=lambda x: -x[1])[:per_gap]
            ev = [{"id": r[0], "score": round(s, 3), "snippet": r[1][:220]} for r, s in ranked if s >= min_sim]
        except Exception as e:
            # 降级路径: embed daemon 不可达时纯 FTS 序序取前 (fail-open, 检索兜底不因 daemon 死而全灭)
            ev = [{"id": r[0], "score": None, "snippet": r[1][:220], "mode": "fts-only"} for r in rows[:per_gap]]
            if not ev:
                out.append({"gap": g, "error": f"embed+fts 均空: {str(e)[:100]}"})
                continue
        if ev:
            out.append({"gap": g, "evidence": ev})
    db.close()
    return out


def render_evidence_block(evidences: list) -> str:
    if not evidences:
        return ""
    lines = ["【gap 证据回填·检索兜底】"]
    for ev in evidences:
        g = ev["gap"]
        lines.append(f"- gap[{g['kind']}]「{g['hit']}」← {g['text'][:80]}")
        for e in ev.get("evidence", []):
            lines.append(f"    · (msg {e['id']} sim={e['score']}) {e['snippet'][:150]}")
        if ev.get("error"):
            lines.append(f"    ⚠ 检索失败: {ev['error']}")
    return "\n".join(lines)


if __name__ == "__main__":
    demo = """- 决策: 修改该文件后重启服务生效
- 压缩耗时提升了 37.5%，稳定在 20ms
- 渲染引擎沿用之前的复合单次方案"""
    gaps = extract_gaps(demo)
    print(f"自测 gap 检出 {len(gaps)} 条:")
    for g in gaps:
        print(f"  [{g['kind']}] 「{g['hit']}」← {g['text'][:60]}")
