#!/usr/bin/env python3
"""A/B 重建批: v2 教师标注流 → 底座无关 token 标签数据集 (2026-10-15 用户拍板
「用同款数据集对 mmBERT-base 训练 token 级序列标注, 比较它与 m3 哪个更强」)

输入: distill_pool/annotated_llmlingua2_stream.jsonl (orig_text/extracted_text, 抽取式=子序列)
输出: corpus/ab_dataset/{train,val}.jsonl  每行 {text, kept_spans, mid, sid}
方法: 字符级 SequenceMatcher 对齐 (提取出的文本是原文保留字符子集, 匹配块=保留区间),
      200 字符切块与现役 v1 规格一致; 消息级 90/10 切分防泄漏。
"""
import json
import os
import re
from difflib import SequenceMatcher

SRC = os.path.expanduser(
    "~/.agent_ctx/state/model-foundation/corpus/distill_pool/annotated_llmlingua2_stream.jsonl"
)
SRC2 = os.path.expanduser(  # v2 扩训: 合并短批教师标注 (2026-10-15 拍板「现在扩训」)
    "~/.agent_ctx/state/model-foundation/corpus/distill_pool/annotated_llmlingua2_short.jsonl"
)
OUT_DIR = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state/model-foundation/corpus/ab_dataset")
CHUNK = 200


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _iter_rows():
    for src in (SRC, SRC2):
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, n_in = [], 0
    for d in _iter_rows():
        if True:
            orig, ext = norm(d.get("orig_text", "")), norm(d.get("extracted_text", ""))
            if not orig or not ext or len(orig) < 40:
                continue
            n_in += 1
            for ci in range(0, len(orig), CHUNK):
                piece = orig[ci : ci + CHUNK]
                if len(piece) < 40:
                    break
                sm = SequenceMatcher(None, piece, ext, autojunk=False)
                spans = [
                    [m.a, m.a + m.size]
                    for m in sm.get_matching_blocks()
                    if m.size >= 2 and m.size != len(piece)
                ]
                if not spans:
                    spans = [[0, len(piece)]]
                rows.append(
                    {
                        "text": piece,
                        "kept_spans": spans,
                        "mid": d.get("mid", ""),
                        "sid": d.get("sid", ""),
                    }
                )
    k = int(len(rows) * 0.9)
    # 消息级切分防泄漏: 按 (sid,mid) 分组边界切
    groups, seen, order = {}, set(), []
    for i, r in enumerate(rows):
        key = (r["sid"], r["mid"])
        if key not in seen:
            seen.add(key)
            order.append(key)
        groups.setdefault(key, []).append(i)
    cut = order[: int(len(order) * 0.9)]
    cut_set = set(cut)
    tr = [rows[i] for key in cut for i in groups[key]]
    va = [rows[i] for key in order for i in groups[key] if key not in cut_set]
    for name, data in [("train", tr), ("val", va)]:
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"输入 {n_in} 条教师标注 → 块 {len(rows)} (train {len(tr)} / val {len(va)})")


if __name__ == "__main__":
    build()
