#!/usr/bin/env python3
"""融合实验 (拍板①): 教师版 LoRA + 锚点覆写 → val 上测「压缩率↑ + 锚点召回 100%」
三指标对比: 纯教师版 / 融合版(教师+锚点覆写) / v1 现役
锚点召回定义: val 样本中命中锚点正则的实体, 覆写后在预测保留 token 中的存活率
"""
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, os.path.expanduser("~/.hermes/scripts"))
from train_ab_lora import SpanDataset, collate

# 锚点正则族 (复用压缩管线真源)
sys.path.insert(0, "~/.hermes/hermes-agent")
from agent.compression_ml_pipeline import RE_FILE_PATH, RE_LINE_NUM

RE_NUM_METRIC = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|ms|s\b|tok|x\b|GB|MB|GiB)")


def anchor_char_spans(text: str):
    """锚点实体的字符区间集合 (路径/行号/数字指标)"""
    spans = []
    for pat in (RE_FILE_PATH, RE_LINE_NUM, RE_NUM_METRIC):
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def run(model_path: str, tag: str, anchor_overwrite: bool = False):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2).eval()
    ds = SpanDataset(VAL, tok)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=lambda b: collate(b, tok.pad_token_id))
    tp = fp = fn = pred_keep = tot = 0
    # 锚点召回统计: 锚点字符区间的 token 是否被预测保留
    anchor_tot = anchor_saved = 0
    for bi, batch in enumerate(loader):
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = out.logits.argmax(-1)
        labels = batch["labels"]
        for bi_row in range(preds.shape[0]):
            row = ds.rows[bi * 8 + bi_row] if bi * 8 + bi_row < len(ds.rows) else None
        # 向量化统计 (与 eval 口径一致)
        m = labels != -100
        p, l = preds[m], labels[m]
        if anchor_overwrite:
            # 覆写: 预测=0 但该 token 字符区间命中锚点 → 改 1
            idx = m.nonzero().flatten()
            for k, gi in enumerate(idx.tolist()):
                row_i = bi * 8 + gi // 256 if False else None
            # 逐样本覆写 (正确实现: 按样本遍历)
            new_p = preds.clone()
            off = 0
            for r_i in range(preds.shape[0]):
                n_real = int((labels[r_i] != -100).sum())
                # 该样本原文与 tokenizer 对齐: 重 tokenize 拿 offsets
                enc = tok(ds.rows[0]["text"] if False else "", return_offsets_mapping=True)
                off += 256
            # 简化: 覆写逻辑放逐样本循环 (见下方逐样本分支)
        # 逐样本精确统计 (含覆写)
        if anchor_overwrite:
            continue  # 覆写走下面的逐样本分支
        pred_keep += int((p == 1).sum())
        tot += int(m.sum())
        tp += int(((p == 1) & (l == 1)).sum())
        fp += int(((p == 1) & (l == 0)).sum())
        fn += int(((p == 0) & (l == 1)).sum())
    if not anchor_overwrite:
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"{tag}: F1={f1:.3f} 保留率={pred_keep/tot:.2%} (P={tp/max(tp+fp,1):.3f} R={tp/max(tp+fn,1):.3f})")
        return {"f1": round(f1, 4), "keep_rate": round(pred_keep / tot, 4)}


VAL = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/corpus/ab_dataset/val.jsonl")

# 逐样本版 (覆写需要 token↔字符对齐, 按样本处理)
def run_persample(model_path: str, tag: str, anchor_overwrite: bool):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2).eval()
    ds = SpanDataset(VAL, tok)
    tp = fp = fn = pred_keep = tot = 0
    a_tot = a_saved = 0
    for i in range(len(ds)):
        item = ds[i]
        row = ds.rows[i]
        with torch.no_grad():
            logits = model(input_ids=item["input_ids"].unsqueeze(0), attention_mask=item["attention_mask"].unsqueeze(0)).logits[0]
        enc = tok(row["text"], truncation=True, max_length=256, return_offsets_mapping=True)
        preds = logits.argmax(-1).tolist()
        labels = item["labels"].tolist()
        spans = anchor_char_spans(row["text"]) if anchor_overwrite else []
        for k, (lab, pr) in enumerate(zip(labels, preds)):
            if lab == -100:
                continue
            s, e = enc["offset_mapping"][k]
            is_anchor = any(s >= a and e <= b for a, b in spans)
            if anchor_overwrite and pr == 0 and is_anchor:
                pr = 1  # 覆写: 锚点强制保留
            if is_anchor and not anchor_overwrite:
                pass
            if anchor_overwrite:
                a_tot += 1 if is_anchor else 0
                a_saved += 1 if (is_anchor and pr == 1) else 0
            pred_keep += pr
            tot += 1
            if pr == 1 and lab == 1: tp += 1
            elif pr == 1 and lab == 0: fp += 1
            elif pr == 0 and lab == 1: fn += 1
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    a_rec = (a_saved / a_tot) if anchor_overwrite and a_tot else None
    ar = f" 锚点召回={a_rec:.2%}({a_saved}/{a_tot})" if a_rec is not None else ""
    print(f"{tag}: F1={f1:.3f} 保留率={pred_keep/tot:.2%}{ar}")
    return {"f1": round(f1, 4), "keep": round(pred_keep / tot, 4), "anchor_recall": round(a_rec, 4) if a_rec is not None else None}


if __name__ == "__main__":
    MMBERT = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/hermes-lingua-mmbert")
    BGE_AB = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/hermes-lingua-bge-ab")
    V1 = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/hermes-lingua-bge")
    print("== 教师版纯推理 (基线) ==")
    run_persample(BGE_AB, "教师版bge-ab", anchor_overwrite=False)
    print("== 融合版: 教师版+锚点覆写 ==")
    run_persample(BGE_AB, "融合版", anchor_overwrite=True)
    print("== v1 现役对照 ==")
    run(V1, "v1现役", anchor_overwrite=False)
