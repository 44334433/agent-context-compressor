#!/usr/bin/env python3
"""保留率对比: v1 机械版 vs 教师版 — 验证「v1 靠保守 KEEP 策略在对齐 val 上讨巧」假说"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys_p = os.path.expanduser("~/.hermes/scripts")
import sys
sys.path.insert(0, sys_p)
from train_ab_lora import SpanDataset, collate

VAL = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/corpus/ab_dataset/val.jsonl")
for name, path in [
    ("v1现役(机械标签927条)", os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/hermes-lingua-bge")),
    ("bge-ab(教师版1585条)", os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/hermes-lingua-bge-ab")),
]:
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForTokenClassification.from_pretrained(path, num_labels=2).eval()
    ds = SpanDataset(VAL, tok)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=lambda b: collate(b, tok.pad_token_id))
    tp = fp = fn = pred_keep = gold_keep = tot = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            p = out.logits.argmax(-1)
            l = batch["labels"]
            m = l != -100
            p, l = p[m], l[m]
            pred_keep += int((p == 1).sum())
            gold_keep += int((l == 1).sum())
            tot += int(m.sum())
            tp += int(((p == 1) & (l == 1)).sum())
            fp += int(((p == 1) & (l == 0)).sum())
            fn += int(((p == 0) & (l == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"{name}: 预测保留率={pred_keep/tot:.2%} 真值={gold_keep/tot:.2%} F1={f1:.3f}")
    del model
