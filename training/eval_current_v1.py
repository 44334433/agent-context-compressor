#!/usr/bin/env python3
"""现役 v1 (hermes-lingua-bge) 在 A/B 同款 val 上的 F1 评测 — 三方对比用"""
import json
import os
import sys

sys.path.insert(0, "~/.hermes/scripts")
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from train_ab_lora import SpanDataset, collate

MODEL = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/hermes-lingua-bge")
VAL = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/corpus/ab_dataset/val.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.unk_token
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
ds = SpanDataset(VAL, tok)
loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=lambda b: collate(b, tok.pad_token_id))

tp = fp = fn = 0
with torch.no_grad():
    for batch in loader:
        out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        preds = out.logits.argmax(-1).cpu()
        labels = batch["labels"]
        m = labels != -100
        p, l = preds[m], labels[m]
        tp += int(((p == 1) & (l == 1)).sum())
        fp += int(((p == 1) & (l == 0)).sum())
        fn += int(((p == 0) & (l == 1)).sum())
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(json.dumps({"model": "v1-current", "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}))
