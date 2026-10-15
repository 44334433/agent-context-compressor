#!/usr/bin/env python3
"""A/B 底座训练: 同数据 token 级二分类保/删 — mmBERT-base vs bge-m3 (2026-10-15 拍板)

用法:
  python3 train_ab_lora.py --base-model ${MODEL_DIR:-~/models}/mmBERT-base \
      --output-dir ~/.hermes/state/model-foundation/models/hermes-lingua-mmbert \
      [--report /tmp/ab_mmbert.json]
底座无关: 任何 HF token-classification 兼容 encoder (mmBERT/XLM-R/bge-m3) 皆可。
标签: 现场 tokenize+offset_mapping, token 字符区间落在 kept_spans 内=1(保) 否则 0(删)。
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MAX_LEN = 256


class SpanDataset(Dataset):
    def __init__(self, path: str, tok, label2id=None):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tok = tok
        self.keep_id = 1  # KEEP=1 恒定

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["text"],
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
        )
        spans = r["kept_spans"]
        labels = []
        for (s, e), wid in zip(enc["offset_mapping"], enc["input_ids"]):
            if e <= s:  # special token
                labels.append(-100)
                continue
            keep = any(s >= a and e <= b for a, b in spans)
            labels.append(self.keep_id if keep else 0)
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def collate(batch, pad_id):
    ml = max(len(x["input_ids"]) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        n = ml - len(x["input_ids"])
        out["input_ids"].append(torch.cat([x["input_ids"], torch.full((n,), pad_id)]))
        out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        out["labels"].append(torch.cat([x["labels"], torch.full((n,), -100)]))
    return {k: torch.stack(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/corpus/ab_dataset"))
    ap.add_argument("--report", default="")
    ap.add_argument("--target-modules", required=True, help="逗号分隔, 如 Wqkv,Wo 或 query,key,value")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=2)
    use_cuda = torch.cuda.is_available()
    lora = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        target_modules=[t.strip() for t in args.target_modules.split(",") if t.strip()],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tr_ds = SpanDataset(os.path.join(args.data_dir, "train.jsonl"), tok, {0: 0, 1: 1})
    va_ds = SpanDataset(os.path.join(args.data_dir, "val.jsonl"), tok, {0: 0, 1: 1})

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=25,
        save_steps=1000,
        save_total_limit=1,
        fp16=use_cuda,
        report_to=[],
        remove_unused_columns=False,
    )

    def compute_metrics(p):
        import numpy as np

        preds = p.predictions.argmax(-1).flatten()
        labels = p.label_ids.flatten()
        m = labels != -100
        preds, labels = preds[m], labels[m]
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return {"precision": prec, "recall": rec, "f1": f1}

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("EVAL:", json.dumps(metrics, ensure_ascii=False))
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"base": args.base_model, "eval": metrics}, f, ensure_ascii=False, indent=2)
    print("DONE", args.base_model)


if __name__ == "__main__":
    main()
