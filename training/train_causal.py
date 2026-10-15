#!/usr/bin/env python3
"""因果判别头训练 (拍板②): 指令↔动作配对二分类, bge-m3 与 mmBERT-base 同数据 A/B
输入: corpus/causal_pairs/{train,val}.jsonl (text_a=指令, text_b=动作, label)
架构: encoder pair 编码 (text_a<sep>text_b) → CLS 分类头 + LoRA
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # GPU 窗口模式: 由调用方控制
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class PairDataset(Dataset):
    def __init__(self, path, tok, max_len=256):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text_a"], r["text_b"], truncation=True, max_length=self.max_len, padding="max_length")
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(r["label"]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state/model-foundation/corpus/causal_pairs"))
    ap.add_argument("--report", required=True)
    ap.add_argument("--target-modules", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                      target_modules=[t.strip() for t in args.target_modules.split(",")])
    model = get_peft_model(model, lora)
    tr = PairDataset(os.path.join(args.data_dir, "train.jsonl"), tok)
    va = PairDataset(os.path.join(args.data_dir, "val.jsonl"), tok)

    targs = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=16, per_device_eval_batch_size=16,
        num_train_epochs=2, learning_rate=2e-4, logging_steps=25, save_strategy="no",
        fp16=torch.cuda.is_available(), report_to=[], remove_unused_columns=False,
    )

    def metrics(p):
        import numpy as np
        preds = np.argmax(p.predictions, -1)
        labels = p.label_ids
        tp = int(((preds == 1) & (labels == 1)).sum()); fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9)}

    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va, compute_metrics=metrics)
    trainer.train()
    m = trainer.evaluate()
    print("EVAL:", json.dumps(m))
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"base": args.base_model, "task": "causal_pair_cls", "eval": m}, f, ensure_ascii=False, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
