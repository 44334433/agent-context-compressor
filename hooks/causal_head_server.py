#!/usr/bin/env python3
"""因果判别头 HTTP 服务 (拍板: 接线提炼摘要支路) — :8767

架构对齐embedding service embed daemon(:8766) 模式: 独立常驻进程, 网关零内存占用, 崩了不连坐。
POST /v1/pair_score  {"pairs":[{"text_a":..,"text_b":..}]}
  → mmBERT-base+LoRA (causal-head-mmbert) 批推理 → {"scores":[p(配对),...]}
模型: ~/.hermes/state/model-foundation/models/causal-head-mmbert
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = os.path.join(os.environ.get("MODEL_DIR", os.path.expanduser("~/models")), "mmBERT-base")
HEAD = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state/model-foundation/models/causal-head-mmbert")
PORT = 8767
DEV = "cpu"  # 常驻低频调用: 强制 CPU (GPU 留给 NInfer 27B; CPU 单对 ~30ms 足够)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.unk_token
base_model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
model = PeftModel.from_pretrained(base_model, HEAD).to(DEV).eval()
print(f"causal-head loaded on {DEV}", flush=True)


def score_pairs(pairs):
    enc = tok(
        [p["text_a"][:400] for p in pairs],
        [p["text_b"][:400] for p in pairs],
        truncation=True, max_length=256, padding=True, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(DEV), attention_mask=enc["attention_mask"].to(DEV)).logits
    probs = torch.softmax(logits, -1)[:, 1].tolist()
    return [round(p, 4) for p in probs]


class H(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            pairs = body.get("pairs") or []
            if not pairs:
                self._r({"error": "no pairs"}); return
            self._r({"scores": score_pairs(pairs[:32])})
        except Exception as e:
            self._r({"error": str(e)[:200]})

    def do_GET(self):
        self._r({"status": "ok", "model": "causal-head-mmbert"})

    def _r(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
