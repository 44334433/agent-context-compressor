#!/usr/bin/env python3
"""audit_correction · 用户纠错自动扫描插件 (2026-10-15 拍「现在做」)

目标: 把「用户提问=唯一 error handler」变成机械账本——每轮 LLM 调用前扫最新
user 消息, 命中纠错信号词 → audit_ledger.log(user_correction)。
纯机械词表匹配, 零 LLM; turn 级去重防重复记账; 任何异常静默失败(绝不阻断主链)。
"""
import hashlib
import json
import os
import re

_LEDGER = os.path.expanduser("~/.agent_ctx/scripts/audit_ledger.py")
_STATE = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state/audit_correction_seen.json")

# 纠错信号词表 (保守起点, 误报宁可漏不滥——漏了人工补, 滥了账本失真)
_SIGNALS = re.compile(
    r"(不是这个意思|你忘了|你又忘|又没记录|没记录|没登记|没落盘|没写进去|"
    r"打回|不对吧|搞错了|弄错了|重做|重新做|我说过|之前说过|刚才说过|"
    r"根本没|查无|哪儿去了|去哪了|怎么没|为什么没|为什么又|闭环没有|闭环了吗|"
    r"拍板过了|说过要|怎么又)"
)
# 系统注入/工具回显类消息豁免
_NOISE_RE = re.compile(r"\[(OUT-OF-BAND|IMPORTANT|CONTEXT|ARC-|LESSON|ANCHORS?)", re.I)

_seen_cache = None


def _load_seen():
    global _seen_cache
    if _seen_cache is None:
        try:
            with open(_STATE, encoding="utf-8") as f:
                _seen_cache = set(json.load(f))
        except Exception:
            _seen_cache = set()
    return _seen_cache


def _save_seen(seen):
    try:
        if len(seen) > 5000:
            seen = set(list(seen)[-3000:])
        with open(_STATE, "w", encoding="utf-8") as f:
            json.dump(sorted(seen)[-5000:], f)
    except Exception:
        pass


def _log_correction(text: str, turn_id: str, session_id: str):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("audit_ledger", _LEDGER)
        if not (spec and spec.loader):
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        hit = _SIGNALS.search(text)
        mod.log("user_correction", "user", f"纠错信号「{hit.group(0) if hit else '纠错'}」: {text[:120]}", f"turn={turn_id} sid={session_id}")
    except Exception:
        pass


def on_pre_llm_call(*, turn_id: str = "", session_id: str = "", messages: object = None, **_) -> None:
    try:
        if not isinstance(messages, list) or not messages:
            return
        last_user = None
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = m
                break
        if not last_user:
            return
        text = str(last_user.get("content") or "")
        if not text or _NOISE_RE.search(text[:60]):
            return
        if not _SIGNALS.search(text):
            return
        fp = hashlib.md5(f"{turn_id}|{text[:200]}".encode()).hexdigest()
        seen = _load_seen()
        if fp in seen:
            return
        seen.add(fp)
        _save_seen(seen)
        _log_correction(text[:300], turn_id, session_id)
    except Exception:
        pass  # 审计绝不阻断主链


HOOKS = (("pre_llm_call", on_pre_llm_call),)
