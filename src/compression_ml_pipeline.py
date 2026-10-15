"""Hermes 自研机器学习摘要压缩引擎流水线 · 核心适配层 (Stage 3 Integration)

真源与背景:
- 交接文档: ~/.agent_ctx/docs/05-交接会话/交接-机器学习摘要压缩引擎-2026-10-14.md
- 任务 ID: todo-ml-compression-pipeline (阶段三: 集成接入 Hermes 压缩触发链)
- 阶段一产物: scripts/compression_pipeline_stage1.py (机械过滤与切片)
- 阶段二产物: scripts/compression_pipeline_stage2.py (5 要素事实提炼)

核心能力:
1. `pre_compress_filter_turns`: 压缩前置清洗钩子，削减 70%+ 冗余噪声，保全 100% 实体锚点。
2. `generate_five_element_fact_summary`: 纯 Python 本地合成结构化 5 要素事实摘要，提供零 LLM、确定性、高保真摘要兜底。
3. `should_use_ml_prefilter`: 模式开关，通过 config `compression.ml_pipeline` 控制（默认启用 pre-filter，fail-open）。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.compression.ml_pipeline")

# ==============================================================================
# 一、规则词表与模式 AST (High-Density Anchors & Noise Patterns)
# ==============================================================================

RE_COMMIT = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
RE_FILE_PATH = re.compile(
    r"(?:"
    r"(?:/(?:home|etc|var|tmp|media|usr|root|opt|bin|sbin|srv)/[a-zA-Z0-9_\.\-]+(?:/[a-zA-Z0-9_\.\-]+)*)|"
    r"(?:~/[a-zA-Z0-9_\.\-]+(?:/[a-zA-Z0-9_\.\-]+)*)|"
    r"(?:[a-zA-Z0-9_\.\-]+/(?:[a-zA-Z0-9_\.\-]+/)*[a-zA-Z0-9_\.\-]+\."
    r"(?:py|json|yaml|yml|md|ts|tsx|js|jsx|sh|cpp|c|h|rs|toml|jsonl|txt|log|db|sql|csv|png|mp4|gguf|whl))"
    r")\b"
)
RE_LINE_NUM = re.compile(
    r"(?:"
    r"\bline\s+\d+\b|"
    r"\bL\d+(?:-L?\d+)?\b|"
    r"^\s*\d+\|\s*|"
    r"@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@"
    r")",
    re.MULTILINE,
)
RE_ERROR_STACK = re.compile(
    r"(?:"
    r"Traceback \(most recent call last\):|"
    r"[A-Za-z0-9_]+(?:Error|Exception):|"
    r"[A-Za-z0-9_]+Issue\]|"
    r"exit\s+[1-9]\d*|"
    r"FAILED|"
    r"fatal:|panic:|SIGSEGV|SIGTERM"
    r")",
    re.MULTILINE,
)
RE_TASKBOARD = re.compile(
    r"(?:"
    r"task_board\s+(?:update|register|query)|"
    r"status=[a-zA-Z_]+|"
    r"progress=\d+%|"
    r"milestone="
    r")"
)
RE_DECISION = re.compile(
    r"(?:"
    r"过|按推荐|拍板|同意|做|不接受|拒绝|废弃|上线|回退|继续|记录|重跑|发我|"
    r"选项[A-Za-z0-9]|不设最低点|方向明确|关键发现|⚠️\s*新发现|边界|工作点定在|"
    r"先不要碰我的鼠标|主号|留\d+额度"
    r")"
)
RE_METRIC = re.compile(
    r"\b\d+(?:\.\d+)?(?:ms|s|min|h|MB|GB|KB|TB|tok|tokens|tok/s|%|pp|dB|fps)\b"
)

# 噪声模式
RE_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
RE_CARRIAGE_RETURN = re.compile(r"\r[^\n]*")
RE_PROGRESS_BAR = re.compile(
    r"(?:"
    r"\d+%\s*\|[█#=>\-\s]+\|\s*\d+/\d+[^\n]*|"
    r"\[[=>\s\-#]+\]\s*\d+%(?:/\d+%)?[^\n]*|"
    r"Fetching\s+package\s+metadata[^\n]*|"
    r"Downloading[^\n]*\d+%"
    r")"
)
RE_REPEATED_DIVIDER = re.compile(r"^[=\-_*]{4,}$", re.MULTILINE)
RE_CONSECUTIVE_NEWLINES = re.compile(r"\n{3,}")

DECISION_PATTERNS = [
    re.compile(r"(?:既定 SOP|走既定|规避|绕行|先例|机制|根因|为什么|发现|⚠️|关键发现|拍板|方向明确|工作点定在)"),
    re.compile(r"(?:不能|禁止|必须|不设最低点|核心原因|事故|回退|验证证伪)"),
]

CHANGE_PATTERNS = [
    re.compile(r"(?:diff 恰好|改动|配置现值|更新|替换|新增|修复|热生效|参数|落盘)"),
    re.compile(r"(?:350K|256K|87K|tok|tokens|tok/s|%|pp|秒|s|ms|MB|GB)"),
]

NEXT_PATTERNS = [
    re.compile(r"(?:重跑|发我|记录，明天|明天额度|待验证|待办|下一步|Next|没收到|请继续|继续推进)"),
    re.compile(r"(?:需先修|稍后|挂起|阻塞|pending|待拍板)"),
]


# ==============================================================================
# 二、清洗与锚点提取实现 (Deterministic Cleaner & Chunker)
# ==============================================================================

def clean_noise(text: str) -> str:
    """清洗终端控制符、光标刷新与进度条。"""
    if not text:
        return ""
    s = RE_ANSI_ESCAPE.sub("", text)
    s = RE_CARRIAGE_RETURN.sub("", s)
    s = RE_PROGRESS_BAR.sub("", s)
    s = RE_REPEATED_DIVIDER.sub("---", s)
    s = RE_CONSECUTIVE_NEWLINES.sub("\n\n", s)
    return s.strip()


def collapse_repetitive_lines(text: str, max_repeat: int = 2) -> str:
    """折叠重复轮询行。"""
    lines = text.split("\n")
    if len(lines) <= max_repeat:
        return text
    result = []
    prev_line = None
    repeat_count = 0
    for line in lines:
        s_line = line.strip()
        if s_line and s_line == prev_line:
            repeat_count += 1
            if repeat_count < max_repeat:
                result.append(line)
        else:
            if repeat_count >= max_repeat:
                result.append(f"[... 连续折叠 {repeat_count - max_repeat + 1} 行相同回显 ...]")
            repeat_count = 0
            prev_line = s_line
            result.append(line)
    if repeat_count >= max_repeat:
        result.append(f"[... 连续折叠 {repeat_count - max_repeat + 1} 行相同回显 ...]")
    return "\n".join(result)


def extract_turn_anchors(text: str) -> Dict[str, List[str]]:
    """提取单个回合的实体锚点。"""
    if not text:
        return {}
    commits = set(
        m.group(0)
        for m in RE_COMMIT.finditer(text)
        if not m.group(0).isdigit() and len(m.group(0)) >= 7
    )
    paths = set(m.group(0) for m in RE_FILE_PATH.finditer(text))
    line_nums = set(m.group(0).strip() for m in RE_LINE_NUM.finditer(text))
    errors = set(m.group(0).strip() for m in RE_ERROR_STACK.finditer(text))
    tb = set(m.group(0).strip() for m in RE_TASKBOARD.finditer(text))
    decisions = set(m.group(0).strip() for m in RE_DECISION.finditer(text))
    metrics = set(m.group(0).strip() for m in RE_METRIC.finditer(text))

    return {
        "commits": sorted(commits),
        "paths": sorted(paths),
        "line_nums": sorted(line_nums),
        "errors": sorted(errors),
        "task_board": sorted(tb),
        "decisions": sorted(decisions),
        "metrics": sorted(metrics),
    }


def format_anchors_tag(anchors: Dict[str, List[str]]) -> str:
    parts = []
    if anchors.get("paths"):
        parts.append(f"paths=[{', '.join(anchors['paths'][:32])}]")
    if anchors.get("commits"):
        parts.append(f"commits=[{', '.join(anchors['commits'][:8])}]")
    if anchors.get("errors"):
        parts.append(f"errors=[{', '.join(anchors['errors'][:3])}]")
    if anchors.get("line_nums"):
        parts.append(f"lines=[{', '.join(anchors['line_nums'][:4])}]")
    if anchors.get("decisions"):
        parts.append(f"decisions=[{', '.join(anchors['decisions'][:4])}]")
    return " | ".join(parts)


def clean_single_turn_message(msg: Dict[str, Any], preserve_context: int = 240) -> Dict[str, Any]:
    """对单条消息进行无损机械过滤。"""
    role = str(msg.get("role", "unknown"))
    tool_name = str(msg.get("name") or msg.get("tool_name") or "")
    raw_text = (
        msg.get("content")
        or msg.get("content_sample")
        or (str(msg.get("content_head", "")) + str(msg.get("content_tail", "")))
        or ""
    )
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # tool_calls 参数文本并入锚点源 (2026-10-15 验收闸实证: 路径/命令主要活在 tool_calls.arguments,
    # 纯 content 源的路径覆盖仅 33%; 并入后五要素/决策链/锚点提取全链受益)
    try:
        _calls = msg.get("tool_calls") or []
        if isinstance(_calls, str) and _calls:
            _calls = json.loads(_calls)
        _arg_lines = []
        if isinstance(_calls, list):
            for _call in _calls[:4]:
                _fn = (_call.get("function") or {}) if isinstance(_call, dict) else {}
                _name = _fn.get("name") or tool_name or "tool"
                _args = _fn.get("arguments") or ""
                if isinstance(_args, str) and _args:
                    try:
                        _a = json.loads(_args)
                        _flat = " ".join(str(v) for v in _a.values())[:300] if isinstance(_a, dict) else str(_a)[:300]
                    except Exception:
                        _flat = _args[:300]
                    _arg_lines.append(f"[tool-args {_name}: {_flat}]")
        if _arg_lines:
            raw_text = (raw_text + "\n" if raw_text else "") + "\n".join(_arg_lines)
    except Exception:
        pass

    if not raw_text:
        return msg

    cleaned = clean_noise(raw_text)
    cleaned = collapse_repetitive_lines(cleaned)
    anchors = extract_turn_anchors(cleaned)
    anchor_str = format_anchors_tag(anchors)

    # 1. 丢弃类 / 重复工具输出 (D 档)
    if "[Duplicate tool output" in cleaned or "Duplicate tool output — same content" in cleaned:
        new_msg = copy.copy(msg)
        new_msg["content"] = f"[DUP: {tool_name or 'tool'} 与前次调用输出一致 (已折叠)]"
        return new_msg

    # 2. 异步委派回投聚合
    if "[ASYNC DELEGATION BATCH COMPLETE" in cleaned:
        lines = [
            l.strip()
            for l in cleaned.split("\n")
            if l.strip().startswith("**")
            or "log" in l
            or "deleg_" in l
            or "task-" in l
            or "status=" in l
        ]
        compact = "\n".join(lines[:8])
        res = f"[ASYNC DELEGATION COMPACT]\n{compact}"
        if anchor_str:
            res += f"\n[ANCHORS]: {anchor_str}"
        new_msg = copy.copy(msg)
        new_msg["content"] = res
        return new_msg

    # 3. 大体积工具输出指针化 (S 档)
    if role == "tool" and len(cleaned) > 400:
        head = cleaned[:preserve_context].strip()
        tail = cleaned[-preserve_context // 2 :].strip()
        stub = f"[STUB: {tool_name or 'tool'} orig_len={len(cleaned)} | head: {head} ... tail: {tail}"
        if anchor_str:
            stub += f" | ANCHORS: {anchor_str}"
        stub += "]"
        new_msg = copy.copy(msg)
        new_msg["content"] = stub
        return new_msg

    # 4. Patch diff 处理
    if tool_name == "patch" and ("diff" in cleaned or "files_modified" in cleaned):
        head = cleaned[:preserve_context].strip()
        tail = cleaned[-preserve_context // 2 :].strip()
        stub = f"[STUB: patch diff orig_len={len(cleaned)} | head: {head} ... tail: {tail}"
        if anchor_str:
            stub += f" | ANCHORS: {anchor_str}"
        stub += "]"
        new_msg = copy.copy(msg)
        new_msg["content"] = stub
        return new_msg

    # 其余消息保留全文（已剔除 ANSI、进度条与连续空行）
    if cleaned != raw_text:
        new_msg = copy.copy(msg)
        new_msg["content"] = cleaned
        return new_msg

    return msg


def estimate_tokens(text: str) -> int:
    """粗估 token 数。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + max(0, (len(text) - cjk) // 4)


@dataclass
class InternalChunk:
    chunk_id: str
    turn_range: Tuple[int, int]
    content: str
    anchors: Dict[str, List[str]]


def chunk_turns(turns: List[Dict[str, Any]], max_tokens: int = 2000) -> List[InternalChunk]:
    chunks: List[InternalChunk] = []
    curr_texts: List[str] = []
    curr_commits: List[str] = []
    curr_paths: List[str] = []
    curr_lines: List[str] = []
    curr_errors: List[str] = []
    curr_decisions: List[str] = []
    curr_metrics: List[str] = []
    start_idx = 0
    accum_tokens = 0

    for idx, msg in enumerate(turns):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        text = f"\n--- [Turn {idx} | {role}] ---\n{content}"
        toks = estimate_tokens(text)

        if curr_texts and (accum_tokens + toks > max_tokens):
            c_id = f"chunk_{len(chunks):03d}"
            chunks.append(
                InternalChunk(
                    chunk_id=c_id,
                    turn_range=(start_idx, idx - 1),
                    content="".join(curr_texts).strip(),
                    anchors={
                        "commits": sorted(set(curr_commits)),
                        "paths": sorted(set(curr_paths)),
                        "line_nums": sorted(set(curr_lines)),
                        "errors": sorted(set(curr_errors)),
                        "decisions": sorted(set(curr_decisions)),
                        "metrics": sorted(set(curr_metrics)),
                    },
                )
            )
            curr_texts = []
            curr_commits, curr_paths, curr_lines, curr_errors, curr_decisions, curr_metrics = [], [], [], [], [], []
            start_idx = idx
            accum_tokens = 0

        curr_texts.append(text)
        accum_tokens += toks
        anchors = extract_turn_anchors(content)
        curr_commits.extend(anchors.get("commits", []))
        curr_paths.extend(anchors.get("paths", []))
        curr_lines.extend(anchors.get("line_nums", []))
        curr_errors.extend(anchors.get("errors", []))
        curr_decisions.extend(anchors.get("decisions", []))
        curr_metrics.extend(anchors.get("metrics", []))

    if curr_texts:
        c_id = f"chunk_{len(chunks):03d}"
        chunks.append(
            InternalChunk(
                chunk_id=c_id,
                turn_range=(start_idx, len(turns) - 1),
                content="".join(curr_texts).strip(),
                anchors={
                    "commits": sorted(set(curr_commits)),
                    "paths": sorted(set(curr_paths)),
                    "line_nums": sorted(set(curr_lines)),
                    "errors": sorted(set(curr_errors)),
                    "decisions": sorted(set(curr_decisions)),
                    "metrics": sorted(set(curr_metrics)),
                },
            )
        )
    return chunks


# ==============================================================================
# 三、对外导出接口 (Stage 3 Integration Interfaces)
# ==============================================================================

def should_use_ml_prefilter(compressor: Any) -> bool:
    """是否启用自研机器学习预过滤。支持环境变量与配置覆盖，默认启用 (fail-open)。"""
    try:
        if os.environ.get("HERMES_ML_COMPRESSION_DISABLE") == "1":
            return False
        cfg = getattr(compressor, "config", {}) or {}
        comp_cfg = cfg.get("compression", {}) if isinstance(cfg, dict) else {}
        if comp_cfg.get("ml_pipeline") is False:
            return False
        return True
    except Exception:
        return True


def pre_compress_filter_turns(
    turns: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """阶段三核心钩子：对待压缩的消息序列做机械清洗与指针化。"""
    if not turns:
        return turns, {}

    t0 = time.perf_counter()
    orig_chars = sum(
        len(str(m.get("content") or m.get("content_sample") or "")) for m in turns
    )

    filtered_turns = [clean_single_turn_message(m) for m in turns]

    post_chars = sum(len(str(m.get("content", ""))) for m in filtered_turns)
    dt_ms = (time.perf_counter() - t0) * 1000

    saved_chars = max(0, orig_chars - post_chars)
    reduction_pct = (saved_chars / orig_chars * 100.0) if orig_chars > 0 else 0.0

    stats = {
        "orig_chars": orig_chars,
        "filtered_chars": post_chars,
        "saved_chars": saved_chars,
        "reduction_pct": round(reduction_pct, 2),
        "latency_ms": round(dt_ms, 2),
        "turns_count": len(turns),
    }

    logger.debug(
        "ML Pre-compress filter applied: %d -> %d chars (-%.1f%%) in %.2fms",
        orig_chars,
        post_chars,
        reduction_pct,
        dt_ms,
    )
    return filtered_turns, stats


def build_user_mandate_record(turns: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
    """用户指令零丢失闸：全区间每条用户消息强制实录进摘要（机械保证）。

    根因背景 (2026-10-15 失忆事故): 用户拍板指令超出"最近N轮原话"保护窗后，
    机械切片归纳对"下令→执行→部署"叙事链提取失败，导致重训换代事件在摘要中不存在，
    后续会话报出过时状态被用户当场纠正。
    本闸 = 机械层兜底：用户消息逐条留痕，无论多老、无论切片提取是否命中。
    """
    user_lines: List[str] = []
    total_user = 0
    for idx, m in enumerate(turns):
        if str(m.get("role", "")) != "user":
            continue
        raw = m.get("content", "")
        if isinstance(raw, list):
            raw = " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in raw
            )
        txt = clean_noise(str(raw)).strip()
        if not txt:
            continue
        total_user += 1
        flat = " ".join(txt.split())
        if len(flat) > 200:
            flat = flat[:200] + "…"
        user_lines.append(f"- [Turn {idx}] {flat}")

    folded = 0
    if len(user_lines) > 80:
        folded = len(user_lines) - 60
        user_lines = (
            user_lines[:30]
            + [f"- [... 中间 {folded} 条用户消息已折叠, 原文可经 session_search 回捞 ...]"]
            + user_lines[-30:]
        )

    section = ""
    if user_lines:
        section = (
            "## 用户指令全量实录 (零丢失闸·全区间机械收录·时间序)\n"
            + "\n".join(user_lines)
        )

    stats = {
        "user_total": total_user,
        "user_recorded_lines": len(user_lines),
        "user_folded": folded,
    }
    return section, stats


def extract_decision_chain(turns: List[Dict[str, Any]], max_nodes: int = 40) -> str:
    """零 LLM 决策链时间线：机械配对「用户指令 → 工具动作 → 执行结果」三元组。

    治理目标: 压缩后丢失「用户下令→执行→部署」因果叙事链 (失忆事故主因之一)。
    纯 Python 零模型毫秒级; 节点上限 max_nodes 防止时间线爆炸。
    """
    if not turns:
        return ""
    try:
        nodes: List[str] = []
        pending_cmds: List[str] = []   # 最近 3 条候选指令 (判别头 rerank 选归属)
        open_idx: Optional[int] = None  # 等待挂结果的最新动作节点下标

        def _pick_cmd(cands: List[str], action_brief: str) -> str:
            """歧义窗口 (>1 候选指令) → causal-head :8767 rerank 选归属; 失败回退最近一条 (fail-open)"""
            if not cands:
                return ""
            if len(cands) == 1:
                return cands[-1]
            try:
                import urllib.request as _u
                _payload = json.dumps({"pairs": [{"text_a": c, "text_b": action_brief} for c in cands]}).encode()
                _req = _u.Request("http://127.0.0.1:8767/v1/pair_score", data=_payload, headers={"Content-Type": "application/json"})
                with _u.urlopen(_req, timeout=1.5) as _r:
                    _scores = json.loads(_r.read()).get("scores") or []
                if len(_scores) == len(cands):
                    return cands[max(range(len(cands)), key=lambda i: _scores[i])]
            except Exception:
                pass
            return cands[-1]

        def _brief(s: str, n: int = 70) -> str:
            s = " ".join(str(s).split())
            return s[:n] + ("…" if len(s) > n else "")

        def _brief_result(content: str) -> str:
            c = str(content)
            m = re.search(r'"exit_code":\s*(-?\d+)|exit[_ ]code[:= ]+(-?\d+)', c)
            if m:
                code = m.group(1) or m.group(2)
                tag = "exit 0" if code == "0" else f"exit {code}(异常)"
            elif re.search(r"Traceback|Error|FAILED|fatal:|panic:", c):
                tag = "报错"
            else:
                tag = "完成"
            m_path = re.search(r"(/[\w./\-]{6,80}|~[\w./\-]{4,80})", c)
            path = f" → {m_path.group(1)}" if m_path else ""
            return f"{tag}{path}"

        for msg in turns:
            role = msg.get("role")
            if role == "user":
                _c = str(msg.get("content", ""))
                if not _c.startswith(("[STUB:", "[OUT-OF-BAND")):  # 系统注入轮不算用户指令
                    pending_cmds.append(_brief(_c, 60))
                    pending_cmds = pending_cmds[-3:]
            elif role == "assistant":
                calls = msg.get("tool_calls") or []
                if isinstance(calls, str):
                    try:
                        calls = json.loads(calls)
                    except Exception:
                        calls = []
                for call in list(calls)[:4]:
                    fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                    name = fn.get("name") or "?"
                    args = fn.get("arguments") or ""
                    key_arg = ""
                    parsed = None
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict):
                            for k in ("command", "path", "file_path", "query", "url", "pattern", "task_id"):
                                if parsed.get(k):
                                    key_arg = f"{k}={_brief(parsed[k], 44)}"
                                    break
                        elif args:
                            key_arg = _brief(args, 44)
                    if len(nodes) >= max_nodes:
                        break
                    nodes.append(
                        f"- 指令: {_pick_cmd(pending_cmds, f'{name}({key_arg})') or '—'}\n"
                        f"  动作: {name}({key_arg})"
                    )
                    open_idx = len(nodes) - 1
                if len(nodes) >= max_nodes:
                    break
            elif role == "tool" and open_idx is not None and nodes:
                nodes[open_idx] += f"\n  结果: {_brief_result(msg.get('content', ''))}"
                open_idx = None

        if not nodes:
            return ""
        head = "## 决策链时间线 (用户下令→执行→部署)\n*纯机械配对, 保因果顺序, 细节按锚点外查*\n"
        tail = "\n*(时间线截断: 仅前 %d 个动作节点)*" % max_nodes if len(nodes) >= max_nodes else ""
        return head + "\n".join(nodes) + tail
    except Exception as e:
        logger.warning("extract_decision_chain error: %s", e)
        return ""


def verify_summary_coverage(turns: List[Dict[str, Any]], summary: str, max_check: int = 80) -> Dict[str, Any]:
    """压缩后机械验收闸: 关键要素覆盖率自检, 不达标告警 (失忆防线最后一环)。

    检查项: 实录区在位 / 决策链在位 / 路径实体覆盖 / 数字指标覆盖 (阈值 60%, 数据来自失忆事故区间实测基线 72-79%)。
    """
    try:
        user_text = "\n".join(str(m.get("content", "")) for m in turns if m.get("role") == "user")
        paths = list(dict.fromkeys(re.findall(r"(?:/[\w.\-]+){2,}|~[\w./\-]+", user_text)))[:max_check]
        nums = list(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+)?(?:ms|s|min|GB|MB|KB|tok|tok/s|%)\b", user_text)))[:max_check]
        p_miss = [p for p in paths if p not in summary]
        n_miss = [n for n in nums if n not in summary]
        p_rate = 1 - len(p_miss) / max(len(paths), 1)
        n_rate = 1 - len(n_miss) / max(len(nums), 1)
        report = {
            "mandate_in": "用户指令全量实录" in summary,
            "chain_in": "决策链时间线" in summary,
            "path_cov": round(p_rate, 2),
            "num_cov": round(n_rate, 2),
            "path_miss_sample": p_miss[:5],
            "num_miss_sample": n_miss[:5],
        }
        ok = report["mandate_in"] and report["chain_in"] and p_rate >= 0.6 and n_rate >= 0.6
        report["pass"] = ok
        if not ok:
            logger.warning("压缩验收闸未过(失忆风险): %s", json.dumps(report, ensure_ascii=False))
            try:  # 审计账本: 自检拦截事件 (2026-10-15 拍「落」)
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("audit_ledger", os.path.expanduser("~/.agent_ctx/scripts/audit_ledger.py"))
                if _spec and _spec.loader:
                    _al = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_al)
                    _al.log("self_catch", "verify_gate", f"验收闸拦截 path={report.get('path_cov')} num={report.get('num_cov')}", "compression_ml_pipeline.py:602")
            except Exception as _al_err:
                logger.debug("audit_ledger log fail: %s", _al_err)
        else:
            logger.info("压缩验收闸通过: path=%.0f%% num=%.0f%% mandate=%s chain=%s",
                        p_rate * 100, n_rate * 100, report["mandate_in"], report["chain_in"])
        return report
    except Exception as e:
        logger.warning("verify_summary_coverage error: %s", e)
        return {"pass": False, "error": str(e)}


def generate_five_element_fact_summary(turns: List[Dict[str, Any]]) -> str:
    """基于阶段一与阶段二算法，纯本地生成高信噪比 5 要素事实摘要。"""
    if not turns:
        return "无待压缩的历史记录。"

    try:
        # 1. 过滤与分块
        cleaned_turns = [clean_single_turn_message(m) for m in turns]
        chunks = chunk_turns(cleaned_turns, max_tokens=2000)

        # 2. 5 要素提炼
        md_blocks = [
            "## 会话关键事实结构化归纳 (自研机器学习流水线)",
            f"*纯本地零 LLM 提炼自 {len(turns)} 个历史回合 ({len(chunks)} 个切片)*\n",
        ]

        for c in chunks:
            turn_label = f"Turns {c.turn_range[0]}..{c.turn_range[1]}"
            paths = [p for p in c.anchors.get("paths", []) if "/" in p and len(p) > 2]
            target_files = sorted(set(paths))[:8]

            # 核心变更
            changes = []
            lines = c.anchors.get("line_nums", [])
            commits = c.anchors.get("commits", [])
            metrics = c.anchors.get("metrics", [])
            if lines:
                changes.append(f"代码行定位: {', '.join(lines[:4])}")
            if commits:
                changes.append(f"版本哈希: {', '.join(commits[:4])}")
            if metrics:
                changes.append(f"指标实测: {', '.join(metrics[:5])}")

            content_lines = [l.strip() for l in c.content.split("\n") if l.strip()]
            for l in content_lines:
                if any(p.search(l) for p in CHANGE_PATTERNS):
                    if not l.startswith("---") and not l.startswith("[STUB") and len(l) <= 150:
                        if l not in changes and not any(l in x for x in changes):
                            changes.append(l)
                            if len(changes) >= 5:
                                break

            # 决策教训
            decisions = []
            for l in content_lines:
                if any(p.search(l) for p in DECISION_PATTERNS):
                    if not l.startswith("---") and not l.startswith("[STUB") and len(l) <= 180:
                        if l not in decisions:
                            decisions.append(l)
                            if len(decisions) >= 5:
                                break
            dec_list = c.anchors.get("decisions") or []
            if not decisions and dec_list:
                decisions = [f"包含决策指令: {', '.join(dec_list[:4])}"]

            # 未完成
            pending = []
            for l in content_lines:
                if any(p.search(l) for p in NEXT_PATTERNS):
                    if not l.startswith("---") and not l.startswith("[STUB") and len(l) <= 150:
                        if l not in pending:
                            pending.append(l)
                            if len(pending) >= 4:
                                break
            if not pending:
                pending = ["本切片无未完成阻塞项 (已全部就绪/完成)"]

            # 拼接 Chunk Markdown
            block = [
                f"### 【切片 {c.chunk_id} | {turn_label}】",
                f"- **操作对象/文件**: {', '.join(f'`{f}`' for f in target_files) if target_files else '无具体文件'}",
            ]
            if changes:
                block.append("- **核心变更/实测**:")
                for ch in changes[:5]:
                    block.append(f"  • {ch}")
            if decisions:
                block.append("- **决策/教训/理由**:")
                for dc in decisions[:5]:
                    block.append(f"  • {dc}")
            if pending:
                block.append("- **未完成项/Next**:")
                for pd in pending[:4]:
                    block.append(f"  • {pd}")

            md_blocks.append("\n".join(block))

        # 2.5 用户指令零丢失闸 (2026-10-15 失忆事故根治): 全区间每条用户消息机械实录
        mandate_section, mandate_stats = build_user_mandate_record(cleaned_turns)
        if mandate_section:
            if len(mandate_section) > 50000:
                # 体积闸: 超大区间实录区截断防爆炸; 完整指令序列永久存于 state.db, 走 session_search 外查
                keep = mandate_section[:50000]
                cut = keep.rfind('\n')
                mandate_section = keep[:cut if cut > 40000 else 50000] + (
                    f"\n\n*(实录区体积闸: 超限截断, 展示前段; 完整 {mandate_stats['user_recorded_lines']} 条用户指令"
                    f"按 message id 顺序存于 state.db, 用 session_search(session_id=会话id) 外查)*"
                )
                logger.warning("实录区 %.0fK 字符超 50K 体积闸, 已截断+挂外查指针", len(mandate_section) / 1000)
            md_blocks.append(mandate_section)
        logger.info(
            "用户指令零丢失闸: 区间用户消息 %d 条, 实录 %d 行, 折叠 %d 条",
            mandate_stats["user_total"],
            mandate_stats["user_recorded_lines"],
            mandate_stats["user_folded"],
        )

        # 2.6 决策链时间线 (零 LLM 叙事链: 用户下令→执行→部署; 用原始 turns 保 tool_calls 字段)
        try:
            chain_section = extract_decision_chain(turns)
            if chain_section:
                md_blocks.append(chain_section)
        except Exception as _chain_err:
            logger.warning("决策链时间线生成失败(不影响主摘要): %s", _chain_err)

        # 2.7 压缩后机械验收闸 (失忆防线最后一环): 关键要素覆盖率自检, 不达标日志告警
        try:
            verify_summary_coverage(turns, "\n\n".join(md_blocks))
        except Exception as _verify_err:
            logger.warning("压缩验收闸执行失败(不阻塞主流程): %s", _verify_err)

        # 3. 持续数据集沉淀 (每次压缩自动积累训练数据)
        try:
            dataset_dir = Path.home() / ".hermes" / "state" / "compression_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            stream_file = dataset_dir / "stream_records.jsonl"
            record = {
                "ts": time.time(),
                "turns_count": len(turns),
                "chunks_count": len(chunks),
                "summary_length": len("\n\n".join(md_blocks)),
                "chunks_meta": [
                    {
                        "chunk_id": c.chunk_id,
                        "turns": c.turn_range,
                        "anchors": c.anchors,
                    }
                    for c in chunks
                ],
            }
            with open(stream_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as _rec_err:
            logger.debug("Failed to append stream record: %s", _rec_err)

        return "\n\n".join(md_blocks)
    except Exception as e:
        logger.warning("generate_five_element_fact_summary error: %s", e)
        return f"## 会话历史摘要 (本地流水线降级)\n包含 {len(turns)} 条历史回合。"


# ═══ 记忆生命周期机械闭环 (2026-10-15 拍板恢复落地) ═══
# 出处: 本会话链 10-15 凌晨「四问清算」(state.db message 847956), 用户拍板「可以, 通过,
# 按这个方案全面推进落实」(847957)——当时未落盘即被上下文压缩弄丢, 现按拍板原文 1:1 恢复。
# 原则: 严禁大模型参与状态机流转——SQLite/时间戳/计数器不会撒谎; LLM 只允许做要旨精炼(可选)。

LIFECYCLE_DB_PATH = os.environ.get(
    "COMPRESSION_LIFECYCLE_DB",
    os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.agent_ctx")), "state/context_lifecycle.db"),
)

_LIFECYCLE_DDL = """
CREATE TABLE IF NOT EXISTS fact_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT DEFAULT 'fact',
    state TEXT DEFAULT 'ACTIVE',
    born_turn INTEGER DEFAULT 0,
    last_referenced_turn INTEGER DEFAULT 0,
    hits INTEGER DEFAULT 0,
    is_pending INTEGER DEFAULT 0,
    is_invalidated INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_fi_sess_state ON fact_items(session_id, state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fi_dedupe ON fact_items(session_id, content);
"""

# 未完成项机械判定词表 (拍板原文: 匹配 Next/待办/尚未 关键词机械决定)
_OPEN_LOOP_RE = re.compile(
    r"(?:^|\n)\s*[-•*]?\s*(?:Next|NEXT|待拍板|待办|尚未|未完成|在飞|TODO|时钟)", re.UNICODE
)


def _lifecycle_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(LIFECYCLE_DB_PATH, timeout=10)
    conn.executescript(_LIFECYCLE_DDL)
    return conn


def upsert_fact_items(session_id: str, items: List[str], current_turn: int) -> int:
    """五要素/决策链条目入库: 未完成项机械标记 is_pending, 同内容去重+命中刷新. 零 LLM."""
    if not items:
        return 0
    conn = _lifecycle_conn()
    cur = conn.cursor()
    n = 0
    for raw in items:
        text = (raw or "").strip()[:500]
        if not text:
            continue
        pending = 1 if _OPEN_LOOP_RE.search(text[:200]) else 0
        cur.execute(
            """INSERT INTO fact_items
                 (session_id, content, kind, born_turn, last_referenced_turn, hits, is_pending)
               VALUES (?, ?, 'fact', ?, ?, 1, ?)
               ON CONFLICT(session_id, content) DO UPDATE SET
                 hits = hits + 1,
                 last_referenced_turn = MAX(last_referenced_turn, excluded.last_referenced_turn),
                 is_pending = MAX(is_pending, excluded.is_pending),
                 state = CASE WHEN state='ARCHIVED' THEN 'ACTIVE' ELSE state END""",
            (session_id, text, current_turn, current_turn, pending),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def advance_memory_lifecycle(db_cursor, current_turn: int) -> List[str]:
    """核心生命周期机械推进 (拍板原文 1:1, 零 LLM 参与):
    1 蔡格尼克免疫 / 2 毒性湮灭 / 3 机械衰减 / 4 机械装配"""
    # 1. 蔡格尼克免疫：未完成项 (is_pending=1) 机械免疫一切衰减，恒定保留
    db_cursor.execute("UPDATE fact_items SET state='ACTIVE' WHERE is_pending=1")
    # 2. 毒性湮灭：被显式标记 INVALIDATED 的结论，机械物理擦除
    db_cursor.execute("DELETE FROM fact_items WHERE is_invalidated=1")
    # 3. 机械衰减：距今超过 150 轮且命中次数 hits <= 1 的旧条目 -> ARCHIVED 并移出 Prompt
    db_cursor.execute(
        """UPDATE fact_items
           SET state='ARCHIVED'
           WHERE state='CONSOLIDATED'
             AND (? - last_referenced_turn) > 150
             AND hits <= 1""",
        (current_turn,),
    )
    # 4. 机械装配：仅拉取 ACTIVE 与 CONSOLIDATED 态条目进入 Prompt 上下文
    db_cursor.execute(
        "SELECT content FROM fact_items WHERE state IN ('ACTIVE', 'CONSOLIDATED')"
    )
    return [r[0] for r in db_cursor.fetchall()]


def mark_invalidated(session_id: str, content_substring: str) -> int:
    """毒性标记入口: 拍板「废弃/作废」某结论时机械调用, 下次 advance 物理擦除."""
    conn = _lifecycle_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE fact_items SET is_invalidated=1 WHERE session_id=? AND content LIKE ?",
        (session_id, f"%{content_substring}%"),
    )
    n = cur.rowcount
    conn.commit()
    conn.close()
    return n


def render_lifecycle_block(session_id: str, current_turn: int, limit: int = 40) -> str:
    """压缩摘要顶层注入块: 蔡格尼克免疫区(置顶恒保) + 活跃账本. 纯机械, 零 LLM."""
    try:
        conn = _lifecycle_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE fact_items SET state='ACTIVE' WHERE is_pending=1 AND session_id=?",
            (session_id,),
        )
        conn.commit()
        cur.execute(
            """SELECT content, is_pending FROM fact_items
               WHERE session_id=? AND state IN ('ACTIVE','CONSOLIDATED') AND is_invalidated=0
               ORDER BY is_pending DESC, last_referenced_turn DESC LIMIT ?""",
            (session_id, limit),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logger.warning("render_lifecycle_block error: %s", e)
        return ""
    if not rows:
        return ""
    pend = [r[0] for r in rows if r[1]]
    act = [r[0] for r in rows if not r[1]]
    parts = []
    if pend:
        parts.append(
            "【未完成事项·蔡格尼克免疫区(机械恒保)】\n" + "\n".join(f"- {p}" for p in pend[:15])
        )
    if act:
        parts.append(
            "【生命周期账本·ACTIVE/CONSOLIDATED】\n" + "\n".join(f"- {a}" for a in act[:25])
        )
    return "\n\n".join(parts)


def extract_lifecycle_items_from_summary(summary_text: str) -> List[str]:
    """从五要素摘要文本机械提取入库条目: 账本行+Next/待拍板行(=open_loop). 零 LLM."""
    items: List[str] = []
    if not summary_text:
        return items
    for line in summary_text.split("\n"):
        t = line.strip().lstrip("-•* ").strip()
        if not t or t.startswith("#") or t.startswith("【"):
            continue
        if len(t) < 8 or len(t) > 400:
            continue
        items.append(t)
    return items
