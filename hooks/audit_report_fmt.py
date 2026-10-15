#!/usr/bin/env python3
"""审计周报格式化 → stdout (供 audit_weekly_feishu.sh 投递)"""
import json
import sys

sys.path.insert(0, os.environ.get("HERMES_SCRIPTS", os.path.expanduser("~/.agent_ctx/scripts")))
from audit_ledger import report

d = report(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
lines = [
    "📊 审计者自动化·周报",
    f"自检覆盖率: {d['self_check_coverage']:.1%} (自检拦 {d['self_caught']} / 您抓 {d['user_caught']})",
    "曲线目标: 该比率周周上升 → 您从审计环里退出去",
    "明细: " + " | ".join(f"{k}: {v}" for k, v in d["by_detector"].items()),
]
print("\n".join(lines))
