#!/usr/bin/env bash
# mem-engine-service 健康看门狗 (2026-10-15 钩子生产化·daemon 稳定件)
# 每次运行: embed 探活 → 连续 2 次失败 → systemctl restart mem-engine-service (自愈)
# 状态文件: /tmp/embed_watchdog.fail (连续失败计数)
# 豁免窗口: ninfer.service 卸载/回装期间 (GPU_WINDOW 文件存在时跳过, 防误杀训练期 engine)

STATE=/tmp/embed_watchdog.fail
[ -f /tmp/gpu_window_open ] && { echo "[wd] GPU 窗口期, 跳过探活"; exit 0; }

R=$(timeout 10 curl -s -m 8 -X POST "${EMBED_SERVICE_URL:-http://127.0.0.1:8766/v1/embed}" \
    -H 'Content-Type: application/json' \
    -d '{"input":["watchdog probe"],"query":true}' 2>/dev/null | head -c 20)

case "$R" in
  *"[["*)
    rm -f "$STATE"; echo "[wd] OK"; exit 0 ;;
  *)
    N=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
    echo "$N" > "$STATE"
    echo "[wd] 探活失败 #$N: ${R:-<empty>}"
    python3 -c "
import sys; sys.path.insert(0, '$HOME/.hermes/scripts')
from audit_ledger import log
log('self_catch', 'watchdog', f'embed 探活失败 #{int(open(\"/tmp/embed_watchdog.fail\").read())}')" 2>/dev/null
    if [ "$N" -ge 2 ]; then
      echo "[wd] 连续失败≥2 → restart mem-engine-service (自愈)"
      systemctl --user restart mem-engine-service.service
      python3 -c "
import sys; sys.path.insert(0, '$HOME/.hermes/scripts')
from audit_ledger import log
log('self_catch', 'watchdog', 'watchdog 触发自愈 restart mem-engine-service')" 2>/dev/null
      rm -f "$STATE"
    fi
    ;;
esac
