#!/bin/bash
# B 方案编排 (2026-11-22, 用户批准): 停 vLLM → 清 47G 磁盘 KV 缓存 → 启清理服务 → 带 recompute 补丁重启 → 等健康 → 写报告
# 由 systemd-run --user 脱离 Hermes cgroup 运行, 进度写 PROG, 最终报告写 REPORT
PROG=/home/l/kv-b-restart.progress
REPORT=/home/l/kv-b-restart.report
exec > "$PROG" 2>&1
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus

echo "=== $(date '+%F %T') B 方案开始 ==="
/home/l/stop-dflash2.sh
echo "$(date '+%T') 已发停服命令 (按进程名杀, 网关不动)"

# 等两卡显存释放 (<1500MiB), 最多 3 分钟
m0=99999; m1=99999
for i in $(seq 1 36); do
  m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 99999)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 99999)
  [ "$m0" -lt 1500 ] && [ "$m1" -lt 1500 ] && break
  sleep 5
done
echo "$(date '+%T') GPU 显存已释放: ${m0}/${m1} MiB"

# 清磁盘 KV 缓存 (只删 checkpoint-* 纯缓存) + 两个实例的旧 shm mmap (腾 ~1.9G 内存)
before=$(df --output=avail -B1 / | tail -1)
rm -rf /home/l/vllm-kv-cache/checkpoint-*
rm -f /dev/shm/vllm_offload_*.mmap
after=$(df --output=avail -B1 / | tail -1)
echo "$(date '+%T') 清理完成: 释放 $(( (after - before) / 1073741824 )) GiB, 磁盘可用 $(df -h / | tail -1 | awk '{print $4}')"

# 清理守护服务 (开机自启 + 立即启动, 上限 40G 热可调)
systemctl --user enable --now kv-cache-cleaner.service
echo "$(date '+%T') kv-cache-cleaner: $(systemctl --user is-active kv-cache-cleaner.service) (上限 $(grep ^LIMIT_GB /home/l/.config/kv-cleaner.conf))"

# 重启 vLLM: 必须 systemd-run 脱离 cgroup + KillMode=none (2026-11-21 血泪)
UNIT=vllm-manual-$(date +%H%M)
systemd-run --user --collect --unit="$UNIT" --property=KillMode=none --property=RemainAfterExit=yes /home/l/start-dflash2.sh
echo "$(date '+%T') 重启已发起: $UNIT (指纹~9分 + 权重~9分 + 预热, 预计 25~33 分钟)"

# 等健康, 最多 40 分钟
ok=0; code=000
for i in $(seq 1 80); do
  sleep 30
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health 2>/dev/null)
  [ "$code" = "200" ] && ok=1 && break
done

{
  echo "=== B 方案最终报告 $(date '+%F %T') ==="
  if [ "$ok" = "1" ]; then
    echo "✅ vLLM 已恢复 (health=200)"
    pid=$(pgrep -f 'entrypoints.openai.api_serve[r]' | head -1)
    echo "读盘失败策略: $(tr '\0' '\n' < /proc/$pid/cmdline | grep -o '\"kv_load_failure_policy\":\"[a-z]*\"' | head -1) (应为 recompute)"
    grep -h 'GPU KV cache size' "$(ls -t /home/l/vllm-2080ti-0.2.x/run-logs/*.log | head -1)"
    smoke=$(curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen38-27b-nvfp4-dflash2-fp8kv-1x262k-text-image-cu130","messages":[{"role":"user","content":"只回复两个字：正常"}],"max_tokens":20}')
    echo "真实推理冒烟: $(echo "$smoke" | grep -o '\"completion_tokens\":[0-9]*' | head -1) (>0 即正常)"
    echo "磁盘: $(df -h / | tail -1 | awk '{print "已用 "$3" / 可用 "$4}')"
    echo "kv-cache-cleaner: $(systemctl --user is-active kv-cache-cleaner.service), $(grep ^LIMIT_GB /home/l/.config/kv-cleaner.conf)"
    echo "网关: $(systemctl is-active vllm-alias-proxy)"
  else
    echo "❌ 40 分钟内未恢复 (health=$code), 需要人工介入"
    echo "--- 启动单元日志尾 ---"
    journalctl --user -u "$UNIT" -n 30 --no-pager
    echo "--- 模型日志尾 ---"
    tail -20 "$(ls -t /home/l/vllm-2080ti-0.2.x/run-logs/*.log | head -1)"
  fi
} > "$REPORT"
echo "$(date '+%T') 最终报告已写入 $REPORT"
cat "$REPORT"
