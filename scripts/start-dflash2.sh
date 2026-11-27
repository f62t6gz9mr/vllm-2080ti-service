#!/bin/bash
# 官方 DFlash2 方案启动脚本
# vLLM 2080 Ti Definitive Edition (CUDA 13 / PyTorch 2.13 / Py3.12)，代码树 v0.2.2-post3 (93c9013)
# 2026-10-02 随 v0.2.2-post3 调整：官方 profile 目录扁平化，fast/ 与 normal/ 子目录被删，
# w4a16 下不再有 MTP text+image 档，且路线 profile 只允许固定键（SERVED_NAME 被禁）。
# 原来那份本地 8 并发档已移植到 profiles/local/...（update.sh 会保留 profiles/local）。
# 旧的 dflash2 fast 档参考：2x2080Ti/qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env
set -u
cd /home/l/vllm-2080ti-0.2.x || exit 1
export HOME=/home/l
export USER=l LOGNAME=l
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/usr/local/cuda-13.0/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export START_TIMEOUT=2400
# 2026-11-21: 放开 launcher 的"每槽满上下文"KV 预检（见 launcher.sh validate_configured_kv_concurrency）。
# 没有这个开关，2 x 262144 = 52.4 万 > 本机 KV 池 26.4 万，启动会被拦。
export ALLOW_KV_OVERCOMMIT=1
export KV_WRITEBACK=0
export KV_PASSTHRU=1
# 2026-11-27 补丁9: 写盘并行组 + 读写线程 env 可调 (fs/manager.py, 备份 .bak-p9-20261127)。
# 写以前单线程一包端(~55-93MB/s)是显卡->内存->盘链路最窄处; 现在 4 砖/组并行(~220MB/s),
# 写不堵 12G 装卸台 => 过路砖及时签收释放, 镜像不被挤爆。
export KV_FS_STORE_GROUP=4
export KV_FS_PARALLEL=4
# 要再压榨可动: KV_FS_READ_THREADS / KV_FS_WRITE_THREADS (默认 16/16)
# 2026-09-19 根因：旧的 ~/models/Qwen3.8-27B-NVFP4 损坏（index 要 lm_head.weight_scale，
# 但权重里只有 BF16 lm_head.weight 且无 scale）-> lm_head 未初始化 -> 出词乱码 -> DFlash2 接受率 0。
# 现换成自洽的上游原版（FP8 lm_head + weight_scale，1953 张量）。
# R0结束：缓存无罪，恢复官方缓存(把下行改回 export DISABLE_PREFIX_CACHING=1 可重做R0)
unset DISABLE_PREFIX_CACHING
# 局域网可访问：launcher 会改绑 0.0.0.0:8000（SERVICE_SCOPE=lan，launcher.sh:5329）
export SERVICE_SCOPE=lan

LAUNCH_ARGS=(
  --model-dir /home/l/models/Qwen3.8-27B-NVFP4-v2
  # 2026-10-02 (v0.2.2-post3)：改回官方 DFlash2 档。
  #   - 0.2.1-RC 时 R1 结论是 dflash 有回滚 bug；post3 的 #241/#243 在 2x2080Ti 的
  #     DFlash2 路线上重新验证过重启/前缀缓存，CHANGELOG 也列了该路线实测。
  #   - 另外 v0.2.2 新加了严格 KV 并发预检：要求 KV tokens >= MAX_NUM_SEQS x MAX_MODEL_LEN，
  #     本机实测只有 379,448 tokens（MTP4 档），旧的 8 x 252000 本地档已无法启动，
  #     所以那份 local MTP 档先停用（文件还在 profiles/local/...）。
  # 回滚到 MTP：--profile local/2x2080Ti/qwen27b/w4a16/mtp4-fp8kv-8x252K-text-image.env
  #             --mode normal + --set MAX_NUM_SEQS=1（否则预检会拦），并去掉下面两行 dflash 相关。
  --profile 2x2080Ti/qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env
  --mode fast
  --speculative-model /home/l/models/Qwen3.8-27B-DFlash2
  # SERVED_NAME 不再能写进路线 profile（v0.2.2 收紧了允许键），这里显式传，
  # 需与 ~/bin/model-alias-proxy.py 里的 REAL_MODEL 一致，否则 Open WebUI 侧模型名会断。
  --set SERVED_NAME=qwen38-27b-nvfp4-dflash2-fp8kv-1x262k-text-image-cu130
  --gpu-devices 0,1
  --tp-size 2
  --port 8000
  # 2026-09-22: 8 并发档需要多留 KV 头部，按 dflash 路线回 0.95
  # （单槽 A/B 档当时降到 0.93 + profile 230000，见 .bak-1seq-230k 备份注释）
  # 2026-11-21: 开磁盘缓存 + 2 槽后 0.95 只剩 4.93GiB KV，比满血 262144 所需 4.96GiB 差 0.03GiB
  # （报错 kv_cache_utils.py:905 want 4.96 got 4.93），按官方文档建议提到 0.96。
  --gpu-util 0.96
  # 2026-11-21: 开 SSD 前缀缓存（launcher 的 KV_DISK_* 开关，接 OffloadingConnector 的 fs 层）。
  # 作用：服务重启后相同前缀直接读盘，不用重新 prefill；不增加 GPU KV 容量，不改变并发数。
  # 代价：CPU 暂存层 = 落盘流水线的装卸台，4G ≈ 16 块(1648 tok/块) ≈ 2.6 万 token 上下文。
  # 2026-11-22 由 1G(仅 4 块) 提到 4G：1G 时大会话前缀块来不及落盘，6.6 万条
  # "cannot store chunks"，重复长上下文 TTFT 86s 重算。本机内存 14GB，32G 条插上后
  # 再考虑提到 16-20G。落盘目录由 kv-cache-cleaner 守护限额(250G)。
  # PYTHONHASHSEED=0 由 launcher 自己固定。
  # 验证：重启后看 /metrics 的 vllm:external_prefix_cache_hits_total 和
  # vllm:kv_offload_tiering_chunk_hits_total。要关：删掉这两行即可。
  --set KV_DISK_CACHE_DIR=/home/l/vllm-kv-cache
  --set KV_DISK_CPU_BYTES=12884901888
  # 2026-11-21: 多并发 2 槽。KV 池 26.4 万 tokens 由所有会话共享，不再要求"每槽满 262K"。
  # 官方预检 "MAX_NUM_SEQS x MAX_MODEL_LEN <= KV池" 会把 2x262K 拦掉，所以用本地补丁开关
  # ALLOW_KV_OVERCOMMIT=1（改在 launcher.sh validate_configured_kv_concurrency，带注释）。
  # 代价：两个会话同时接近满上下文时会互相抢占（scheduler 把被抢的请求 num_computed_tokens=0，
  # 已算完的前缀块若已落盘，回来可从盘里读回，不用全重算）。
  # 真正的并发上限由网关 :8001 的 MAX_INFLIGHT 控制（默认 2，见 model-alias-proxy.py）。
  # launcher.sh 被 update.sh 覆盖后需重打本补丁，diff 见 launcher.sh.bak-kvovercommit-20261121。
  --set MAX_NUM_SEQS=2
  # 工具调用：客户端带 tools / tool_choice=auto 时必需，否则 vLLM 直接回 400。
  # 该模型 chat template 用 XML 风格包裹工具调用；本 fork 里 qwen3_xml 与 qwen3_coder
  # 都指向同一个 Qwen3EngineToolParser。官方这条 fast profile 本身没开工具调用。
  --set ENABLE_AUTO_TOOL_CHOICE=1
  --set TOOL_CALL_PARSER=qwen3_xml
  # DFlash2 档自带 SPECULATIVE_TOKENS=7 / MAX_NUM_SEQS=1，不要再 --set 覆盖。
  # （历史：2026-09-20 并发 1 -> 8；投机步数试过 4，真实中文 +4.6%，但 32K 长代码 -27%）
  # 2026-09-22 用户要求恢复多并发：T OK / S4 / M252000，
  # profile 里的 MAX_MODEL_LEN 已恢复到 252000（不吃 --set，改 profile 文件），
  # --gpu-util 从 0.93 回到 0.95（8 x 252K 的 KV 池需要多留头部；单槽档备份见
  # start-dflash2.sh.bak-8seq-spec4，单槽 profile 备份在
  # profiles/.../mtp-fp8kv-1x240k-text-image.env.bak-1seq-230k）。
  # 想回到单槽 A/B 档：把下面两行取消注释换成 MAX_NUM_SEQS=1 / SPECULATIVE_TOKENS=3，
  # 同时把 profile 改回 .bak-1seq-230k（MAX_MODEL_LEN=230000, GPU_UTIL=0.96, spec3）。
  # --set MAX_NUM_SEQS=8        # MTP 多并发档专用，DFlash2 单槽档不要打开
  # --set SPECULATIVE_TOKENS=4  # 同上，DFlash2 官方档是 7
)

# 失败后必须等显存真的降下来再重试：实测失败尝试的残留进程会占住 ~1.1 GiB，
# 导致下一次报 "Free memory on device cuda:1 (20.35/21.48 GiB) ... less than 0.95" —— 只差 60 MiB。
wait_gpu_free() {
  local i hi
  for i in $(seq 1 36); do
    hi=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -n "${hi}" ] && [ "${hi}" -lt 1500 ] && return 0
    sleep 5
  done
  return 1
}

ATTEMPT=1
MAX_ATTEMPT="${DFLASH_BOOT_ATTEMPTS:-3}"
while :; do
  if ./launcher.sh "${LAUNCH_ARGS[@]}"; then
    exit 0
  fi
  echo "[start-dflash2] 第 ${ATTEMPT}/${MAX_ATTEMPT} 次启动失败" >&2
  /home/l/stop-dflash2.sh
  if [ "${ATTEMPT}" -ge "${MAX_ATTEMPT}" ]; then
    echo "[start-dflash2] 已达最大重试次数，放弃" >&2
    exit 1
  fi
  ATTEMPT=$((ATTEMPT + 1))
  if wait_gpu_free; then
    echo "[start-dflash2] 显存已释放，10 秒后重试" >&2
    sleep 10
  else
    echo "[start-dflash2] 等 180 秒显存仍没降下来（桌面占用过高？），仍再试一次" >&2
  fi
done
