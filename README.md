# vllm-2080ti-service

双 RTX 2080 Ti 22G（魔改卡）+ 老 X99 平台上，本地 Qwen3.8-27B NVFP4 推理服务的**整套部署方案**：启动脚本、局域网网关、KV 磁盘缓存治理、systemd 服务、以及打在 vLLM fork 上的 6 处自研补丁。

基于公开的 [vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)（fork 自上游 vLLM）构建，代码树 `v0.2.2-post3 (93c9013)`。本仓库收录的是运行这套服务所需的本机侧脚本、配置与补丁，不含上游源码，也不含模型权重。

## 硬件环境

| 项 | 值 |
|---|---|
| GPU | 2× RTX 2080 Ti 22G（2066S 魔改，NVLink 桥，TP2 张量并行） |
| CPU | Intel X99 单路（E5 系，decode 阶段是 CPU 瓶颈，governor 常停在低频） |
| 内存 | 32 GiB |
| 系统盘 | 173G NVMe；独立数据盘 300G（ext4，专用于磁盘 KV 缓存） |
| 软件 | Ubuntu 26.04 / 驱动 595.91 / CUDA 13.0 / PyTorch 2.13 / Python 3.12 |

## 服务链

```
客户端(局域网) ──:8001 OpenAI兼容──┐
                ──:11434 Ollama shim──┤→ model-alias-proxy(网关) → 127.0.0.1:8000 vLLM(TP2)
```

| 组件 | 文件 | systemd 单元 |
|---|---|---|
| vLLM 主服务 | `scripts/start-dflash2.sh` / `stop-dflash2.sh` | `systemd/vllm-dflash2.service`（oneshot，根用户级单元） |
| 别名网关 | `gateway/model-alias-proxy.py`（纯 stdlib，零依赖） | `systemd/vllm-alias-proxy.service` |
| KV 缓存清理 | `tools/kv-cache-cleaner.py` + `systemd/kv-cleaner.conf` | `systemd/kv-cache-cleaner.service`（用户级，linger） |
| 重启编排 | `tools/kv-b-orchestrator.sh`（停服→清缓存→重启→冒烟→报告） | 按需手动/定时触发 |

当前路线：`Qwen3.8-27B-NVFP4-v2`（w4a16）+ `Qwen3.8-27B-DFlash2` 草稿模型（投机 7 token，greedy 草稿采样），fp8 KV，262144 满上下文，`MAX_NUM_SEQS=2`，prefix caching + 磁盘前缀缓存。

### 网关的关键行为（为什么必须有它）

- **max_tokens 兜底注入（65536）**：客户端不传时强制注入。否则单条请求最坏生成到 262K token，把整个局域网钉死一个多小时。
- **MAX_INFLIGHT=2 门口排队**：信号量限并发（队满 429、超时 503），与服务端槽位对齐。
- 客户端中途断开时**主动关闭上游**——否则孤儿请求占死 KV 槽。
- `reasoning_effort=high` 自动改写为 `xhigh`；`developer` role 归一为 system；流式自动补 `include_usage`。
- 模型名任意（如 `qwen3.8-27b`），网关改写为真实 served name；同时提供 Ollama 兼容层（`/api/tags`、NDJSON `/api/chat`），老客户端零改动迁移。

## KV 分层架构（本方案的核心）

这台机器的显存/内存/磁盘不是"容量叠加"，而是三层各司其职：

1. **显卡 = 灶**：KV 池 ~272K tokens。要么 1×256K 独占，要么 2×128K 并发。
2. **内存 = 镜像（12G，与显存池同宽）**：只存**最近下灶的一条会话**。不扩内存塞两份镜像——超预算的切换本来就该走盘。
3. **硬盘 = 仓库（目标：显存全量）**：GPU **边算边写盘**，块按内容哈希命名、只写一次（已存在则跳过）。重启不亏，盘上全有。上限由 kv-cleaner 兜底（超限删最老，带宽限期）。

会话切换三档，慢就慢在后两档，是设计内降级不是 bug：

| 档 | 路径 | 实测 |
|---|---|---|
| 镜像命中 | 纯内存 | 120K 上下文续问 6s（缓存命中 97.4%） |
| 镜像没有 | 并行读盘（补丁 6：4 块/组 × 16 线程，55→~220 MB/s） | ~45-75s / 10G |
| 盘也没有 | 全量重算 | 冷建 256K 会话 TTFT ~460s |

正确性验证：120K 随机文本里藏暗号，冷建 / 镜像切回 / 再回锅三种路径答案逐字一致。

## patches/ —— 6 处 fork 补丁

对 Definitive fork 的本地改动（`git diff` 导出，基线 93c9013e37）。**上游 `update.sh` 覆盖 launcher.sh 后必须重新打**。按顺序 `git apply`：

| 补丁 | 作用 |
|---|---|
| `0001` launcher.sh | `ALLOW_KV_OVERCOMMIT=1` 放宽"每槽满上下文"KV 预检（本机 272K 池 < 2×262K）；`kv_load_failure_policy=recompute`（运行中删盘块只重算不报 500——这是 kv-cleaner 敢在线删块的**安全前提**）；`KV_MEM_ONLY=1` 只建内存层不挂盘层 |
| `0002` checkpoint_fingerprint.py | 权重指纹缓存：stat 不变则秒回，启动省 ~13 分钟（实测全程 25-33min → ~14min） |
| `0003` shared_offload_region.py | EFAULT 也视为 cudaHostRegister 不支持，直接走逐页写 fallback |
| `0004` cpu/manager.py | KV_PASSTHRU 直通 + 补丁 7B：在途块 pin 名单，防被 LRU 误逐 |
| `0005` tiering/manager.py | 补丁 7B 配套：盘签收后转正释放，镜像队列不被写盘尾巴挤爆 |
| `0006` tiering/fs/manager.py | 补丁 8+9：读盘/写盘并行组（`KV_FS_PARALLEL` / `KV_FS_STORE_GROUP` / `KV_FS_READ_THREADS` / `KV_FS_WRITE_THREADS`）。写盘原来单线程 ~55MB/s 是全链路最窄处 |

### 死路清单（我们都试过了，别再试）

- **动块的存亡/归还时机**（落盘即释放、LRU 镜像保护、SGLang 式写回）→ 必崩在 `offloading/scheduler.py` 的 free_list 账本断言。vLLM 禁区。
- **跳过 cudaHostRegister 的"lazy-pin"** → 表面省内存，实际 store DMA 全废（`cannot store chunks` 刷屏、切回=全量重算）。tmpfs 本来就按需占物理页，注册只是标 DMA 范围。
- `madvise(POPULATE_WRITE)` 探测在本内核 EFAULT。
- 换 SGLang：NVFP4/Turing 不支持。

## profiles/

从 fork 仓库带出的本机相关路由档（含已停用的历史档，见 `profiles/2x2080Ti/README.md` 的证据口径说明）。当前主力：`2x2080Ti/qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env`。

## 部署到新机器

1. 按上游 README 构建 vLLM 2080 Ti Definitive（venv + CUDA 13 + editable 安装）。
2. 下载 `Qwen3.8-27B-NVFP4-v2` 与 `Qwen3.8-27B-DFlash2` 权重，路径写进 `scripts/start-dflash2.sh`。
3. 依次 `git apply patches/*.patch`（在 fork 仓库根目录）。
4. 改 `start-dflash2.sh` / 网关 / cleaner 里的路径、端口、盘容量上限（`kv-cleaner.conf` 的 `LIMIT_GB`）。
5. 装三个 systemd 单元，`kv-cache-cleaner` 走用户级 + `loginctl enable-linger`。

### 已知大坑（实测踩过）

- **停止必须按进程名杀**（`stop-dflash2.sh` 已内置）：launcher 用 setsid 把 api_server 甩出 systemd cgroup，`systemctl stop` 杀不干净，worker 残留显存。
- **失败启动残留 ~1.1 GiB 显存** → 下次启动报 "Free memory less than 0.95"。启动脚本内置 `wait_gpu_free` 重试。
- **桌面环境常驻 GPU 1 约 700 MiB**：gnome-shell/Xwayland 不可退，gpu-util 只能 0.93-0.96 之间找平衡。
- **agent/CI 里直接启动会进 oomd 靶心**：用 `systemd-run --user --property=KillMode=none ...` 脱离调用方 cgroup，否则启动到一半被整组 OOM-kill。
- 磁盘 KV 目录**没有自动淘汰**（fs 层无 quota/evict 逻辑），不管它一天能堆 47G 塞满盘；kv-cleaner 就是补这个的。
- 权重自洽性：NVFP4 权重 index 要 `lm_head.weight_scale`，缺了 lm_head 未初始化 → 乱码 + 投机接受率 0。
- `temperature` 实测：T=1 不比 T=0 慢（拒绝采样比严格匹配对草稿更宽容，接受长度反而更高）。
- `xhigh` 思考档在部分题上过度反刍（思考 2.5 万字顶满 max_tokens、正文 0 字），不是"更聪明"的通用开关。

## 待解决问题

见 [TODO.md](TODO.md)——190G 顶格抖动观察、盘速硬件瓶颈、update.sh 覆盖补丁、读盘档精确测试、网关预取（未实现）、CPU 降频、IP 漂移等，按影响排序，附死路备忘。

## 许可与致谢

- 上游：[vllm-project/vllm](https://github.com/vllm-project/vllm)（Apache-2.0）
- fork：[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) — 本仓库的补丁与 profile 均基于该项目，遵循其 AGENTS.md 的署名要求
- 本仓库脚本/补丁/文档：MIT
