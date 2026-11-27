#!/usr/bin/env python3
"""kv-cache-cleaner v2: vLLM 磁盘 KV 前缀缓存的限额清理守护脚本 (LRU + 高低水位)。

只删 CACHE_DIR/checkpoint-*/ 下的 *.bin / *.tmp 块文件, 永不碰 config.json、
永不越出 CACHE_DIR、不跟随符号链接。

相对 v1 的改动 (对应 TODO.md 第 1 条 "顶格删除抖动"):

1. 淘汰顺序: 按 "最近使用时间" = max(mtime, atime) 从旧到新删, 而不是只看 mtime。
   块文件 "只写一次、已存在就跳过", mtime 永远是创建时间; 长会话开头 (system prompt /
   早期上下文) 创建得最早、却最常被复用, v1 会最先删它们, 正好撞上读盘。
   前提: 数据盘 atime 要真实更新, 见下方 "atime 前提"。
2. 高低水位: 总量 > LIMIT_GB 触发, 一口气删到 LIMIT_GB - LOW_WATER_GB。v1 删到刚好
   低于上限, 之后每轮 (60s) 都在边缘小批量删, 等于永远 "顶格"。
3. 宽限期同样按最近使用时间算: 最近 GRACE_MIN 分钟内被写 *或读* 的块不删。
4. 诊断: 每次清理日志里带 "其中曾被读过 N 个" (atime 明显晚于 mtime), 用来验证
   "热块被误删" 的假设是否成立, 并在 atime 不可用时告警。
5. 可选 FSTRIM_CMD: 批量删除后限频执行 (DRAM-less 盘不 TRIM 会一直认为块在用)。
6. --once / --dry-run: 单轮运行 / 只打印不删, 方便验证。

atime 前提 (二选一):
  a. 数据盘挂载选项用 strictatime (relatime 下 atime 只在 "atime<=mtime" 或超过 24h 时才
     更新, 同一块被反复读不会刷新)。例: /etc/fstab 里把该盘选项改成 defaults,strictatime
     后 `mount -o remount,strictatime <挂载点>`。
  b. 或在 vLLM 读盘命中路径 (补丁 0006 的 fs/manager.py) 对读到的块 os.utime(p, None)。
  两者都不满足时 (noatime 等) 本脚本退化成 v1 的按 mtime 淘汰, 并打 WARN。

配置文件 ~/.config/kv-cleaner.conf 每轮重读, 无需重启。

安全前提: vLLM 侧 kv_load_failure_policy=recompute (补丁 0001)。该策略下删掉正在被
引用的块最坏只是重算前缀, 不报错。
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

CONF_PATH = os.path.expanduser("~/.config/kv-cleaner.conf")
DEFAULTS = {
    "CACHE_DIR": "/home/l/vllm-kv-cache",
    "LIMIT_GB": "40",
    "LOW_WATER_GB": "10",   # 触发后删到 LIMIT_GB - LOW_WATER_GB
    "INTERVAL_SEC": "300",
    "GRACE_MIN": "10",
    "SORT_KEY": "lru",      # lru = max(mtime, atime); mtime = 退回 v1 行为
    "FSTRIM_CMD": "",       # 例: sudo -n /usr/sbin/fstrim -v /home ; 留空不执行
    "FSTRIM_MIN_FREED_GB": "5",
    "FSTRIM_MIN_INTERVAL_MIN": "60",
}
GB = 1024 ** 3
HOT_DELTA_SEC = 60  # atime 比 mtime 晚超过这么久, 认为 "创建之后又被读过"


def log(msg):
    print(f"[kv-cleaner] {msg}", flush=True)


def read_conf():
    conf = dict(DEFAULTS)
    try:
        with open(CONF_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([A-Z_]+)\s*=\s*(.+)$", line)
                if m and m.group(1) in DEFAULTS:
                    conf[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except OSError as e:
        log(f"WARN 读配置失败, 用默认值: {e}")
    return conf


def atime_mount_options(path):
    """返回 path 所在挂载点的 (挂载点, 选项集合); 读不到返回 (None, set())。"""
    best, best_opts = None, set()
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                parts = line.split()
                mnt, opts = parts[4], set(parts[5].split(","))
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and \
                        (best is None or len(mnt) > len(best)):
                    best, best_opts = mnt, opts
    except OSError:
        return None, set()
    return best, best_opts


def warn_if_atime_unreliable(cache_dir, sort_key):
    if sort_key != "lru":
        return
    mnt, opts = atime_mount_options(cache_dir)
    if mnt is None:
        return
    if "noatime" in opts:
        log(f"WARN {mnt} 挂了 noatime: atime 不更新, LRU 退化成按 mtime 淘汰 "
            f"(与 v1 相同)。改 strictatime, 或在读盘路径里 os.utime。")
    elif "relatime" in opts:
        log(f"WARN {mnt} 挂了 relatime: 块被读一次后 atime 才更新, 反复读不刷新, "
            f"LRU 只是近似。建议 remount,strictatime。")


def collect(cache_dir, grace_sec, sort_key):
    """返回 (总字节, 候选列表[(最近使用时间, 大小, 路径, 是否曾被读过)])。

    候选 = 最近使用时间早于宽限期的块文件; 最老在前。
    """
    total = 0
    candidates = []
    cutoff = time.time() - grace_sec
    for entry in os.scandir(cache_dir):
        if not entry.name.startswith("checkpoint-") or not entry.is_dir(follow_symlinks=False):
            continue
        for root, dirs, files in os.walk(entry.path, followlinks=False):
            for name in files:
                if not (name.endswith(".bin") or name.endswith(".tmp")):
                    continue
                p = os.path.join(root, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                total += st.st_size
                key = max(st.st_mtime, st.st_atime) if sort_key == "lru" else st.st_mtime
                if key < cutoff:
                    hot = (st.st_atime - st.st_mtime) > HOT_DELTA_SEC
                    candidates.append((key, st.st_size, p, hot))
    candidates.sort()  # 最老(最久没用)在前
    return total, candidates


def prune_empty_dirs(cache_dir):
    """自底向上删空目录, 只在 cache_dir 内部, 绝不删 checkpoint 根。"""
    for entry in os.scandir(cache_dir):
        if not entry.name.startswith("checkpoint-") or not entry.is_dir(follow_symlinks=False):
            continue
        for root, dirs, files in os.walk(entry.path, topdown=False, followlinks=False):
            if root == entry.path:
                continue  # checkpoint 根保留 (config.json 元数据层)
            try:
                if not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass


def vllm_policy_is_fail():
    """检测运行中的 api_server 是否仍用 fail 策略 (补丁需重启才生效)。"""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        if "vllm.entrypoints.openai.api_server" in cmd and "kv-transfer-config" in cmd:
            return "kv_load_failure_policy" not in cmd or '"recompute"' not in cmd
    return False  # 没在跑, 无所谓


def maybe_fstrim(conf, freed, state):
    cmd = conf["FSTRIM_CMD"].strip()
    if not cmd:
        return
    try:
        min_freed = float(conf["FSTRIM_MIN_FREED_GB"]) * GB
        min_gap = float(conf["FSTRIM_MIN_INTERVAL_MIN"]) * 60
    except ValueError:
        return
    state["freed_since_trim"] += freed
    now = time.time()
    if state["freed_since_trim"] < min_freed or now - state["last_trim"] < min_gap:
        return
    try:
        r = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=900)
        out = (r.stdout or r.stderr).strip().replace("\n", " ")
        log(f"fstrim rc={r.returncode}: {out[:200]}")
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"WARN fstrim 失败: {e!r}")
    state["last_trim"] = now
    state["freed_since_trim"] = 0


def run_once(conf, state, dry_run=False):
    try:
        limit = float(conf["LIMIT_GB"]) * GB
        low = max(0.0, float(conf["LOW_WATER_GB"])) * GB
        grace = max(0, int(conf["GRACE_MIN"])) * 60
        interval = max(30, int(conf["INTERVAL_SEC"]))
    except ValueError:
        log("WARN 配置数值非法, 本轮用默认 40GB/10GB 低水位/300s/10min")
        limit, low, grace, interval = 40 * GB, 10 * GB, 600, 300
    low = min(low, limit)  # 低水位不能把目标压成负数
    target = limit - low
    sort_key = conf["SORT_KEY"] if conf["SORT_KEY"] in ("lru", "mtime") else "lru"

    cache_dir = os.path.realpath(conf["CACHE_DIR"])
    if not cache_dir.startswith("/home/") or cache_dir == "/home":
        log(f"ERROR CACHE_DIR 非法({cache_dir}), 本轮跳过")
        return interval
    if not os.path.isdir(cache_dir):
        if state["loop"] % 12 == 0:
            log(f"目录不存在({cache_dir}), 等待")
        state["loop"] += 1
        return interval

    if not state["atime_checked"]:
        warn_if_atime_unreliable(cache_dir, sort_key)
        state["atime_checked"] = True

    total, candidates = collect(cache_dir, grace, sort_key)
    deleted = freed = hot_deleted = 0
    if total > limit:
        now = time.time()
        if now - state["last_policy_warn"] > 3600 and vllm_policy_is_fail():
            log("WARN 运行中的 vLLM 仍是 fail 策略: 删块可能让个别请求报错; "
                "重启 vLLM 后(recompute 补丁生效)即完全安全")
            state["last_policy_warn"] = now
        oldest = newest = None
        for key, size, p, hot in candidates:
            if total <= target:
                break
            if not os.path.realpath(p).startswith(cache_dir + os.sep):
                continue  # 双保险: 路径必须在 cache_dir 内
            if not dry_run:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    continue  # 已被别人删
                except OSError as e:
                    log(f"WARN 删除失败 {p}: {e}")
                    continue
            total -= size
            freed += size
            deleted += 1
            hot_deleted += 1 if hot else 0
            oldest = key if oldest is None else oldest
            newest = key
        if deleted:
            if not dry_run:
                prune_empty_dirs(cache_dir)
            age = lambda t: f"{(time.time() - t) / 3600:.1f}h"
            tag = "[DRY-RUN 未实删] " if dry_run else ""
            log(f"{tag}清理 {deleted} 个块, 释放 {freed / GB:.2f} GiB, 剩余 {total / GB:.2f} GiB "
                f"(上限 {limit / GB:.0f}, 目标 {target / GB:.0f}); 最近使用距今 "
                f"{age(newest)}~{age(oldest)}, 其中创建后被读过 {hot_deleted} 个")
            if not dry_run:
                maybe_fstrim(conf, freed, state)
        elif candidates == []:
            log(f"WARN 超限 ({total / GB:.2f}>{limit / GB:.0f} GiB) 但没有超过宽限期的候选块")
    state["loop"] += 1
    if deleted == 0 and state["loop"] % 12 == 1:  # 每小时一条心跳
        log(f"OK 当前 {total / GB:.2f} GiB / 上限 {limit / GB:.0f} GiB, 宽限 {grace // 60} min, 无需清理")
    return interval


def main():
    ap = argparse.ArgumentParser(description="vLLM 磁盘 KV 缓存限额清理 (LRU + 高低水位)")
    ap.add_argument("--once", action="store_true", help="只跑一轮就退出")
    ap.add_argument("--dry-run", action="store_true", help="只打印会删什么, 不实际删除")
    args = ap.parse_args()

    log(f"启动 v2, 配置={CONF_PATH} (每轮重读)")
    state = {"loop": 0, "last_policy_warn": 0.0, "atime_checked": False,
             "last_trim": 0.0, "freed_since_trim": 0}
    while True:
        interval = 300
        try:
            interval = run_once(read_conf(), state, dry_run=args.dry_run)
        except Exception as e:  # 守护脚本绝不因单轮异常退出
            log(f"ERROR 本轮异常: {e!r}")
        if args.once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
