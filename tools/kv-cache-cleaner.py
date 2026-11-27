#!/usr/bin/env python3
"""kv-cache-cleaner: vLLM 磁盘 KV 前缀缓存的限额清理守护脚本。

只删 /home/l/vllm-kv-cache/checkpoint-*/ 下的 *.bin / *.tmp 块文件,
按 mtime 从最老开始删, 直到总量 <= LIMIT_GB。永不碰 config.json、
永不越出 CACHE_DIR、不跟随符号链接。

配置文件 ~/.config/kv-cleaner.conf 每轮循环重读, 改 LIMIT_GB=80 后
最多 INTERVAL_SEC 秒生效, 无需重启本服务。

安全前提: vLLM 侧 kv_load_failure_policy=recompute (launcher.sh 本地补丁,
2026-11-22 起默认)。该策略下删掉正在被引用的块最坏只是重算前缀, 不报错。
若检测到运行中的 vLLM 仍是 fail 策略, 会周期性打 warning。
"""

import os
import re
import sys
import time

CONF_PATH = os.path.expanduser("~/.config/kv-cleaner.conf")
DEFAULTS = {
    "CACHE_DIR": "/home/l/vllm-kv-cache",
    "LIMIT_GB": "40",
    "INTERVAL_SEC": "300",
    "GRACE_MIN": "10",
}
GB = 1024 ** 3


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


def collect(cache_dir, grace_sec):
    """返回 (总字节, 可删候选列表[(mtime, size, path)])。候选=超过宽限期的块文件。"""
    total = 0
    candidates = []
    now = time.time()
    cutoff = now - grace_sec
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
                if not os.path.isfile(p) or os.path.islink(p):
                    continue
                total += st.st_size
                if st.st_mtime < cutoff:
                    candidates.append((st.st_mtime, st.st_size, p))
    candidates.sort()  # 最老在前
    return total, candidates


def prune_empty_dirs(cache_dir):
    """自底向上删空目录, 只在 cache_dir 内部, 绝不删 checkpoint 根以上的目录。"""
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
            if "kv_load_failure_policy" not in cmd or '"recompute"' not in cmd:
                return True
            return False
    return False  # 没在跑, 无所谓


def main():
    log(f"启动, 配置={CONF_PATH} (每轮重读, 改 LIMIT_GB 即生效)")
    last_policy_warn = 0.0
    loop = 0
    while True:
        conf = read_conf()
        try:
            limit = float(conf["LIMIT_GB"]) * GB
            interval = max(30, int(conf["INTERVAL_SEC"]))
            grace = max(0, int(conf["GRACE_MIN"])) * 60
        except ValueError:
            log("WARN 配置数值非法, 本轮用默认 40GB/300s/10min")
            limit, interval, grace = 40 * GB, 300, 600
        cache_dir = os.path.realpath(conf["CACHE_DIR"])
        if not cache_dir.startswith("/home/") or cache_dir == "/home":
            log(f"ERROR CACHE_DIR 非法({cache_dir}), 本轮跳过")
            time.sleep(interval)
            continue

        try:
            if os.path.isdir(cache_dir):
                total, candidates = collect(cache_dir, grace)
                deleted = freed = 0
                if total > limit:
                    now = time.time()
                    if now - last_policy_warn > 3600 and vllm_policy_is_fail():
                        log("WARN 运行中的 vLLM 仍是 fail 策略: 删块可能让个别请求报错; "
                            "重启 vLLM 后(recompute 补丁生效)即完全安全")
                        last_policy_warn = now
                    for mtime, size, p in candidates:
                        if total <= limit:
                            break
                        # 双保险: 路径必须在 cache_dir 内
                        if not os.path.realpath(p).startswith(cache_dir + os.sep):
                            continue
                        try:
                            os.unlink(p)
                        except FileNotFoundError:
                            continue  # 已被别人删/服务自己清了
                        except OSError as e:
                            log(f"WARN 删除失败 {p}: {e}")
                            continue
                        total -= size
                        freed += size
                        deleted += 1
                    if deleted:
                        prune_empty_dirs(cache_dir)
                        log(f"清理 {deleted} 个最老块, 释放 {freed / GB:.2f} GiB, "
                            f"剩余 {total / GB:.2f} GiB (上限 {limit / GB:.0f} GiB)")
                loop += 1
                if deleted == 0 and loop % 12 == 1:  # 每小时一条心跳
                    log(f"OK 当前 {total / GB:.2f} GiB / 上限 {limit / GB:.0f} GiB, "
                        f"宽限 {grace // 60} min, 无需清理")
            else:
                if loop % 12 == 0:
                    log(f"目录不存在({cache_dir}), 等待")
                loop += 1
        except Exception as e:  # 守护脚本绝不因单轮异常退出
            log(f"ERROR 本轮异常: {e!r}")
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
