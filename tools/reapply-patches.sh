#!/usr/bin/env bash
# reapply-patches.sh — 把 patches/ 里的本机补丁幂等地重放到 vLLM fork 上。
#
# 解决 TODO.md 第 3 条: 上游 update.sh 覆盖 launcher.sh 后, 补丁 1/2 (及其它) 全部回退。
# 每个补丁单独处理:
#   - 已经打上  (git apply -R --check 通过)  -> 跳过
#   - 还没打上  (git apply --check 通过)      -> 打上, 并立刻用 -R --check 复验
#   - 两者都不通过 (上游改了同一处, 冲突)     -> 标记失败, 不动该文件, 继续检查其余补丁
#
# 用法:
#   tools/reapply-patches.sh [FORK_DIR]            # 打补丁
#   tools/reapply-patches.sh --check [FORK_DIR]    # 只报告状态, 不改任何文件
#
# FORK_DIR 缺省取环境变量 VLLM_FORK_DIR, 再缺省取当前目录。PATCH_DIR 缺省为本脚本上级的 patches/。
#
# 退出码: 0 = 全部补丁就位; 2 = 有补丁冲突/失败 (需人工); 3 = --check 模式下有补丁缺失。
# 失败时若设置了 NOTIFY_CMD (如 NOTIFY_CMD='notify-send reapply'), 会以报告文本为参数调用它;
# 同时用 logger 写进 journal (若有)。
#
# 接入 systemd (启动前自检, 补丁缺失/冲突就拒绝启动):
#   在 systemd/vllm-dflash2.service 的 [Service] 里加一行
#     ExecStartPre=/path/to/tools/reapply-patches.sh /path/to/fork
#   重放补丁会修改 fork 工作树, 请确认该用户对 fork 目录有写权限。
set -u

MODE=apply
FORK_DIR="${VLLM_FORK_DIR:-}"
for arg in "$@"; do
  case "$arg" in
    --check) MODE=check ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) FORK_DIR="$arg" ;;
  esac
done
FORK_DIR="${FORK_DIR:-$PWD}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-$SCRIPT_DIR/../patches}"

die() { echo "reapply-patches: $*" >&2; exit 2; }

git -C "$FORK_DIR" rev-parse --show-toplevel >/dev/null 2>&1 || die "$FORK_DIR 不是 git 仓库"
FORK_DIR="$(git -C "$FORK_DIR" rev-parse --show-toplevel)"
[ -d "$PATCH_DIR" ] || die "找不到补丁目录 $PATCH_DIR"
shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
[ "${#PATCHES[@]}" -gt 0 ] || die "$PATCH_DIR 里没有 .patch 文件"

echo "fork: $FORK_DIR  (HEAD $(git -C "$FORK_DIR" rev-parse --short HEAD), 补丁基线 93c9013)"
echo "模式: $MODE"

ok=0; applied=0; missing=0; failed=0
report=""
for p in "${PATCHES[@]}"; do
  name="$(basename "$p")"
  if git -C "$FORK_DIR" apply -R --check "$p" >/dev/null 2>&1; then
    line="  [已就位] $name"; ok=$((ok+1))
  elif git -C "$FORK_DIR" apply --check "$p" >/dev/null 2>&1; then
    if [ "$MODE" = check ]; then
      line="  [缺失  ] $name  (可干净打上)"; missing=$((missing+1))
    elif git -C "$FORK_DIR" apply "$p" >/dev/null 2>&1 \
         && git -C "$FORK_DIR" apply -R --check "$p" >/dev/null 2>&1; then
      line="  [已重打] $name"; applied=$((applied+1))
    else
      line="  [失败  ] $name  (apply 后复验未通过)"; failed=$((failed+1))
    fi
  else
    line="  [冲突  ] $name  (既未打上也无法干净打上: 上游改了同一处, 需手工合并)"
    failed=$((failed+1))
  fi
  echo "$line"
  report+="$line"$'\n'
done

summary="就位 $ok / 重打 $applied / 缺失 $missing / 失败或冲突 $failed (共 ${#PATCHES[@]})"
echo "$summary"

if [ "$failed" -gt 0 ]; then
  msg="reapply-patches 失败: $summary"$'\n'"$report"
  echo "$msg" >&2
  command -v logger >/dev/null 2>&1 && logger -t reapply-patches -p user.err "$msg"
  [ -n "${NOTIFY_CMD:-}" ] && $NOTIFY_CMD "$msg" >/dev/null 2>&1
  exit 2
fi
if [ "$missing" -gt 0 ]; then
  exit 3
fi
exit 0
