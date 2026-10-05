#!/bin/bash
# 把暂存的 COVERAGE 正式启用（移入 data/Datasets/）。
#
# 为什么需要"暂存"这一步（★ 可比性陷阱，别绕过）：
#   消融的 4 个组必须跑在**同一套评测集**上，否则结果不可并列。
#   `full` 组 2026-09-30 00:48 起跑时 data/Datasets 里**没有** COVERAGE，
#   它的 val/test 不含 COVERAGE；若在其余 3 组之前把 COVERAGE 放进原位，
#   这 3 组的 val/test 就会多出 19/20 张 COVERAGE，
#   于是 4 组的 val_cls_auc（best.pt 挑选依据）与 test 指标口径不一致，不可比。
#   ⇒ 消融期间保持"无 COVERAGE"；等 4 组全部跑完（或不再做组间对比）后，用本脚本启用。
#
# 用法：bash /root/autodl-tmp/activate_coverage.sh [--force]
set -u
SRC=/root/autodl-tmp/_pending/COVERAGE
DST=/root/autodl-tmp/data/Datasets/COVERAGE
FORCE="${1:-}"

[ -d "$SRC" ] || { echo "[ABORT] 暂存目录不存在：$SRC"; exit 1; }
if [ -e "$DST" ]; then
  echo "[SKIP] $DST 已存在，无需启用。"
  ls "$DST"; exit 0
fi

if pgrep -f 'src\.evaluation\.ablation' >/dev/null 2>&1 && [ "$FORCE" != "--force" ]; then
  echo "[ABORT] 仍有消融进程在跑。此刻启用会让尚未开跑的组与已跑的组口径不一致。"
  echo "        确认要强制启用请加 --force。"
  exit 3
fi

echo "校验暂存内容 ..."
for s in train val test; do
  n=$(ls "$SRC/$s/image" 2>/dev/null | wc -l)
  m=$(ls "$SRC/$s/mask"  2>/dev/null | wc -l)
  printf "  %-6s image=%-4s mask=%-4s\n" "$s" "$n" "$m"
  [ "$n" -gt 0 ] && [ "$n" -eq "$m" ] || { echo "[ABORT] $s 的 image/mask 数量不一致或为空"; exit 2; }
done

mv "$SRC" "$DST" || { echo "[ABORT] 移动失败"; exit 4; }
echo "已启用 -> $DST"
ls /root/autodl-tmp/data/Datasets/
echo
echo "提示：COVERAGE/val=19、test=20 张，远小于 configs 里声明的 100 —— "
echo "      因为官方 COVERAGE 只有 100 对；其中 9 对掩码与图像尺寸不一致已剔除。"
