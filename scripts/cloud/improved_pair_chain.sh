#!/bin/bash
# =============================================================================
# 改进版两臂接力：等 improved(增强开) 结束 -> 自动起 improved_noaug(增强关，对照臂)
#
# 为什么需要它
#   ① 两臂必须**串行** —— 12GB 卡装不下两个训练（单臂 stage2/3 就要 8.5GB）；
#   ② 第二臂要能无人值守接上，不能靠人守着第一臂结束的那一刻。
#
# ⚠ 两臂配置的 project.output_dir 都是**基线名 `outputs/run_crossval`**
#   （派生时只改了 augment / max_samples / stage2 轮数，没改 output_dir）
#   ⇒ 第二臂**必须显式 `--out-dir`**，否则会覆盖第一臂的 train_log/history。
#
# 单变量保证：两臂唯一差异 = `data.augment`（其余逐字段相同，见 deliverables/08 §1 A3）。
#
# 取消：touch /root/autodl-tmp/pair_chain.stop   （等第一臂结束的循环里会检查）
# 日志：/root/autodl-tmp/improved_pair_chain.log
# =============================================================================
set -u
R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
CHAIN_LOG=/root/autodl-tmp/improved_pair_chain.log
STOP=/root/autodl-tmp/pair_chain.stop
say(){ echo "[$(date '+%F %T')] $*" >> "$CHAIN_LOG"; }

say "接力链路启动 PID=$$ ：等待 improved(增强开) 结束"

# 注意 PAT 用 `configs/improved\.yaml` —— 它**不会**误匹配 `configs/improved_noaug.yaml`
# （后者不含子串 "improved.yaml"），所以第二臂起来后本循环不会自己续等。
while pgrep -f "configs/improved\.yaml" >/dev/null 2>&1; do
  [ -e "$STOP" ] && { say "检测到 $STOP -> 主动退出（不关机）"; exit 0; }
  sleep 120
done
[ -e "$STOP" ] && { say "检测到 $STOP -> 主动退出（不关机）"; exit 0; }

say "improved 已结束。第一臂结论："
grep -E "完成。最佳|早停" /root/autodl-tmp/improved_aug.log 2>/dev/null | tail -3 >> "$CHAIN_LOG"
cp -a "$R/outputs/run_crossval/train_log.txt" /root/autodl-tmp/improved_train_log.txt 2>/dev/null

say "启动 improved_noaug（对照臂，增强关）"
cd "$R" || { say "[X] 项目目录不存在 -> 退出"; exit 1; }
LOGF=/root/autodl-tmp/improved_noaug.log
printf '%s\n' "$LOGF" > /root/autodl-tmp/.current_train_log
mkdir -p outputs/run_improved_noaug
(setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/train.py \
    --config configs/improved_noaug.yaml --device cuda --amp --workers 8 \
    --data-root /root/autodl-tmp/data/Datasets --out-dir outputs/run_improved_noaug \
    --tag improved_noaug --cudnn on > "$LOGF" 2>&1 < /dev/null &)
sleep 90

if pgrep -f "improved_noaug\.yaml" >/dev/null 2>&1; then
  say "improved_noaug 已确认在跑"
else
  say "[!] improved_noaug 未起来 —— 看 $LOGF"
fi
grep -E "训练集|total\(M\)|已启用|兜底骨干|冻结前|数据|split=train" "$LOGF" 2>/dev/null | head -8 >> "$CHAIN_LOG"
say "接力完成：第二臂已在跑，收尾交给它自己（守夜会在两臂都结束后关机）"
