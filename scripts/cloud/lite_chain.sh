#!/bin/bash
# =============================================================================
# lite_chain.sh —— 等 improved 对链跑完，接力跑 lite 档（两阶段）
#
# 为什么要补这一环
# ----------------
# `configs/lite.yaml` / `improved_lite*.yaml` / `gate_lite_trainability.yaml`
# 此前**都漏了 `backbone_weights`** ⇒ s16 骨干是**随机初始化**。
# 所以 2026-10-02 那次 lite 闸门的结论
#   （cls_acc=0.5176 / cls_auc=0.4950 / loc_miou=0.0000）
# 测的其实是"随机骨干能不能训起来"，必然是"不能"。
# 现已补上 `pretrained/vit_s16_imagenet.pt`
#   （timm vit_small_patch16_224.augreg_in21k_ft_in1k，150 张量，
#     与随机骨干余弦 0.0347；`check_backbone.py` 结论 FAIL -> WARN）
#
# 两阶段
# ------
#   ② lite 可训练性闸门（stage1-only，约 2h）—— 便宜，先确认"能训起来"
#   ③ 闸门 val_cls_auc > 0.60 才继续跑完整 improved_lite（约 5h），否则停手省钱
#      （随机基线约 0.52，0.60 是"明显在学"的保守门限）
#
# ⚠ 与守夜的配合（务必看）
# ----------------------
# nightwatch v3.2 的 `PAT_CHAIN` 原本只认 `(ablation_chain|beta_control_chain)\.sh`，
# **不认 `improved_pair_chain.sh`** —— 现在机器没被关机，只是因为 `scripts/train.py`
# 在跑（走分支①）。一旦 improved_noaug 结束，两样都不匹配 ⇒ 连续 3 次（15 min）
# 判"训练与链路均已结束" ⇒ 关机，把刚接上的 lite 直接砍掉。
# 故本次同时把 `PAT_CHAIN` 改为 `(ablation_chain|beta_control_chain|lite_chain)\.sh`，
# 并重启守夜。本脚本在**等待期**每 2 分钟 `touch $HB`（chain.heartbeat），
# 满足分支②「链路卡死 40 min」的新鲜度判据。
#
# 取消：touch /root/autodl-tmp/lite_chain.stop   （等待循环里检查，不关机、不干预在跑的训练）
# 日志：/root/autodl-tmp/lite_chain.log
# =============================================================================
set -u
R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/lite_chain.log
HB=/root/autodl-tmp/chain.heartbeat
CURLOG=/root/autodl-tmp/.current_train_log
STOP=/root/autodl-tmp/lite_chain.stop
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }

say "lite 链启动 PID=$$ ：等待 improved 对链（improved + improved_noaug）结束"

while pgrep -f "improved_pair_chain\.sh" >/dev/null 2>&1 \
   || pgrep -f "configs/improved\.yaml" >/dev/null 2>&1 \
   || pgrep -f "configs/improved_noaug\.yaml" >/dev/null 2>&1; do
  if [ -e "$STOP" ]; then say "检测到 $STOP -> 主动退出（不关机）"; exit 0; fi
  touch "$HB"
  sleep 120
done
if [ -e "$STOP" ]; then say "检测到 $STOP -> 主动退出（不关机）"; exit 0; fi

say "对链已结束 -> 进入阶段②：lite 可训练性闸门（预训练 s16 骨干）"
cd "$R" || { say "[X] 项目目录不存在 -> 退出"; exit 1; }

# 装车前先确认骨干真的挂上了预训练权重（不许再"静默随机初始化"）
"$PY" scripts/check_backbone.py --config configs/gate_lite_trainability.yaml 2>&1 \
  | grep -E "backbone_weights|已加载预训练骨干|随机初始化|结论" >> "$LOG" 2>&1 || true

GLOG=/root/autodl-tmp/lite_gate_pretrained.log
printf '%s\n' "$GLOG" > "$CURLOG"
(setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/train.py \
    --config configs/gate_lite_trainability.yaml --device cuda --amp --workers 8 \
    --data-root /root/autodl-tmp/data/Datasets \
    --tag gate_lite_pretrained --cudnn on > "$GLOG" 2>&1 < /dev/null &)
sleep 90

if ! pgrep -f "gate_lite_trainability\.yaml" >/dev/null 2>&1; then
  say "[!] 闸门进程未起来 —— 末尾 20 行："
  tail -20 "$GLOG" >> "$LOG" 2>/dev/null
  exit 1
fi
say "闸门已在跑（$GLOG）"
grep -E "兜底骨干|已启用|冻结前|total\(M\)|split=train" "$GLOG" 2>/dev/null | head -8 >> "$LOG"

while pgrep -f "gate_lite_trainability\.yaml" >/dev/null 2>&1; do
  if [ -e "$STOP" ]; then say "检测到 $STOP -> 主动退出（闸门继续跑，不干预）"; exit 0; fi
  touch "$HB"
  sleep 120
done

AUC=$(grep -oE "最佳 val_cls_auc=[0-9.]+" "$GLOG" | tail -1 | sed 's/.*=//')
say "闸门结束。最佳 val_cls_auc=${AUC:-未解析到}"
grep -E "完成。最佳" "$GLOG" 2>/dev/null | tail -1 >> "$LOG"

if [ -z "${AUC:-}" ]; then
  say "闸门结果解析失败 -> 不启动阶段③（人工看 $GLOG）"; exit 0
fi
if awk -v a="$AUC" 'BEGIN{exit !(a+0 > 0.60)}'; then
  say "闸门通过（val_cls_auc=$AUC > 0.60）-> 进入阶段③：完整 improved_lite"
else
  say "闸门未过（val_cls_auc=$AUC <= 0.60；随机基线约 0.52）-> 不启动阶段③，到此为止"
  exit 0
fi

if [ -e "$STOP" ]; then say "检测到 $STOP -> 主动退出（不关机）"; exit 0; fi
FLOG=/root/autodl-tmp/improved_lite.log
printf '%s\n' "$FLOG" > "$CURLOG"
mkdir -p outputs/run_lite_improved
(setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/train.py \
    --config configs/improved_lite.yaml --device cuda --amp --workers 8 \
    --data-root /root/autodl-tmp/data/Datasets --out-dir outputs/run_lite_improved \
    --tag improved_lite --cudnn on > "$FLOG" 2>&1 < /dev/null &)
sleep 90

if pgrep -f "improved_lite\.yaml" >/dev/null 2>&1; then
  say "阶段③ improved_lite 已确认在跑（$FLOG）"
  grep -E "兜底骨干|已启用|冻结前|total\(M\)|split=train" "$FLOG" 2>/dev/null | head -8 >> "$LOG"
else
  say "[!] improved_lite 未起来 —— 末尾 20 行："
  tail -20 "$FLOG" >> "$LOG" 2>/dev/null
fi
say "lite 链收尾：训练交给它自己（守夜会在训练结束后关机，数据盘保留）"
