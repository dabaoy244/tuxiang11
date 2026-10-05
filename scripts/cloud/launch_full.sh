#!/bin/bash
# 重跑 full 组消融（2026-09-30，修复 train.amp 后的第一次）
# 背景：09-29 因 train.amp=false 全 fp32 激活，在 stage2 首次前向 OOM（只差 392 MiB）。
#      置 true 后显存 11080 -> 5530 MiB，为 stage2/3 留出 6.7 GB 余量。
# 用法：bash /root/autodl-tmp/launch_full.sh
set -u
R=/root/autodl-tmp/vibnet-forgery-detector
cd "$R" || { echo "[ABORT] 项目目录不存在"; exit 1; }

# ---- 前置检查：配置必须已生效（防止静默跑错口径） ----
if ! grep -qE '^[[:space:]]+amp:[[:space:]]+true' configs/default_crossval.yaml; then
  echo "[ABORT] configs/default_crossval.yaml 的 train.amp 不是 true —— 拒绝启动"
  exit 2
fi
grep -qE '^[[:space:]]+beta_anchor:[[:space:]]*"?vib' configs/default_crossval.yaml \
  || echo "[WARN] beta_anchor 不是 vib，β 退火口径可能跑偏，请人工确认"
grep -qE '^[[:space:]]+beta_epoch_offset:[[:space:]]*0' configs/default_crossval.yaml \
  || echo "[WARN] 存在非 0 的 beta_epoch_offset，会把 β 起始锚推后、跳过退火，请人工确认"
if pgrep -f 'src\.evaluation\.ablation' >/dev/null 2>&1; then
  echo "[ABORT] 已有 ablation 进程在跑，拒绝重复启动"
  exit 3
fi

TS=$(date +%Y%m%d_%H%M%S)

# ---- 归档上一轮的「追加式」日志 ----
# ablation 的 outputs/ablation/<preset>/train_log.txt 是 append 打开的，重跑不清空；
# 留着它会让看板的 [关键] 行抓到上一轮的 epoch（误导性取证）。重名前先确认没有进程占用。
TR="$R/outputs/ablation/full/train_log.txt"
if [ -f "$TR" ]; then
  mv "$TR" "$R/outputs/ablation/full/train_log.prev-${TS}.txt" \
    && echo "[info] 已归档上一轮 train_log.txt -> train_log.prev-${TS}.txt"
fi

# ---- 关键：expandable_segments 只能靠环境变量生效（仓库内无代码读它，写 yaml 无效） ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$R/outputs/ablation_logs"
LOG="$R/outputs/ablation_logs/full_s082_${TS}.log"
echo "$LOG" > /root/autodl-tmp/.current_train_log
rm -f /root/autodl-tmp/nightwatch.off

setsid nohup /root/miniconda3/bin/python -m src.evaluation.ablation \
  --config configs/default_crossval.yaml \
  --mode train --presets full --epochs-scale 0.82 \
  > "$LOG" 2>&1 < /dev/null &
PID=$!

echo "[launch] pid=$PID"
echo "[launch] log=$LOG"
echo "[launch] alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
