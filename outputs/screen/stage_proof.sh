#!/bin/bash
# 云端取证：stage 进度 + β 退火 + 显存 + 预算（输出必须 ≤28 行，窗口可视约 37 行）
R=/root/autodl-tmp/vibnet-forgery-detector
L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
echo "==== VIB-Net 云端训练取证   $(date '+%F %T') ===="
echo "[链路] $(tr '\n' ' ' < /root/autodl-tmp/chain.status 2>/dev/null)"
echo "[进程] 训练 $(pgrep -c -f src.evaluation.ablation) | 链路 $(pgrep -c -f ablation_chain.sh) | 守夜 $(pgrep -c -f nightwatch.sh)"
echo "[阶段] $(grep -E '阶段 [0-9]/3:' "$L" 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] //')"
echo "[早停] $(grep -E '早停：' "$L" 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] *//')"
echo "[β声明] $(grep -oE '\[beta\].*' "$L" 2>/dev/null | tail -1)"
echo "[轮次] $(grep -E '\[stage[0-9]' "$L" 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] *//')"
echo "[进度] $(grep -E 'iter [0-9]+/' "$L" 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] *//')"
echo "[验证] $(grep -E '^\[.*\] +val: ' "$L" 2>/dev/null | tail -1 | grep -oE 'cls_auc=[0-9.]+|cls_acc=[0-9.]+|loc_miou_tampered_only=[0-9.]+' | tr '\n' ' ')"
echo "[日志mtime] $(date -r "$L" '+%T')  (现在 $(date '+%T'))"
echo "[GPU] $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader)"
echo "[余额] 账本 $(cat /root/autodl-tmp/budget.tsv 2>/dev/null)"
echo "[隔离] data/Datasets = $(ls /root/autodl-tmp/data/Datasets/ | tr '\n' ' ')  (必须只有 CASIAv2 ForenSynths)"
echo "[已归档臂] $(ls -1 $R/outputs/ablation/_archive/ 2>/dev/null | wc -l) 个"
