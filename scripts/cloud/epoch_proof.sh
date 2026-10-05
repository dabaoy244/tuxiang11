#!/bin/bash
# epoch 级过程记录：展示了训练的阶段推进、每轮评估指标、β 退火位置
B="\033[1;36m"; G="\033[1;32m"; Y="\033[1;33m"; N="\033[0m"
hr(){ printf "${B}%s${N}\n" "---------------------------------------------------------------"; }
LOG=$(cat /root/autodl-tmp/.current_train_log)
R=/root/autodl-tmp/vibnet-forgery-detector

printf "${B}  [A] 实验口径（决定性配置，防止跑错）${N}\n"; hr
grep -nE '^[[:space:]]+(amp|amp_dtype|epochs|batch_size):' $R/configs/default_crossval.yaml | head -4 | sed 's/^/  /'
printf "  beta_anchor=%s\n" "$(grep -E 'beta_anchor' $R/configs/default_crossval.yaml | head -1 | awk '{print $2}')"
printf "  数据集 include_generators 条数=%s (跨生成器留出口径)\n" "$(grep -cE 'include_generators|exclude_generators' $R/configs/default_crossval.yaml)"

printf "\n${B}  [B] beta 退火调度（启动即注册，非事后补）${N}\n"; hr
grep -E '\[beta\]' "$LOG" | head -2 | sed 's/^/  /'

printf "\n${B}  [C] 阶段与每轮评估（这是"过程"的逐轮档案）${N}\n"; hr
grep -nE '阶段|epoch [0-9]+/|val_cls_auc|早停|新的最佳|best' "$LOG" | tail -16 | cut -c1-130 | sed 's/^/  /'

printf "\n${B}  [D] 已落盘的权重文件${N}\n"; hr
ls -la $R/outputs/ablation/full/ckpt/ 2>/dev/null | tail -n +2 | awk '{printf "  %-42s %10s  %s %s\n", $9, $5, $6, $7}'

printf "\n${B}  [E] 当前时刻${N}\n"; hr
printf "  远端时间 %s ；训练已持续 %s\n" "$(date '+%F %T')" \
  "$(ps -o etime= -p $(pgrep -f 'src.evaluation.ablation' | head -1) | tr -d ' ')"
printf "${Y}  全部取自远端 %s${N}\n" "$LOG"
