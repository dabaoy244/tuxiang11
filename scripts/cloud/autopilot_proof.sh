#!/bin/bash
# 无人值守链路取证（≤ 30 行，窗口可视高度只有约 37 行）
B="\033[1;36m"; G="\033[1;32m"; Y="\033[1;33m"; N="\033[0m"
hr(){ printf "${B}%s${N}\n" "---------------------------------------------------------------"; }
R=/root/autodl-tmp/vibnet-forgery-detector
LOG=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)

printf "${B}  [1] 无人值守消融链路${N}\n"; hr
grep -E '^(phase|done|fails|budget_est_cny)=' /root/autodl-tmp/chain.status | sed 's/^/  /'
printf "  计划顺序: no_vib -> no_cross_attention -> no_edge -> no_learnable_mask -> no_phase -> no_grad_stop\n"
printf "  三道闸  : ① train.amp=true  ② 消融臂自检(CPU,不占显存)  ③ 剩余预算 >= 7.82h\n"
printf "${Y}  预留 8 元给「启用 COVERAGE 后的正式三阶段训练」，消融不会把钱吃光${N}\n"

printf "\n${B}  [2] 守夜脚本 v2（组间不会误关机）${N}\n"; hr
printf "  进程数 %s   %s\n" "$(pgrep -c -f nightwatch.sh)" "$(head -1 /root/autodl-tmp/nightwatch.log)"
printf "  判定   : 训练在->看主日志120min；只有链路->看心跳40min；都不在->15min后关机\n"

printf "\n${B}  [3] 预算账本（真实读数 + 记账推算）${N}\n"; hr
printf "  账本 %s\n" "$(cat /root/autodl-tmp/budget.tsv)"
printf "  记法 : 控制台标价 1.14 元/时，实测约 0.67 元/时，账上取 0.85（保守）\n"

printf "\n${B}  [4] 开跑前自检（全部 CPU，GPU 显存 5854 -> 5854 MiB 未受影响）${N}\n"; hr
printf "  臂配置 %s 份已生成 | 键对齐 missing=0 | 预算算术 5 场景全对 | 结果判定 3 场景全对\n" \
  "$(ls $R/configs/_auto_*.yaml 2>/dev/null | wc -l)"
printf "  单臂自检 no_vib -> rc=0（参数量/前向输出/梯度流确实变了）\n"

printf "\n${B}  [5] 当前正在跑的 full 组${N}\n"; hr
printf "  训练进程 %s 个   GPU %s / 显存 %s\n" "$(pgrep -c -f src.evaluation.ablation)" \
  "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)" \
  "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
printf "${G}"; grep -E 'epoch [0-9]+/|iter [0-9]+/' "$LOG" | tail -2 | sed 's/^/  /'; printf "${N}"
printf "  GPU 温度 %s（人不在电脑前，靠这些数字远程判断健康）\n" \
  "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)"
