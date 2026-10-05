#!/bin/bash
B="\033[1;36m"; G="\033[1;32m"; Y="\033[1;33m"; N="\033[0m"
hr(){ printf "${B}%s${N}\n" "---------------------------------------------------------------"; }
LOG=$(cat /root/autodl-tmp/.current_train_log)

printf "${B}  [1] 远端身份 —— 这条 SSH 会话落在哪台机器上${N}\n"; hr
printf "  主机名 / 用户 : %s   (whoami=%s)\n" "$(hostname)" "$(whoami)"
printf "  远端系统时间  : %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf "  系统 / 容器IP : %s | %s\n" "$(grep PRETTY /etc/os-release | cut -d= -f2 | tr -d '\"')" "$(hostname -I | awk '{print $1}')"
printf "  项目根目录    : /root/autodl-tmp/vibnet-forgery-detector\n"

printf "\n${B}  [2] GPU 硬件 —— 本机没有这块卡${N}\n"; hr
printf "  %s\n" "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
printf "  实时状态      : 利用率 %s | 显存 %s | 温度 %s\n" \
  "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)" \
  "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" \
  "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)"

printf "\n${B}  [3] 正在跑的模型任务${N}\n"; hr
printf "  训练进程数 %s 个 (1 主 + 10 worker) ; 守夜守护 %s 个\n" \
  "$(pgrep -f 'src.evaluation.ablation' | wc -l)" "$(pgrep -f nightwatch.sh | wc -l)"
printf "  主进程 : %s\n" "$(pgrep -af 'src.evaluation.ablation' | head -1 | cut -c1-120)"

printf "\n${B}  [4] 训练进度 —— 直接从远端日志读${N}\n"; hr
printf "  日志 %s\n" "$LOG"
printf "  最后写入 %s\n" "$(stat -c '%y' "$LOG" | cut -c1-19)"
printf "${G}"; grep -E 'iter [0-9]+/|epoch [0-9]+/' "$LOG" | tail -3 | sed 's/^/  /'; printf "${N}"

printf "\n${B}  [5] 异常扫描（OOM / Traceback / NaN）${N}\n"; hr
A=$(grep -cE 'OutOfMemoryError|CUDA out of memory' "$LOG"); C=$(grep -cE 'Traceback' "$LOG"); D=$(grep -cE 'loss=nan|loss=NaN' "$LOG")
printf "  OutOfMemory=%s   Traceback=%s   NaN_loss=%s   " "$A" "$C" "$D"
if [ "$A" = 0 ] && [ "$C" = 0 ] && [ "$D" = 0 ]; then printf "${G}=> 训练健康${N}\n"; else printf "${R}=> 有异常！${N}\n"; fi
printf "${Y}  以上全部取自远端主机，非本机模拟。${N}\n"
