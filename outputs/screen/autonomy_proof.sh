#!/bin/bash
# 无人值守链路取证（<= 30 行：窗口可视约 37 行，超了会把顶部关键信息顶出画面）
R=/root/autodl-tmp/vibnet-forgery-detector
L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
echo "=== VIB-Net 无人值守链路   $(date '+%F %T') ==="
echo "--- ① 在岗进程（训练 / 链路 / 守夜 / 解锁守护）---"
for p in src.evaluation.ablation ablation_chain.sh nightwatch.sh unblock_reserve.sh; do
  n=$(ps -eo cmd | grep -F "$p" | grep -vc grep)
  printf '  %-26s x%s\n' "$p" "$n"
done
echo "--- ② 链路状态与心跳 ---"
tr '\n' ' ' < /root/autodl-tmp/chain.status 2>/dev/null; echo
tail -2 /root/autodl-tmp/chain.log 2>/dev/null
echo "--- ③ 训练进度 ---"
tail -1 "$L" 2>/dev/null | sed 's/^\[[^]]*\] *//'
echo "  日志 mtime: $(date -r "$L" '+%T' 2>/dev/null)   现在: $(date '+%T')"
echo "--- ④ GPU / 磁盘 ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
df -h /root/autodl-tmp | tail -1 | awk '{print "  数据盘已用 "$3" / "$2" ("$5")"}'
echo "--- ⑤ 预算账本 ---"
cat /root/autodl-tmp/budget.tsv
echo "--- ⑥ 数据隔离（必须只有这两项）---"
ls /root/autodl-tmp/data/Datasets/ | tr '\n' ' '; echo
echo "--- ⑦ 外部域 COVERAGE（各应 191）---"
echo "  image=$(ls $R/data/external/COVERAGE/test/image 2>/dev/null | wc -l)  mask=$(ls $R/data/external/COVERAGE/test/mask 2>/dev/null | wc -l)"
echo "--- ⑧ 解锁守护日志 ---"
tail -2 /root/autodl-tmp/unblock.log 2>/dev/null || echo "  (无)"
