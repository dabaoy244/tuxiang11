#!/bin/bash
# 无人值守链路取证（<= 30 行：窗口可视约 37 行，超了会把顶部关键信息顶出画面）
# 2026-10-01 17:05 修订：主链路收尾后主体改为「主链路 + β 对照链路」双视角；
# 移除已退役的 unblock_reserve.sh（RESERVE=0，日志末行"监视结束"是历史残留，易被误读）。
R=/root/autodl-tmp/vibnet-forgery-detector
L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
BL=$(ls -t $R/outputs/ablation_logs/beta_*_s082_*.log 2>/dev/null | head -1)
echo "=== VIB-Net 无人值守链路   $(date '+%F %T') ==="
echo "--- ① 在岗进程（训练 / 链路 / 守夜）---"
for p in src.evaluation.ablation src.evaluation.beta_control ablation_chain.sh beta_control_chain.sh nightwatch.sh; do
  n=$(ps -eo cmd | grep -F "$p" | grep -vc grep)
  printf '  %-30s x%s\n' "$p" "$n"
done
echo "--- ② 主消融链路 ---"
tr '\n' ' ' < /root/autodl-tmp/chain.status 2>/dev/null; echo
tail -1 /root/autodl-tmp/chain.log 2>/dev/null
echo "--- ②-beta β 对照链路（产物与主表物理隔离）---"
tr '\n' ' ' < /root/autodl-tmp/beta_control.status 2>/dev/null; echo
echo "--- ③ 训练进度 ---"
tail -1 "$L" 2>/dev/null | sed 's/^\[[^]]*\] *//'
echo "  主日志 mtime: $(date -r "$L" '+%T' 2>/dev/null)   现在: $(date '+%T')"
echo "  β  日志 mtime: $(date -r "$BL" '+%T' 2>/dev/null)  $(tail -1 "$BL" 2>/dev/null | sed 's/^\[[^]]*\] *//')"
echo "--- ④ GPU / 磁盘 ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
df -h /root/autodl-tmp | tail -1 | awk '{print "  数据盘已用 "$3" / "$2" ("$5")"}'
echo "--- ⑤ 预算账本 ---"
cat /root/autodl-tmp/budget.tsv
echo "--- ⑥ 数据隔离（必须只有这两项）---"
ls /root/autodl-tmp/data/Datasets/ | tr '\n' ' '; echo
echo "--- ⑦ 外部域 COVERAGE（各应 191）---"
echo "  image=$(ls $R/data/external/COVERAGE/test/image 2>/dev/null | wc -l)  mask=$(ls $R/data/external/COVERAGE/test/mask 2>/dev/null | wc -l)"
