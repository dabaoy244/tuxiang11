#!/bin/bash
# 故障恢复取证（<= 30 行）
R=/root/autodl-tmp/vibnet-forgery-detector
L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
echo "=== VIB-Net 故障恢复后现场   $(date '+%F %T') ==="
echo "--- ① 在岗进程 ---"
for p in src.evaluation.ablation ablation_chain.sh nightwatch.sh; do
  printf '  %-26s x%s\n' "$p" "$(ps -eo cmd | grep -F "$p" | grep -vc grep)"
done
echo "--- ② 链路阶段与最近心跳 ---"
tr '\n' ' ' < /root/autodl-tmp/chain.status 2>/dev/null; echo
tail -4 /root/autodl-tmp/chain.log 2>/dev/null | sed 's/^/  /'
echo "--- ③ 跨生成器评测进度（full，之前从未跑成） ---"
ls -la $R/outputs/cross_gen/full/ 2>/dev/null | tail -4 | sed 's/^/  /' || echo "  (尚未产出文件)"
echo "--- ④ GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "--- ⑤ 预算（已按真实读数修正） ---"
echo "  budget.tsv: $(cat /root/autodl-tmp/budget.tsv)"
echo "--- ⑥ stderr 日志（新增，用于将来排查静默退出） ---"
if [ -s /root/autodl-tmp/chain.stderr.log ]; then tail -3 /root/autodl-tmp/chain.stderr.log | sed 's/^/  /'; else echo "  (空 = 无异常)"; fi
echo "--- ⑦ 数据隔离 ---"
echo "  $(ls /root/autodl-tmp/data/Datasets/ | tr '\n' ' ')"
echo "--- ⑧ 已归档臂 ---"
ls -1 $R/outputs/ablation/_archive/ 2>/dev/null | sed 's/^/  /'
