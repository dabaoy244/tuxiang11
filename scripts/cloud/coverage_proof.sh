#!/bin/bash
R=/root/autodl-tmp/vibnet-forgery-detector
echo "=== COVERAGE 外部域基准行 + 训练进度   $(date '+%F %T') ==="
echo "--- 链路状态 ---"
tr '\n' ' ' < /root/autodl-tmp/chain.status 2>/dev/null; echo
echo "--- full 组进度（主日志） ---"
L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
tail -2 "$L" 2>/dev/null
grep -E "epoch +[0-9]+/" "$L" 2>/dev/null | tail -2
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "--- 预算账本 ---"
cat /root/autodl-tmp/budget.tsv
echo "--- 外部域数据（应各 191） ---"
echo "image $(ls $R/data/external/COVERAGE/test/image | wc -l)  mask $(ls $R/data/external/COVERAGE/test/mask | wc -l)"
echo "--- 隔离检查：data/Datasets 必须只有 CASIAv2 与 ForenSynths ---"
ls /root/autodl-tmp/data/Datasets/
echo "--- COVERAGE 独立基准表 ---"
grep -E '^#|^\|' $R/outputs/coverage_ext/coverage_rows.md 2>/dev/null | head -7
