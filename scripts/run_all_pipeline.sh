#!/usr/bin/env bash
# ============================================================================
#  run_all_pipeline.sh —— 正式训练 + 评测 + 打包 + 镜像 + 关机 的「一条龙」链路
#
#  为什么需要它：cloud_autodl.sh 的 train 子命令只做「训练 + 写 RUN_REPORT +
#  自动关机」，**不含评测**。于是训练一结束实例就自动关机，而 deliverables/04 的
#  §6 评测（ACC / 定位 mIoU）还要再开机才能跑 —— 按量计费「关机后不保证还有卡」，
#  这一步很容易卡住。本脚本把四件事串成一条串行链路，一次开机跑完。
#
#  用法（云端 · 有卡模式）：
#      cd /root/autodl-tmp/vibnet-forgery-detector
#      ln -sfn /root/autodl-tmp/data data        # ★ 必须，见下面「前置」
#      setsid nohup env AUTOSHUTDOWN=1 bash scripts/run_all_pipeline.sh \
#          > outputs/cloud_runall_launch.log 2>&1 < /dev/null &
#      tail -f outputs/gpu_train_log.txt
#
#  前置（★ 不能省）：scripts/train.py 有 --data-root，但 src/evaluation/
#  evaluate.py **没有**这个参数，只认配置里的相对路径 data/Datasets。云端数据在
#  /root/autodl-tmp/data/Datasets，项目目录下没有 data 这一层时，评测会把每个
#  数据集都判成「一个样本都没有」而**静默跳过** —— 不报错、照写 eval_test.json，
#  拿到的却是空壳指标（见 docs/08 D44）。
#
#  设计原则：
#    · 评测失败不阻断打包与关机（权重是主产物，评测可复跑）
#    · 训练失败则保留实例供排查（与 cloud_autodl.sh 的既有约定一致）
#    · 产物同时留在数据盘与 /root/autodl-fs（跨实例中转盘，免费 200G）
#
#  产物：
#    outputs/gpu_train_log.txt            训练日志
#    outputs/gpu_eval_log.txt             评测日志
#    outputs/eval_full/eval_test.json     评测 A（混合 test：分类 + 推理性能）
#    outputs/eval_casia/eval_test.json    评测 B（--dataset CASIAv2：定位 mIoU）
#    outputs/RUN_REPORT.txt               运行报告（含训练尾部 + 两份评测 JSON）
#    outputs/ALL_DONE.flag                完成标志（仅训练成功时写）
#    /root/autodl-tmp/vibnet_results_<ts>.tar.gz   打包产物
#    /root/autodl-fs/vibnet_out/          跨实例镜像
# ============================================================================
set -u

PROJ=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
DATASETS=/root/autodl-tmp/data/Datasets
cd "$PROJ" || { echo "PROJ not found: $PROJ"; exit 1; }

LOG=outputs/gpu_train_log.txt
ELOG=outputs/gpu_eval_log.txt
RUN=outputs/RUN_REPORT.txt
EF=outputs/eval_full
EC=outputs/eval_casia
rm -f outputs/ALL_DONE.flag
# 清掉可能残留的评测冒烟产物，避免被 pack 一起打包带走
rm -rf outputs/eval_sanity outputs/eval_sanity_casia

# ---- 归档上一次运行的日志（本脚本**可重复启动**）--------------------------------
# 为什么必须做：下面所有日志都是 `>>` 追加写的。若不归档，第二次运行时
# gpu_train_log.txt 会接在**上一次被中断**的日志后面 —— 一份日志里混着两次
# 不同长度的运行，epoch/iter 数会跳变，非常容易误读成"训练卡住了"或
# "为什么 iter 归零又变大"。归档后每次运行都有一份独立、自洽的日志。
_TS=$(date +%Y%m%d_%H%M)
mkdir -p outputs/_archive
for _f in "$LOG" "$ELOG" "$RUN"; do
  [ -s "$_f" ] && mv -f "$_f" "outputs/_archive/$(basename "$_f" .txt).$_TS.txt"
done

{
  echo "[$(date '+%F %T')] ===== 正式三阶段训练开始（从 stage1 全量重跑，口径见 VERSION.txt）====="
  echo "[$(date '+%F %T')] VERSION.txt md5: $(md5sum VERSION.txt | cut -d' ' -f1)"
} >> "$LOG"

t0=$(date +%s)
$PY -u scripts/train.py --config configs/default.yaml --device cuda --amp \
    --workers 8 --data-root "$DATASETS" --tag vibnet --cudnn on >> "$LOG" 2>&1
rc_train=$?
t1=$(date +%s)
el=$(( t1 - t0 ))
echo "[$(date '+%F %T')] train 退出码=$rc_train  用时 $((el/60))分$((el%60))秒" >> "$LOG"

rc_ef=1; rc_ec=1
if [ "$rc_train" = "0" ]; then
  echo "[$(date '+%F %T')] ===== 评测 A：混合 test 集（分类 + 推理性能）=====" >> "$ELOG"
  $PY -u -m src.evaluation.evaluate --config configs/default.yaml \
      --ckpt checkpoints/vibnet_best.pt --split test --save-dir "$EF" >> "$ELOG" 2>&1
  rc_ef=$?

  echo "[$(date '+%F %T')] ===== 评测 B：CASIAv2 单数据集（定位 mIoU 判定口径）=====" >> "$ELOG"
  $PY -u -m src.evaluation.evaluate --config configs/default.yaml \
      --ckpt checkpoints/vibnet_best.pt --split test --dataset CASIAv2 --save-dir "$EC" >> "$ELOG" 2>&1
  rc_ec=$?
fi

# ------------------------------ 运行报告 ------------------------------
{
  echo "训练运行报告  (由 run_all_pipeline.sh 生成)"
  echo "  生成时间      : $(date '+%F %T')"
  echo "  train 退出码  : $rc_train   用时 $((el/60))分$((el%60))秒"
  echo "  eval A(混合)  : rc=$rc_ef  -> $EF/eval_test.json"
  echo "  eval B(定位)  : rc=$rc_ec  -> $EC/eval_test.json"
  echo "  训练日志      : outputs/gpu_train_log.txt"
  echo "  评测日志      : outputs/gpu_eval_log.txt"
  echo "  代码版本      : $(head -3 VERSION.txt | tr '\n' ' ')"
  echo
  echo "------ 训练日志末尾 40 行 ------"
  tail -n 40 "$LOG" 2>/dev/null
  echo
  echo "------ 评测 A 报告 (eval_full) ------"
  cat "$EF/eval_test.json" 2>/dev/null || echo "(缺失)"
  echo
  echo "------ 评测 B 报告 (eval_casia) ------"
  cat "$EC/eval_test.json" 2>/dev/null || echo "(缺失)"
} > "$RUN" 2>&1

# ------------------------------ 打包 + 镜像 ------------------------------
echo "" >> "$RUN"
echo "------ pack ------ " >> "$RUN"
bash scripts/cloud_autodl.sh pack >> "$RUN" 2>&1

TAR=$(ls -t /root/autodl-tmp/vibnet_results_*.tar.gz 2>/dev/null | head -1)
if [ -n "${TAR:-}" ]; then
  mkdir -p /root/autodl-fs/vibnet_out
  cp -f "$TAR" /root/autodl-fs/vibnet_out/ 2>/dev/null && \
      echo "已镜像权重包: /root/autodl-fs/vibnet_out/$(basename "$TAR")" >> "$RUN"
  # 小文件（报告/指标/日志）再单独镜像一份，便于快速取用
  cp -f "$RUN" /root/autodl-fs/vibnet_out/ 2>/dev/null
  cp -f "$ELOG" /root/autodl-fs/vibnet_out/ 2>/dev/null
  cp -f "$EF/eval_test.json" /root/autodl-fs/vibnet_out/eval_full.json 2>/dev/null
  cp -f "$EC/eval_test.json" /root/autodl-fs/vibnet_out/eval_casia.json 2>/dev/null
  ls -la /root/autodl-fs/vibnet_out/ >> "$RUN" 2>&1
fi

if [ "$rc_train" = "0" ]; then
  echo "train=$rc_train evalA=$rc_ef evalB=$rc_ec $(date '+%F %T')" > outputs/ALL_DONE.flag
fi

# ------------------------------ 关机 ------------------------------
if [ "${AUTOSHUTDOWN:-0}" = "1" ]; then
  if [ "$rc_train" != "0" ]; then
    {
      echo "[$(date '+%F %T')] train 非正常退出（$rc_train）—— 不自动关机，保留实例排查"
      echo "  先看: tail -40 outputs/gpu_train_log.txt"
      echo "  手动关机: /usr/bin/shutdown"
    } >> "$RUN"
    exit 0
  fi
  sync
  # 关机前的「回捞窗口」。
  # 为什么要它：链路跑完若立刻关机，本地的结果回捞任务就再也连不上实例，
  # 只能靠人手动开一台无卡机把 autodl-fs 里的副本拿回来 —— 而按量计费实例
  # 关机后不保证还能开起来（CPU/卡被占走），这一步很容易变成阻塞。
  # 给一个可配置的延迟，让定时回捞任务有机会在关机前把结果拉回本地；
  # 兜底：回捞任务没跑，最多也就是多保留这段时间，不会一直烧卡。
  #   SHUTDOWN_GRACE_MIN=0（默认）→ 立即关机，保持原行为不变
  #   SHUTDOWN_GRACE_MIN=180       → 跑完后保留 3 小时；回捞任务若已执行会主动提前关机
  _GRACE="${SHUTDOWN_GRACE_MIN:-0}"
  if [ "$_GRACE" -gt 0 ] 2>/dev/null; then
    {
      echo "[$(date '+%F %T')] 延迟 ${_GRACE} 分钟后自动关机（给结果回捞留窗口）"
      echo "                 回捞任务若已跑过，它会主动执行 /usr/bin/shutdown 提前结束"
      echo "                 想立刻关机：/usr/bin/shutdown"
    } >> "$RUN"
    setsid nohup bash -c "sleep $((_GRACE * 60)); /usr/bin/shutdown" >/dev/null 2>&1 </dev/null &
  else
    echo "[$(date '+%F %T')] 自动关机（按量计费：关机后不保留这张卡）" >> "$RUN"
    /usr/bin/shutdown
  fi
fi
