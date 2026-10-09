#!/bin/bash
# ==========================================================================
# T0-1：跨生成器评测采样 150 -> 300（4 权重 × {13 生成器, 严格 9 生成器}）
#
# 为什么：审计总表是 --per-class 150（n=3900 / 2700）。采样加倍到 300
#        （n=7800 / 5400）能把配对 CI 宽度收窄约 29%，让 ±1~2pp 的差异真正可判。
#
# ⚠ 口径纪律：这是**新战役**。300 口径的表**不得**与审计的 150 口径表混用/相减，
#    报数时标签必须写 (权重 md5, per-class 300, 宏平均)。
# ⚠ --exclude-generators 是**逗号分隔**（脚本按 "," split），写空格会报错。
# ==========================================================================
set -u
cd /root/autodl-tmp/vibnet-forgery-detector || exit 9
PY=/root/miniconda3/bin/python
OUT=outputs/crossgen300_20261009
mkdir -p "$OUT"
rm -f "$OUT/_DONE"

echo "===== T0-1 开始 $(date '+%F %T') ====="
echo "脚本 md5: $(md5sum scripts/eval_cross_generator.py | cut -c1-12)"
echo "协议: --per-class 300 --seed 3407 --dedup-real(默认开) --device cuda"

run_one () {
  local tag="$1"
  local ckpt="$2"
  echo ""
  echo "################ [$tag] 13 生成器 ################"
  $PY scripts/eval_cross_generator.py \
      --config configs/default.yaml --ckpt "$ckpt" \
      --per-class 300 --batch-size 32 --device cuda --seed 3407 \
      --out "$OUT/${tag}_all" --tag "${tag}_all" \
      --scores-out "$OUT/${tag}_all/scores.npz"
  echo "---- [$tag] all rc=$? ----"

  echo "################ [$tag] 严格 9 生成器 ################"
  $PY scripts/eval_cross_generator.py \
      --config configs/default.yaml --ckpt "$ckpt" \
      --per-class 300 --batch-size 32 --device cuda --seed 3407 \
      --exclude-generators progan,biggan,cyclegan,stargan \
      --out "$OUT/${tag}_strict" --tag "${tag}_strict" \
      --scores-out "$OUT/${tag}_strict/scores.npz"
  echo "---- [$tag] strict rc=$? ----"
}

# 四个权重（md5 已核）：0928 出厂件 / 主表 full / improved noaug g43 / improved aug g50
run_one "0928"  "checkpoints/vibnet_best.pt"
run_one "full"  "outputs/ablation/full/ckpt/ablation_full_best.pt"
run_one "g43"   "checkpoints/improved_noaug_best.pt"
run_one "aug50" "checkpoints/improved_best.pt"

echo ""
echo "===== T0-1 全部结束 $(date '+%F %T') ====="
touch "$OUT/_DONE"
