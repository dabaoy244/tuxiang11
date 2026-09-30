#!/bin/bash
# =============================================================================
# β 生效版（outputs/ablation_betafull）的 COVERAGE 外部域出行
#
# 与 `eval_coverage_row.sh` 的唯一差别：候选权重来自**β 对照链路**的产物目录，
# tag 统一加 `_beta` 后缀，绝对不碰主消融表的行（两者协议不同，须可区分）。
# 输出写进同一个 `outputs/coverage_ext/`，这样 `report_coverage_rows.py`
# 会把「full（β≈0）」与「full_beta（β 生效）」渲染在同一张表里，便于直接对照。
#
# 幂等：某 tag 的 json 已存在就跳过。用纯 CPU（CUDA_VISIBLE_DEVICES=""），
# 不与正在跑的 GPU 训练抢显存。
#
# 用法：bash /root/autodl-tmp/eval_coverage_row_beta.sh
# =============================================================================
set -u

R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
EXT=/root/autodl-tmp/data/external
SAVEDIR=outputs/ablation_betafull
ARCH="$R/$SAVEDIR/_archive"
OUTREL=outputs/coverage_ext
LOG=/root/autodl-tmp/coverage_rows_beta.log

mkdir -p "$R/$OUTREL"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

cd "$R" || { echo "[ABORT] 项目目录不存在：$R"; exit 1; }

say "==================== β 组 COVERAGE 基准行：开始 ===================="

# ---- 0) 隔离守卫：外部域绝不能出现在训练域根下 -------------------------------
if [ -d "$R/data/Datasets/COVERAGE" ]; then
  say "[ABORT] 发现 $R/data/Datasets/COVERAGE —— 会污染各臂 val/test 口径。"
  exit 2
fi

# ---- 1) 数据守卫：必须是 191 张（100 真 + 91 篡改） ---------------------------
for d in image mask; do
  n=$(ls "$EXT/COVERAGE/test/$d" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" != "191" ]; then
    say "[ABORT] $EXT/COVERAGE/test/$d 有 $n 个文件（应为 191）。"
    exit 3
  fi
done
say "[数据] 外部域 COVERAGE/test: image 191 / mask 191 ✓"

# ---- 2) 收集候选（只取**已归档**的 β 臂，避免中间权重被当成最终行） -----------
DONE_ARMS=$("$PY" - "$ARCH" "$SAVEDIR" <<'PYEOF'
import glob, json, os, sys
arch, savedir = sys.argv[1], sys.argv[2]
ok = []
for fp in sorted(glob.glob(os.path.join(arch, "*", "ablation.json"))):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    for k, v in (d or {}).items():
        if ((v or {}).get("cls") or {}).get("acc") is not None and k not in ok:
            ok.append(k)
print(" ".join(ok))
PYEOF
)
say "[已完成] ${DONE_ARMS:-（无）}"

if [ -z "${DONE_ARMS// /}" ]; then
  say "[空转] β 组还没有已归档的臂。稍后再跑本脚本。"
  exit 0
fi

for arm in $DONE_ARMS; do
  p=$(ls -t "$R/$SAVEDIR/$arm/ckpt"/*best*.pt 2>/dev/null | head -1)
  if [ -z "$p" ]; then
    say "[警告] $arm 已归档但找不到权重（$SAVEDIR/$arm/ckpt/*best*.pt）-> 跳过"
    continue
  fi
  tag="${arm}_beta"
  js="$R/$OUTREL/external_COVERAGE_$tag.json"
  if [ -f "$js" ]; then
    say "[跳过] $tag 已产出（$(basename "$js")）"
    continue
  fi
  say "[评测] $tag <- $p"
  # CUDA_VISIBLE_DEVICES 置空：强制走 CPU，绝不与训练抢显存
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/eval_external_domain.py \
      --domain COVERAGE --ckpt "$p" --device cpu --cpu-threads 8 \
      --expect 191 --out "$OUTREL" --tag "$tag" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    say "[完成] $tag rc=0"
  else
    say "[失败] $tag rc=$rc（详见 $LOG 末尾）"
  fi
done

# ---- 3) 刷新合并表 -----------------------------------------------------------
say "[汇总] 渲染 $OUTREL/coverage_rows.md"
"$PY" scripts/report_coverage_rows.py --out "$OUTREL" >> "$LOG" 2>&1 \
  && say "[汇总] 完成" || say "[汇总] 失败 rc=$?"

say "==================== β 组 COVERAGE 基准行：结束 ===================="
