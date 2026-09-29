#!/bin/bash
# =============================================================================
# 外部域 COVERAGE「独立基准行」批量产出（云端，幂等，可反复调用）
#
# 做什么
#   给每个**已完成**的本项目权重，在 COVERAGE 外部域上各评一行（零样本、未训练过），
#   最后刷新合并表 outputs/coverage_ext/coverage_rows.md。
#
# 为什么必须与训练域物理隔离
#   COVERAGE 放在 `/root/autodl-tmp/data/external/COVERAGE/`，**绝不在**
#   `data/Datasets/` 下。`build_dataloaders` 会按配置里的 root 去找每个数据集 ——
#   一旦 COVERAGE 落进 `data/Datasets/`，正在排队的消融臂就会被多喂一个数据集，
#   各臂 val/test 口径不再一致，整张消融表作废。本脚本每轮先检查这一点。
#
# 为什么用 CPU
#   与正在跑的 GPU 训练共存：CPU 推理不吃显存（不会被 OOM 牵连），
#   再用 --cpu-threads 8 限流，别把 80 核吃满拖慢训练。
#
# 幂等
#   某个 tag 的 json 已存在就跳过。所以训练每完成一个臂，重跑本脚本即可，
#   不会重复烧时间。
#
# 用法：bash /root/autodl-tmp/eval_coverage_row.sh
# =============================================================================
set -u

R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
EXT=/root/autodl-tmp/data/external
OUTREL=outputs/coverage_ext
LOG=/root/autodl-tmp/coverage_rows.log

mkdir -p "$R/$OUTREL"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

cd "$R" || { echo "[ABORT] 项目目录不存在：$R"; exit 1; }

say "==================== 外部域 COVERAGE 基准行：开始 ===================="

# ---- 0) 隔离守卫：外部域绝不能出现在训练域根下 -------------------------------
if [ -d "$R/data/Datasets/COVERAGE" ]; then
  say "[ABORT] 发现 $R/data/Datasets/COVERAGE —— 这会被消融链路按 root 扫到、"
  say "        污染各臂 val/test 口径。请先把它移出 data/Datasets/ 再跑本脚本。"
  exit 2
fi

# ---- 1) 数据守卫：必须是 191 张（100 真 + 91 篡改） ---------------------------
for d in image mask; do
  n=$(ls "$EXT/COVERAGE/test/$d" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" != "191" ]; then
    say "[ABORT] $EXT/COVERAGE/test/$d 有 $n 个文件（应为 191）。"
    say "        修法：从 /root/autodl-tmp/_pending/COVERAGE/{image,mask} 重新复制。"
    exit 3
  fi
done
say "[数据] 外部域 COVERAGE/test: image 191 / mask 191 ✓"

# ---- 2) 收集候选权重（**只取已完成的臂**） ------------------------------------
# ★ 为什么必须有"已完成"这道闸：`full` 臂在训练期间就会不断刷新
#   `outputs/ablation/full/ckpt/ablation_full_best.pt`。若拿它现在的中间权重评一次，
#   由于本脚本是幂等的（json 存在就跳过），**这个中间值会被永久当成 full 的最终行**
#   —— 一个典型的静默失效。所以判据用「该臂已归档」：
#   链路 `archive_arm()` 在每组跑完时写 `_archive/<臂>_<ts>/ablation.json`（只拷
#   json/md/txt，不拷权重；权重留在 outputs/ablation/<臂>/ckpt/）。
DONE_ARMS=$("$PY" - <<'PYEOF'
import glob, json, os
R = "/root/autodl-tmp/vibnet-forgery-detector"
ok = []
for fp in sorted(glob.glob(R + "/outputs/ablation/_archive/*/ablation.json")):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    for k, v in (d or {}).items():
        if ((v or {}).get("cls") or {}).get("acc") is not None and k not in ok:
            ok.append(k)
# 兜底：链路的 chain.status 里 done= 也会记成功臂
try:
    for line in open("/root/autodl-tmp/chain.status", encoding="utf-8", errors="replace"):
        if line.startswith("done="):
            for k in line.split("=", 1)[1].split():
                if k not in ok:
                    ok.append(k)
except FileNotFoundError:
    pass
print(" ".join(ok))
PYEOF
)
say "[已完成] ${DONE_ARMS:-（无）}"

CANDS=""
add(){ [ -n "${2:-}" ] && [ -f "$2" ] && CANDS="$CANDS $1=$2"; }

for arm in $DONE_ARMS; do
  p=$(ls -t "$R/outputs/ablation/$arm/ckpt"/*best*.pt 2>/dev/null | head -1)
  if [ -z "$p" ]; then
    say "[警告] $arm 已归档但找不到权重（$R/outputs/ablation/$arm/ckpt/*best*.pt）-> 跳过"
    continue
  fi
  add "$arm" "$p"
done
# 训练域外部参比：09-28 云端正式产物（与消融无关，作为"旧模型"基线）
add formal0928 "$R/checkpoints/vibnet_best.pt"

if [ -z "${CANDS// /}" ]; then
  say "[空转] 目前没有任何**已完成**的臂可评（训练中的臂不算完成）。稍后再跑本脚本。"
  exit 0
fi
say "[候选] ${CANDS# }"

# ---- 3) 逐个评测（幂等） -----------------------------------------------------
for pair in $CANDS; do
  tag=${pair%%=*}
  ckpt=${pair#*=}
  js="$R/$OUTREL/external_COVERAGE_$tag.json"
  if [ -f "$js" ]; then
    say "[跳过] $tag 已产出（$(basename "$js")）"
    continue
  fi
  say "[评测] $tag <- $ckpt"
  # CUDA_VISIBLE_DEVICES 置空：强制走 CPU，绝不与训练抢显存
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/eval_external_domain.py \
      --domain COVERAGE --ckpt "$ckpt" --device cpu --cpu-threads 8 \
      --expect 191 --out "$OUTREL" --tag "$tag" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    say "[完成] $tag rc=0"
  else
    say "[失败] $tag rc=$rc（详见 $LOG 末尾）"
  fi
done

# ---- 4) 刷新合并表 -----------------------------------------------------------
say "[汇总] 渲染 $OUTREL/coverage_rows.md"
"$PY" scripts/report_coverage_rows.py --out "$OUTREL" >> "$LOG" 2>&1 \
  && say "[汇总] 完成" || say "[汇总] 失败 rc=$?"

say "==================== 外部域 COVERAGE 基准行：结束 ===================="
