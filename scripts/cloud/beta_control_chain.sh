#!/bin/bash
# =============================================================================
# β 生效版受控对照链路（full + no_vib）—— 接在主消融链路之后自动执行
#
# 为什么不和主消融链路合成一个脚本
#   主消融表用 `configs/default_crossval.yaml`（β 升温窗 [20,40]），在实况早停下
#   β 峰值只有 0.0050（= 目标的 5%）⇒ VIB 的 KL 正则几乎没参与训练。
#   本链路用 `configs/default_crossval_betafull.yaml`（窗口下移到 [11,18]，
#   零成本验算 β 可达 0.1000 并保持饱和）另跑一组对照。
#   **两组协议不同 ⇒ 不可混表**，所以产物、日志、状态文件、心跳全部独立。
#
# 做什么
#   1) 等主消融链路（ablation_chain.sh）自然结束 —— 否则两个训练会抢同一张卡；
#   2) 依次跑 full -> no_vib（每组一次独立调用，理由同主链路：汇总表是覆盖写）；
#   3) 每组跑完归档 + 跨生成器双行（tag 带 _beta 后缀，不覆盖主表产物）；
#   4) 全部完成后跑 COVERAGE 外部域出行，刷新合并表，打包提示。
#
# 零覆盖保证
#   `src/evaluation/beta_control.py` 直接调 `run_ablation(save_dir=...)`，
#   产物落 `outputs/ablation_betafull/`，**绝不碰** `outputs/ablation/`
#   （那里有 09-30 那次 full 的权威权重，跨生成器双行与 COVERAGE 行都靠它复现）。
#
# 三道闸（同主链路）：① train.amp=true 与 β 窗口 ② 空开关自检(CPU) ③ 预算
#
# 手工干预
#   touch /root/autodl-tmp/beta_control.stop   # 优雅退出（不关机）
#   cat /root/autodl-tmp/beta_control.status ; tail -50 /root/autodl-tmp/beta_control.log
# =============================================================================
set -u

R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/beta_control.log
HB=/root/autodl-tmp/beta_control.heartbeat
STATUS=/root/autodl-tmp/beta_control.status
BUDGET=/root/autodl-tmp/budget.tsv
STOP=/root/autodl-tmp/beta_control.stop
STDERR_LOG=/root/autodl-tmp/beta_control.stderr.log
MAIN_CHAIN_PAT='ablation_chain\.sh'

SAVEDIR=outputs/ablation_betafull          # ★ 与主消融表物理隔离
ARCH="$R/$SAVEDIR/_archive"
CONFIG=configs/default_crossval_betafull.yaml
SCALE=0.82

# 异常退出必须留痕（2026-09-30 05:30 的"无声消失"教训）
trap 'rc=$?; if [ "$rc" != 0 ]; then echo "[$(date "+%F %T")] [FATAL] β 对照链路异常退出 rc=$rc（详情见 $STDERR_LOG）" | tee -a "$LOG"; fi' EXIT

RATE=0.85
RESERVE=0.0
ARM_H=5.5                       # 实测校准（full 实跑 38 轮 / 4.67 h + 缓冲）
MARGIN=1.15
ARMS="${ARMS:-full no_vib}"     # full 先跑，作为本组的 Δ 基准
STRICT_EXCLUDE="progan,biggan,cyclegan,stargan"
MAX_TOTAL_H=24
ARM_TIMEOUT_H=12

cd "$R" || { echo "[ABORT] 项目目录不存在：$R"; exit 1; }
mkdir -p "$ARCH" outputs/ablation_logs
T0=$(date +%s)
DONE_STR=""
BUDGET_EST="?"
FAILS=0

say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; touch "$HB"; }
st(){
  printf 'updated=%s\nphase=%s\ndone=%s\nfails=%s\nbudget_est_cny=%s\n' \
    "$(date +%s)" "$1" "$DONE_STR" "$FAILS" "$BUDGET_EST" > "$STATUS"
}

# ---------------------------------------------------------------- 预算账本
budget_refresh(){
  local ts=0 bal=0 now est
  if [ -f "$BUDGET" ]; then read -r ts bal < "$BUDGET" 2>/dev/null || true; fi
  case "${ts:-}"  in ''|*[!0-9]*)  ts=$(date +%s);; esac
  case "${bal:-}" in ''|*[!0-9.]*) bal=0;; esac
  now=$(date +%s)
  est=$(awk -v b="$bal" -v r="$RATE" -v d="$((now-ts))" \
        'BEGIN{printf "%.4f", b - r*d/3600}')
  est=$(awk -v e="$est" 'BEGIN{printf "%.2f", (e<0?0:e)}')
  printf '%s %s\n' "$now" "$est" > "$BUDGET"
  BUDGET_EST="$est"
}

budget_ok(){
  local rem need
  budget_refresh
  rem=$(awk -v b="$BUDGET_EST" -v r="$RATE" -v res="$RESERVE" \
        'BEGIN{printf "%.2f", (b-res)/r}')
  need=$(awk -v h="$ARM_H" -v m="$MARGIN" 'BEGIN{printf "%.2f", h*m}')
  say "[预算] 账本余额 ${BUDGET_EST} 元 ≈ 可用 ${rem} h；本组开跑门槛 ≥ ${need} h"
  awk -v a="$rem" -v b="$need" 'BEGIN{exit !(a>=b)}'
}

# ------------------------------------------------------------------ 三道闸
guard_config(){
  if ! grep -qE '^[[:space:]]+amp:[[:space:]]+true' "$CONFIG"; then
    say "[闸① 配置] $CONFIG 的 train.amp 不是 true -> 拒绝起跑"
    return 1
  fi
  grep -qE '^[[:space:]]+beta_anchor:[[:space:]]*"?vib' "$CONFIG" \
    || say "[闸① 配置] 警告：beta_anchor 不是 vib，β 退火口径可能跑偏"
  grep -qE '^[[:space:]]+beta_epoch_offset:[[:space:]]*0' "$CONFIG" \
    || say "[闸① 配置] 警告：存在非 0 的 beta_epoch_offset，会跳过 β 退火"
  # 本组的意义全在"β 真的走满" ⇒ 把窗口值打进日志，事后可核
  say "[闸① 配置] $(grep -hE 'beta_warmup_(start|end):' "$CONFIG" | tr -s ' ' | paste -sd' ' -)"
  return 0
}

guard_arm(){   # $1=preset；强制 CPU 跑，避免与训练抢显存
  local p="$1" out rc
  out=$(CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "$PY" scripts/check_ablation_arms.py \
        --config "$CONFIG" --presets "$p" 2>&1); rc=$?
  printf '%s\n' "$out" | tail -8 | sed 's/^/[闸②] /' >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    say "[闸② 空开关] $p 未通过自检 -> 跳过该组"
    return 1
  fi
  say "[闸② 空开关] $p 通过自检"
  return 0
}

# preset -> 独立配置；路径全部指向 $SAVEDIR（与主链路的 _auto_*.yaml 不重名）
make_arm_config(){
  local p="$1" rc
  "$PY" - "$p" "$CONFIG" "$SAVEDIR" >> "$LOG" 2>&1 <<'PYEOF'
import sys, yaml
sys.path.insert(0, ".")
from src.models.vibnet import load_config
from src.evaluation.ablation import apply_preset, describe_preset
p, cfg_path, savedir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = apply_preset(load_config(cfg_path), p)
cfg["project"]["output_dir"] = f"{savedir}/{p}"
cfg["project"]["ckpt_dir"] = f"{savedir}/{p}/ckpt"
out = f"configs/_auto_beta_{p}.yaml"
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(f"[闸③ 配置落地] {out} <- {describe_preset(p)}")
PYEOF
  rc=$?
  [ "$rc" -eq 0 ] && say "[闸③ 配置落地] configs/_auto_beta_${p}.yaml 已生成" \
                   || say "[闸③ 配置落地] $p 生成失败 rc=$rc"
  return "$rc"
}

check_align(){
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, torch
sys.path.insert(0, ".")
from src.models.vibnet import build_model, load_config
cfg = load_config(sys.argv[1])
model = build_model(cfg)
model = model[0] if isinstance(model, tuple) else model
st = torch.load(sys.argv[2], map_location="cpu")
miss, unexp = model.load_state_dict(st.get("model", st), strict=False)
print(f"[对齐] missing={len(miss)} unexpected={len(unexp)}  ckpt={sys.argv[2]}")
if miss:
    print("[对齐] 缺失键示例:", list(miss)[:5])
    sys.exit(3)
PYEOF
}

# ------------------------------------------------------------ 归档 / 判定
archive_arm(){   # $1=preset（只归档 json/md/txt，权重留在 $SAVEDIR/<p>/ckpt）
  local p="$1" ts d tl
  ts=$(date +%Y%m%d_%H%M%S); d="$ARCH/${p}_${ts}"
  mkdir -p "$d"
  for f in ablation.json ablation.md; do
    [ -f "$R/$SAVEDIR/$f" ] && cp -a "$R/$SAVEDIR/$f" "$d/"
  done
  if [ -d "$R/$SAVEDIR/$p" ]; then
    find "$R/$SAVEDIR/$p" -maxdepth 2 -type f \
      \( -name '*.json' -o -name '*.md' -o -name '*.txt' \) \
      -exec cp -a {} "$d/" \; 2>/dev/null
  fi
  tl=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
  [ -n "$tl" ] && [ -f "$tl" ] && cp -a "$tl" "$d/train_nohup.log"
  say "[归档] $d"
}

arm_result_ok(){   # $1=preset：本组汇总表里有该组且 ACC 非空
  "$PY" - "$1" "$R/$SAVEDIR/ablation.json" <<'PYEOF'
import json, os, sys
p, fp = sys.argv[1], sys.argv[2]
if not os.path.exists(fp):
    print(f"[判定] {p}: 汇总表不存在 -> 未产出"); sys.exit(4)
d = json.load(open(fp, encoding="utf-8"))
row = d.get(p) or {}
acc = (row.get("cls") or {}).get("acc")
print(f"[判定] {p}: 在表内={bool(row)}  cls.acc={acc}")
sys.exit(0 if acc is not None else 4)
PYEOF
}

refresh_merge(){
  if "$PY" scripts/merge_ablation.py --archive "$ARCH" --out "$R/$SAVEDIR" \
       >> "$LOG" 2>&1; then
    say "[合并] 已刷新 $SAVEDIR/ablation_merged.{json,md}"
  else
    say "[合并] 失败（不阻断链路）"
  fi
}

# --------------------------------------------------------------- 跨生成器
run_cross_gen(){   # $1=preset；tag 带 _beta 后缀，不覆盖主表产物
  # ⚠ 拆成两句：同一条 local 里后面的赋值读不到前面刚定义的变量（会撞 set -u 静默退出）
  local p="$1" cfg="$2" ck="$3"
  local out="$R/outputs/cross_gen/${p}_beta"
  [ -f "$ck" ] || { say "[跨生成器] $p 找不到权重 $ck -> 跳过"; return 0; }
  mkdir -p "$out"
  if ! check_align "$cfg" "$ck" >> "$LOG" 2>&1; then
    say "[跨生成器] $p 权重与配置键不对齐（missing>0）-> 跳过，避免静默错误数字"
    return 0
  fi
  say "[跨生成器] ${p}_beta 标准 13 生成器 …"
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" scripts/eval_cross_generator.py \
       --config "$cfg" --ckpt "$ck" --per-class 150 --size 224 --device cuda \
       --out "$out" --tag "${p}_beta" --scores-out "$out/scores.npz" \
       > "$R/outputs/cross_gen/${p}_beta_std.log" 2>&1; then
    say "[跨生成器] ${p}_beta 标准集完成"
  else
    say "[跨生成器] ${p}_beta 标准集失败（不阻断链路）"
  fi
  say "[跨生成器] ${p}_beta 严格 9 生成器（剔除 $STRICT_EXCLUDE）…"
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" scripts/eval_cross_generator.py \
       --config "$cfg" --ckpt "$ck" --per-class 150 --size 224 --device cuda \
       --exclude-generators "$STRICT_EXCLUDE" \
       --out "$out/strict" --tag "${p}_beta_strict" \
       > "$R/outputs/cross_gen/${p}_beta_strict.log" 2>&1; then
    say "[跨生成器] ${p}_beta 严格集完成"
  else
    say "[跨生成器] ${p}_beta 严格集失败（不阻断链路）"
  fi
}

# ------------------------------------------------------------------ 跑一组
run_arm(){   # $1=preset；前台跑（链路已 setsid 脱离 ssh）
  local p="$1" rc ts log tr before after
  ts=$(date +%Y%m%d_%H%M%S)
  log="$R/outputs/ablation_logs/beta_${p}_s082_${ts}.log"

  tr="$R/$SAVEDIR/$p/train_log.txt"
  if [ -f "$tr" ]; then
    mv "$tr" "${tr%.txt}.prev-${ts}.txt" && say "[$p] 已归档上一轮 train_log.txt"
  fi

  before=""
  [ -f "$R/$SAVEDIR/ablation.json" ] && \
    before=$(md5sum "$R/$SAVEDIR/ablation.json" | awk '{print $1}')

  say "[$p] 起跑（β 生效版；日志 $log；硬超时 ${ARM_TIMEOUT_H}h）"
  st "train:${p}(beta)"
  echo "$log" > /root/autodl-tmp/.current_beta_log
  touch "$HB"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    timeout -k 60 $(( ARM_TIMEOUT_H * 3600 )) \
    "$PY" -m src.evaluation.beta_control --config "$CONFIG" \
      --presets "$p" --epochs-scale "$SCALE" --save-dir "$SAVEDIR" > "$log" 2>&1
  rc=$?
  say "[$p] 进程退出 rc=$rc（124=单组硬超时）"

  after=""
  [ -f "$R/$SAVEDIR/ablation.json" ] && \
    after=$(md5sum "$R/$SAVEDIR/ablation.json" | awk '{print $1}')
  if [ -z "$after" ] || [ "$after" = "$before" ]; then
    say "[$p] [!] 汇总表未刷新 -> 判定本组失败"
    return 1
  fi
  say "[$p] 汇总表已刷新"
  return 0
}

# ==========================================================================
say "==================== β 对照链路启动 PID=$$ ===================="
say "计划：等主消融链路结束 -> 依次跑 [$ARMS]（配置 $CONFIG）"
say "产物根：$SAVEDIR（与主消融表 outputs/ablation 物理隔离）"
budget_refresh
say "[预算] 启动时账本余额 ${BUDGET_EST} 元"
st "wait:main-chain"

# ---------------- 阶段 1：等主消融链路结束 ----------------
while pgrep -f "$MAIN_CHAIN_PAT" >/dev/null 2>&1; do
  touch "$HB"; st "wait:main-chain"
  if [ -e "$STOP" ]; then say "检测到 $STOP -> 优雅退出（不关机）"; st "stopped"; exit 0; fi
  sleep 300
done
say "[接力] 主消融链路已结束（等待 $(( ($(date +%s)-T0)/60 )) 分钟）-> 开始 β 对照实验"

# ---------------- 阶段 2：逐臂跑 ----------------
for p in $ARMS; do
  if [ -e "$STOP" ]; then say "检测到 $STOP -> 停止安排新组"; break; fi
  if [ $(( $(date +%s) - T0 )) -gt $(( MAX_TOTAL_H*3600 )) ]; then
    say "链路总时长超过 ${MAX_TOTAL_H}h -> 停止安排新组"; break
  fi
  if [ "$FAILS" -ge 2 ]; then
    say "[!] 连续 ${FAILS} 组失败 -> 判定系统性故障，停止链路"; break
  fi
  if ! guard_config; then break; fi

  # full 是本组的 Δ 基准（无 override）⇒ 不做"空开关"自检，其余臂必须过
  if [ "$p" != "full" ]; then
    if ! budget_ok; then
      say "[预算] 不足 -> 不安排 $p（宁可少跑一组，也不开一个跑不完的组）"; break
    fi
    if ! guard_arm "$p"; then FAILS=$((FAILS+1)); continue; fi
  else
    if ! budget_ok; then
      say "[预算] 不足 -> 不安排 $p"; break
    fi
  fi
  if ! make_arm_config "$p"; then FAILS=$((FAILS+1)); continue; fi

  if run_arm "$p" && arm_result_ok "$p"; then
    DONE_STR="${DONE_STR}${DONE_STR:+,}${p}"; st "done:${p}(beta)"
    archive_arm "$p"
    run_cross_gen "$p" "configs/_auto_beta_${p}.yaml" \
      "$R/$SAVEDIR/$p/ckpt/ablation_${p}_best.pt"
    refresh_merge
    FAILS=0
  else
    FAILS=$((FAILS+1))
    say "[$p] 失败（连续失败计数 $FAILS）"
    archive_arm "$p"
    refresh_merge
  fi
  touch "$HB"
done

# ---------------- 阶段 3：COVERAGE 外部域出行 ----------------
say "[COVERAGE] 为 β 组补外部域基准行 …"
if [ -f /root/autodl-tmp/eval_coverage_row_beta.sh ]; then
  bash /root/autodl-tmp/eval_coverage_row_beta.sh >> "$LOG" 2>&1 \
    && say "[COVERAGE] 出行完成" || say "[COVERAGE] 出行失败（不阻断）"
else
  say "[COVERAGE] 未找到 eval_coverage_row_beta.sh -> 跳过"
fi

# ---------------- 收尾 ----------------
st "finished"
say "==================== β 对照链路结束：已完成 [${DONE_STR:-无}] ===================="
refresh_merge
BUNDLE=/root/autodl-tmp/results_beta_$(date +%Y%m%d_%H%M%S).tar.gz
tar -czf "$BUNDLE" -C "$R" \
  "$SAVEDIR/_archive" "$SAVEDIR/ablation_merged.json" "$SAVEDIR/ablation_merged.md" \
  "$SAVEDIR/ablation.json" "$SAVEDIR/ablation.md" \
  outputs/coverage_ext outputs/cross_gen \
  $(ls -d "$R"/configs/_auto_beta_*.yaml 2>/dev/null | sed "s#^$R/##") \
  2>/dev/null || true
say "结果包：$BUNDLE"
say "守夜脚本将在约 15 分钟后自动关机（数据盘保留，开机即恢复）"
