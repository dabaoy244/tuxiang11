#!/bin/bash
# =============================================================================
# VIB-Net 无人值守消融链路
#
# 做什么
#   1) 等当前正在跑的 `full` 组自然结束（进程消失即判定结束）；
#   2) 依次跑 no_vib -> no_cross_attention -> no_edge，每组一次独立调用；
#   3) 每组跑完立刻归档汇总表，再补跑跨生成器评测，最后刷新合并表。
#
# 为什么一组一次独立调用（不要用 --presets 串起来）
#   `src/evaluation/ablation.py` 只在**全部 presets 循环结束后**才写汇总表
#   `outputs/ablation/ablation.json|.md`（而且是覆盖写）。串 3 组 ≈ 20 小时，
#   中途任何一组崩掉，**前面已跑完组的汇总一起丢**。分轮跑把风险切成小块，
#   汇总改由 `scripts/merge_ablation.py` 合并（它用 full 行的原始指标重算 Δ，
#   不会踩「拿 0 当基线 → 得到 -0.65 假差值」那个坑）。
#
# 每组跑之前过三道闸
#   ① 配置闸：`train.amp` 必须为 true（09-29 的 stage2 OOM 就是它引起的）；
#   ② 空开关闸：跑 `scripts/check_ablation_arms.py`（CUDA_VISIBLE_DEVICES="" 强制 CPU，
#      不占显存、不干扰训练），确认该 preset 真的改变了模型 ——
#      否则会白烧 6.6 小时得到一个与 full 完全相同的"消融臂"；
#   ③ 预算闸：账户是预付费余额，余额耗尽会被平台强制关机、当前组白跑。
#      **宁可少跑一组，也不开一个跑不完的组。**
#      余额账本 `/root/autodl-tmp/budget.tsv`（一行两个字段：<unix_ts> <余额元>）。
#      本机侧每 2 小时的汇报任务会写入浏览器里读到的**真实**读数；
#      链路每次开跑前也会按 RATE 自行记账刷新，避免文件过期导致误判"没钱"。
#      账上永久预留 RESERVE 元 —— 那是留给「启用 COVERAGE 后的正式三阶段训练」的，
#      消融再重要也不能把项目下一步的钱吃掉。
#
# 手工干预
#   touch /root/autodl-tmp/chain.stop      # 让链路在当前位置优雅退出（不关机）
#   touch /root/autodl-tmp/nightwatch.off  # 彻底禁用自动关机
#   查看状态：cat /root/autodl-tmp/chain.status ; tail -50 /root/autodl-tmp/chain.log
# =============================================================================
set -u

R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/chain.log
HB=/root/autodl-tmp/chain.heartbeat
STATUS=/root/autodl-tmp/chain.status
BUDGET=/root/autodl-tmp/budget.tsv
STOP=/root/autodl-tmp/chain.stop
ARCH="$R/outputs/ablation/_archive"
CONFIG=configs/default_crossval.yaml
SCALE=0.82                      # β 退火走满的下限（见 memory §5）
STDERR_LOG=/root/autodl-tmp/chain.stderr.log

# ★ 异常退出必须留痕（2026-09-30 05:30 踩过）：
#   本脚本 `set -u`，而启动器是 `> /dev/null 2>&1` ⇒ 一旦撞到未定义变量，
#   bash 打印的 "...: unbound variable" 进黑洞，脚本"无声消失"、chain.log 一行都没有，
#   排查只能靠手工重放。加 EXIT trap 把非零退出连同退出码写进 chain.log。
#   （真正的信息在 $STDERR_LOG，见 chain_restart.sh 的启动重定向。）
trap 'rc=$?; if [ "$rc" != 0 ]; then echo "[$(date "+%F %T")] [FATAL] 链路异常退出 rc=$rc（错误详情见 $STDERR_LOG）" | tee -a "$LOG"; fi' EXIT

# 计费速率：控制台标价 ￥1.14/时，但实测消耗约 ￥0.84/时（01:05→01:25 读数
# 21.52->21.24 是 **20 分钟** 0.28 元；早期写的"25 分钟/0.67 元每时"是算错的，别再引用）。
# 取 0.85 作账：略高于实测、明显低于标价，既不会乐观到"开了跑不完的组"，
# 也不会悲观到"明明有钱却提前停"。
RATE=0.85
# 预留余额：**已于 2026-09-30 改为 0**。原为「启用 COVERAGE 后的正式训练」预留 ￥8，
# 但用户已把 COVERAGE 定为纯外部测试域（不参与训练）⇒ 这笔预留失去用途；
# 而它会实打实地把第 6 个臂挡在门外（门槛 BUDGET_EST ≥ 14.65，跑完 5 臂后只剩 ≈14.56）。
RESERVE=0.0
# 单组预计小时：**实测校准**。原来写 6.8（按 57 轮 × 6.5 min 估），但 2026-09-30
# `full` 实跑只 38 轮（stage1/stage3 都早停）、耗时 4.67 h（00:48→05:28）
# ⇒ 6.8 高估 46%，会让预算闸提前停机白丢一个臂。取 5.5 = 实测 4.67 + 归档/评测缓冲。
ARM_H=5.5
MARGIN=1.15                     # 起跑门槛系数：剩余可用 ≥ ARM_H*MARGIN 才开跑
# 按优先级排序。前 3 个是"必跑"（VIB 是模型名里的核心贡献、CS-CAM 与边缘监督是申报书卖点）；
# 后 3 个是"有钱就跑"。链路每一组开跑前都会重算预算，钱不够就停在前一组 ——
# 宁可少跑一组，也不开一个跑不完的组。
ARMS="${ARMS:-no_vib no_cross_attention no_edge no_learnable_mask no_phase no_grad_stop}"
STRICT_EXCLUDE="progan,biggan,cyclegan,stargan"
MAX_TOTAL_H=48                  # 链路总时长上限
ARM_TIMEOUT_H=12                # 单组硬超时（超时判定卡死，避免无限占卡）

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
alive_main(){ pgrep -f 'src\.evaluation\.ablation' >/dev/null 2>&1; }

# ---------------------------------------------------------------- 预算账本
budget_refresh(){
  local ts=0 bal=0 now est
  if [ -f "$BUDGET" ]; then read -r ts bal < "$BUDGET" 2>/dev/null || true; fi
  case "${ts:-}"  in ''|*[!0-9]*)   ts=$(date +%s);; esac
  case "${bal:-}" in ''|*[!0-9.]*)  bal=0;; esac
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
  say "[预算] 账本余额 ${BUDGET_EST} 元 ≈ 可用 ${rem} h（已扣预留 ${RESERVE} 元）；本组开跑门槛 ≥ ${need} h"
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
  return 0
}

guard_arm(){   # $1=preset；强制 CPU 跑，避免与训练抢显存
  local p="$1" out rc
  out=$(CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "$PY" scripts/check_ablation_arms.py \
        --config "$CONFIG" --presets "$p" 2>&1); rc=$?
  printf '%s\n' "$out" | tail -8 | sed 's/^/[闸②] /' >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    say "[闸② 空开关] $p 未通过自检 -> 跳过该组（否则白烧 ${ARM_H}h 得到一个与 full 相同的臂）"
    return 1
  fi
  say "[闸② 空开关] $p 通过自检（参数量/前向输出/梯度流至少一项确实变了）"
  return 0
}

# 把 preset 落到一份独立配置里。
# 为什么必须这么做：跨生成器评测要拿臂的权重去 build_model，
# 而臂的结构与 full 不同（如 no_vib 的 cls_head 输入维度 512 vs 256）。
# `eval_cross_generator.py` 里是 `load_state_dict(..., strict=False)`，
# 配置对不上不会报错 —— 只会静默给出错误结构的数字。
make_arm_config(){
  local p="$1" rc
  "$PY" - "$p" "$CONFIG" >> "$LOG" 2>&1 <<'PYEOF'
import sys, yaml
sys.path.insert(0, ".")
from src.models.vibnet import load_config
from src.evaluation.ablation import apply_preset, describe_preset
p, cfg_path = sys.argv[1], sys.argv[2]
cfg = apply_preset(load_config(cfg_path), p)
cfg["project"]["output_dir"] = f"outputs/ablation/{p}"
cfg["project"]["ckpt_dir"] = f"outputs/ablation/{p}/ckpt"
out = f"configs/_auto_{p}.yaml"
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(f"[闸③ 配置落地] {out} <- {describe_preset(p)}")
PYEOF
  rc=$?
  [ "$rc" -eq 0 ] && say "[闸③ 配置落地] configs/_auto_${p}.yaml 已生成" \
                   || say "[闸③ 配置落地] $p 生成失败 rc=$rc"
  return "$rc"
}

# 权重-结构键对齐守卫：missing>0 说明配置与权重不配套 -> 拒绝出数字
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
archive_arm(){   # $1=preset
  local p="$1" ts d tl
  ts=$(date +%Y%m%d_%H%M%S); d="$ARCH/${p}_${ts}"
  mkdir -p "$d"
  for f in ablation.json ablation.md; do
    [ -f "$R/outputs/ablation/$f" ] && cp -a "$R/outputs/ablation/$f" "$d/"
  done
  if [ -d "$R/outputs/ablation/$p" ]; then
    find "$R/outputs/ablation/$p" -maxdepth 2 -type f \
      \( -name '*.json' -o -name '*.md' -o -name '*.txt' \) \
      -exec cp -a {} "$d/" \; 2>/dev/null
  fi
  tl=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
  [ -n "$tl" ] && [ -f "$tl" ] && cp -a "$tl" "$d/train_nohup.log"
  say "[归档] $d"
}

arm_result_ok(){   # $1=preset：汇总表里有该组且 ACC 非空
  "$PY" - "$1" "$R/outputs/ablation/ablation.json" <<'PYEOF'
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
  if "$PY" scripts/merge_ablation.py --archive "$ARCH" --out "$R/outputs/ablation" \
       >> "$LOG" 2>&1; then
    say "[合并] 已刷新 outputs/ablation/ablation_merged.{json,md}"
  else
    say "[合并] 失败（不阻断链路，稍后可手工重跑）"
  fi
}

# --------------------------------------------------------------- 跨生成器
run_cross_gen(){   # $1=preset $2=config $3=ckpt；best-effort，失败不阻断
  # ⚠ 必须拆成两句：bash 在**同一条** `local` 里，后面的赋值读不到前面刚定义的变量
  #   （`local p="$1" out="...$p"` 中 $p 仍是未定义）⇒ 撞上 set -u 当场退出。
  #   2026-09-30 05:30 就是死在这里：full 成功后第一次走到本函数，
  #   因 stderr 被丢进 /dev/null，chain.log 一行都没有，看起来像"进程凭空消失"。
  local p="$1" cfg="$2" ck="$3"
  local out="$R/outputs/cross_gen/$p"
  [ -f "$ck" ] || { say "[跨生成器] $p 找不到权重 $ck -> 跳过"; return 0; }
  mkdir -p "$out"
  if ! check_align "$cfg" "$ck" >> "$LOG" 2>&1; then
    say "[跨生成器] $p 权重与配置键不对齐（missing>0）-> 跳过，避免静默错误数字"
    return 0
  fi
  say "[跨生成器] $p 标准 13 生成器 …"
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" scripts/eval_cross_generator.py \
       --config "$cfg" --ckpt "$ck" --per-class 150 --size 224 --device cuda \
       --out "$out" --tag "$p" --scores-out "$out/scores.npz" \
       > "$R/outputs/cross_gen/${p}_std.log" 2>&1; then
    say "[跨生成器] $p 标准集完成"
  else
    say "[跨生成器] $p 标准集失败（不阻断链路）"
  fi
  say "[跨生成器] $p 严格 9 生成器（剔除 $STRICT_EXCLUDE）…"
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" scripts/eval_cross_generator.py \
       --config "$cfg" --ckpt "$ck" --per-class 150 --size 224 --device cuda \
       --exclude-generators "$STRICT_EXCLUDE" \
       --out "$out/strict" --tag "${p}_strict" \
       > "$R/outputs/cross_gen/${p}_strict.log" 2>&1; then
    say "[跨生成器] $p 严格集完成"
  else
    say "[跨生成器] $p 严格集失败（不阻断链路）"
  fi
}

# ------------------------------------------------------------------ 跑一组
run_arm(){   # $1=preset；前台跑（链路自身已 setsid 脱离 ssh，子进程不会被 sshd 收走）
  local p="$1" rc ts log tr before after
  ts=$(date +%Y%m%d_%H%M%S)
  log="$R/outputs/ablation_logs/${p}_s082_${ts}.log"

  # 该组的 train_log.txt 是 append 打开、重跑不清空 -> 先归档，避免看板取到上一轮尾巴
  tr="$R/outputs/ablation/$p/train_log.txt"
  if [ -f "$tr" ]; then
    mv "$tr" "${tr%.txt}.prev-${ts}.txt" && say "[$p] 已归档上一轮 train_log.txt"
  fi

  before=""
  [ -f "$R/outputs/ablation/ablation.json" ] && \
    before=$(md5sum "$R/outputs/ablation/ablation.json" | awk '{print $1}')

  say "[$p] 起跑（日志 $log；硬超时 ${ARM_TIMEOUT_H}h）"
  st "train:$p"
  echo "$log" > /root/autodl-tmp/.current_train_log
  touch "$HB"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    timeout -k 60 $(( ARM_TIMEOUT_H * 3600 )) \
    "$PY" -m src.evaluation.ablation --config "$CONFIG" \
      --mode train --presets "$p" --epochs-scale "$SCALE" > "$log" 2>&1
  rc=$?
  say "[$p] 进程退出 rc=$rc（124=单组硬超时）"

  after=""
  [ -f "$R/outputs/ablation/ablation.json" ] && \
    after=$(md5sum "$R/outputs/ablation/ablation.json" | awk '{print $1}')
  if [ -z "$after" ] || [ "$after" = "$before" ]; then
    say "[$p] [!] 汇总表未刷新 -> 判定本组失败"
    return 1
  fi
  say "[$p] 汇总表已刷新"
  return 0
}

# ==========================================================================
say "==================== 消融链路启动 PID=$$ ===================="
say "计划：等 full 结束 -> 依次跑 [$ARMS]"
say "三道闸：① train.amp=true ② 消融臂自检(CPU) ③ 剩余预算 ≥ $(awk -v h="$ARM_H" -v m="$MARGIN" 'BEGIN{printf "%.1f", h*m}') h"
budget_refresh
say "[预算] 启动时账本余额 ${BUDGET_EST} 元（来源 $(stat -c %y "$BUDGET" 2>/dev/null | cut -c1-19)）"
st "wait-full"

# ---------------- 阶段 1：等正在跑的 full ----------------
while alive_main; do
  touch "$HB"; st "wait-full"
  if [ -e "$STOP" ]; then say "检测到 $STOP -> 链路优雅退出（不关机）"; st "stopped"; exit 0; fi
  sleep 60
done
say "[full] 训练进程已结束（链路等待 $(( ($(date +%s)-T0)/60 )) 分钟）"

archive_arm full
if arm_result_ok full; then
  say "[full] 汇总表正常 -> 计入结果"
  DONE_STR="full"; st "done:full"
  run_cross_gen full "$CONFIG" "$R/outputs/ablation/full/ckpt/ablation_full_best.pt"
  refresh_merge
else
  say "[full] [!] 没拿到可用的汇总表（可能崩在 stage2/3 或早退）"
  if budget_ok; then
    say "[full] 预算允许 -> 重跑 full 一次"
    if guard_config && run_arm full && arm_result_ok full; then
      DONE_STR="full"; st "done:full"
      archive_arm full
      run_cross_gen full "$CONFIG" "$R/outputs/ablation/full/ckpt/ablation_full_best.pt"
      refresh_merge
    else
      say "[full] 重跑仍未成功 -> 放弃 full，继续跑其余臂（合并表里 Δvs full 会记为 —）"
    fi
  else
    say "[full] 预算不足 -> 不重跑，直接继续跑其余臂"
  fi
fi

# ---------------- 阶段 2：其余臂 ----------------
for p in $ARMS; do
  if [ -e "$STOP" ]; then say "检测到 $STOP -> 停止安排新组"; break; fi
  if [ $(( $(date +%s) - T0 )) -gt $(( MAX_TOTAL_H*3600 )) ]; then
    say "链路总时长超过 ${MAX_TOTAL_H}h -> 停止安排新组"; break
  fi
  if [ "$FAILS" -ge 2 ]; then
    say "[!] 连续 ${FAILS} 组失败 -> 判定系统性故障，停止链路（避免继续烧钱）"; break
  fi
  if ! guard_config; then break; fi
  if ! budget_ok; then
    say "[预算] 不足 -> 不安排 $p（宁可少跑一组，也不开一个跑不完的组）"; break
  fi
  if ! guard_arm "$p"; then FAILS=$((FAILS+1)); continue; fi
  if ! make_arm_config "$p"; then FAILS=$((FAILS+1)); continue; fi

  if run_arm "$p" && arm_result_ok "$p"; then
    DONE_STR="${DONE_STR},${p}"; st "done:$p"
    archive_arm "$p"
    run_cross_gen "$p" "configs/_auto_${p}.yaml" \
      "$R/outputs/ablation/$p/ckpt/ablation_${p}_best.pt"
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

# ---------------- 收尾 ----------------
st "finished"
say "==================== 链路结束：已完成 [${DONE_STR:-无}] ===================="
refresh_merge
BUNDLE=/root/autodl-tmp/results_bundle_$(date +%Y%m%d_%H%M%S).tar.gz
tar -czf "$BUNDLE" -C "$R" \
  outputs/ablation/_archive outputs/ablation/ablation_merged.json \
  outputs/ablation/ablation_merged.md outputs/ablation/ablation.json \
  outputs/ablation/ablation.md outputs/cross_gen \
  $(ls -d "$R"/configs/_auto_*.yaml 2>/dev/null | sed "s#^$R/##") \
  2>/dev/null || true
say "结果包：$BUNDLE"
say "守夜脚本将在约 15 分钟后自动关机（数据盘保留，开机即恢复）"
