#!/bin/bash
# =============================================================================
# 消融链路「开跑前离线自检」
#
# 为什么需要：链路真正开跑要等到当前 full 组结束（约 5 小时后），
# 那时人已经不在电脑前。链路里最容易出错的不是训练本身，而是我新写的三段小逻辑：
#   A) 臂配置生成（跨生成器评测必须用带 preset 的配置，否则权重结构对不上、
#      而 load_state_dict(strict=False) 不会报错，只会静默给出错误数字）
#   B) 权重-结构键对齐守卫本身有没有效
#   C) 预算算术与门槛判断（决定"开不开下一组"，错了就是白烧钱或白留钱）
# 所以提前把它们单独跑一遍。全部在 CPU 上完成，不碰 GPU、不干扰正在进行的训练。
#
# ★ 本脚本"原样抽取"运行中链路脚本里的预算函数来测（sed 抽取，不改动链路脚本本身 ——
#   bash 是边读边执行脚本的，覆盖正在运行的脚本会导致不可预期行为）。
#
# 用法：bash /root/autodl-tmp/preflight_chain.sh
# =============================================================================
set -u
R=/root/autodl-tmp/vibnet-forgery-detector
PY=/root/miniconda3/bin/python
CONFIG=configs/default_crossval.yaml
CHAIN=/root/autodl-tmp/ablation_chain.sh
cd "$R" || { echo "[ABORT] 项目目录不存在"; exit 1; }

echo "################ A) 生成臂配置 ################"
for p in no_vib no_cross_attention no_edge no_learnable_mask no_phase no_grad_stop; do
  "$PY" - "$p" "$CONFIG" <<'PYEOF'
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
print(f"  OK  {out:34s} <- {describe_preset(p)}")
PYEOF
  [ $? -eq 0 ] || echo "  [FAIL] $p 配置生成失败"
done
ls configs/_auto_*.yaml 2>/dev/null | wc -l | sed 's/^/  生成文件数: /'

echo
echo "################ B) 键对齐守卫（拿 full 的 best.pt 验证守卫本身有效） ################"
"$PY" - "$CONFIG" "$R/outputs/ablation/full/ckpt/ablation_full_best.pt" <<'PYEOF'
import sys, torch
sys.path.insert(0, ".")
from src.models.vibnet import build_model, load_config
cfg = load_config(sys.argv[1])
model = build_model(cfg)
model = model[0] if isinstance(model, tuple) else model
st = torch.load(sys.argv[2], map_location="cpu")
miss, unexp = model.load_state_dict(st.get("model", st), strict=False)
print(f"  [对齐] missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("  缺失键示例:", list(miss)[:5]); sys.exit(3)
PYEOF
echo "  退出码 $? （0 = 键对齐；3 = 有缺失键，守卫会拦住）"

echo
echo "######## C) 预算函数（从链路脚本里原样抽出，逐场景验算） ########"
T=/tmp/chain_budget_test.sh
{
  echo 'BUDGET=/tmp/budget_probe.tsv'
  echo 'RATE=0.85; RESERVE=8.0; ARM_H=6.8; MARGIN=1.15; BUDGET_EST="?"'
  echo 'say(){ echo "    [log] $*"; }'
  sed -n '/^budget_refresh()/,/^}$/p' "$CHAIN"
  sed -n '/^budget_ok()/,/^}$/p' "$CHAIN"
  cat <<'EOS'

echo "  场景 1：余额 31.24 元、刚记的 -> 应通过"
printf '%s 31.24\n' "$(date +%s)" > "$BUDGET"
budget_ok && echo "    => 通过" || echo "    => 拒绝"

echo "  场景 2：余额 9.00 元（仅够 ~1.2h）-> 应拒绝"
printf '%s 9.00\n' "$(date +%s)" > "$BUDGET"
budget_ok && echo "    => 通过" || echo "    => 拒绝"

echo "  场景 3：余额 31.24 元、但是 12 小时前的读数 -> 应扣掉约 10.2 元后判断"
printf '%s 31.24\n' "$(( $(date +%s) - 43200 ))" > "$BUDGET"
budget_ok && echo "    => 通过" || echo "    => 拒绝"

echo "  场景 4：余额 60.00 元、12 小时前 -> 应仍通过"
printf '%s 60.00\n' "$(( $(date +%s) - 43200 ))" > "$BUDGET"
budget_ok && echo "    => 通过" || echo "    => 拒绝"

echo "  场景 5：文件损坏（非数字）-> 应保守拒绝，不能崩"
printf 'abc def\n' > "$BUDGET"
budget_ok && echo "    => 通过（不该发生）" || echo "    => 拒绝"
EOS
} > "$T"
bash "$T"
rm -f "$T" /tmp/budget_probe.tsv

echo
echo "######## D) arm_result_ok 判定（无表 / 有表 / ACC 为空） ########"
T2=/tmp/chain_result_test.sh
{
  echo 'PY=/root/miniconda3/bin/python'
  echo 'R='$R
  sed -n '/^arm_result_ok()/,/^}$/p' "$CHAIN"
  cat <<'EOS'
echo "  场景 1：汇总表里没有该组 -> 应 rc=4"
echo '{"full": {"cls": {"acc": 0.69}}}' > /tmp/ab_probe.json
arm_result_ok no_vib; echo "     rc=$?"
echo "  场景 2：有该组且 ACC 非空 -> 应 rc=0"
echo '{"full": {"cls": {"acc": 0.69}}, "no_vib": {"cls": {"acc": 0.65}}}' > /tmp/ab_probe.json
arm_result_ok no_vib; echo "     rc=$?"
echo "  场景 3：有该组但 ACC 为 null（跑到一半崩）-> 应 rc=4"
echo '{"no_vib": {"cls": {"acc": null}}}' > /tmp/ab_probe.json
arm_result_ok no_vib; echo "     rc=$?"
rm -f /tmp/ab_probe.json
EOS
} > "$T2"
# arm_result_ok 内部用的是固定路径 $R/outputs/ablation/ablation.json，这里换掉以便探测
sed -i 's#"\$R/outputs/ablation/ablation.json"#"/tmp/ab_probe.json"#' "$T2"
bash "$T2"
rm -f "$T2"

echo
echo "######## 自检完成：以上四项全部符合预期，链路才值得无人值守 ########"
