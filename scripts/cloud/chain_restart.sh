#!/bin/bash
# =============================================================================
# 幂等重启消融链路（链路中断后专用）
#
# 为什么不能直接再跑一遍 ablation_chain.sh（两个都会白烧钱）
#   ① 汇总表是**覆盖写**的：`outputs/ablation/ablation.json` 每轮只装当前这轮的
#      臂。跑完 no_vib 之后表里只剩 no_vib，链路的阶段 1 就查不到 full 行，
#      于是误判"full 没跑成功" -> 重跑 full（6.6 小时）。
#   ② 链路的 ARMS 默认是全部 6 个臂，直接重启会把已完成的臂再跑一遍
#      （6.8 小时/组）。
#
# 本脚本做的事
#   1) 扫归档目录，用**成功**的归档行重建 `outputs/ablation/ablation.json`
#      （成功 = 该臂的 cls.acc 非空；链路对失败的臂也会归档残迹，不能算成功）；
#   2) 只把"还没成功的臂"通过 ARMS=... 传给链路；
#   3) 若已有链路在跑则拒绝启动（先确认是不是卡死）。
#
# 用法：bash /root/autodl-tmp/chain_restart.sh
#       bash /root/autodl-tmp/chain_restart.sh --dry-run   # 只做重建与剩余臂计算，不启动
#
# 可覆盖 R / ARCH 便于在临时目录上做自测（--dry-run 配合使用，不会真的启动链路）。
# =============================================================================
set -u
R="${R:-/root/autodl-tmp/vibnet-forgery-detector}"
ARCH="${ARCH:-$R/outputs/ablation/_archive}"
PY="${PY:-/root/miniconda3/bin/python}"
ALL="full no_vib no_cross_attention no_edge no_learnable_mask no_phase no_grad_stop"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

cd "$R" || { echo "[ABORT] 项目目录不存在"; exit 1; }
[ -d "$ARCH" ] || { echo "[ABORT] 归档目录不存在：$ARCH"; exit 1; }

echo "=== 1) 用成功的归档行重建汇总表（保住 full 行，避免阶段 1 误判） ==="
DONE=$("$PY" - "$ARCH" "$R/outputs/ablation/ablation.json" <<'PYEOF'
import glob, json, os, sys
arch, out = sys.argv[1], sys.argv[2]
merged = {}
for fp in sorted(glob.glob(os.path.join(arch, "*", "ablation.json"))):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"  跳过 {fp}: {e}\n"); continue
    for k, v in d.items():
        row = v or {}
        acc = (row.get("cls") or {}).get("acc")
        old = (merged.get(k) or {}).get("cls") or {}
        if k not in merged or (old.get("acc") is None and acc is not None):
            merged[k] = row
json.dump(merged, open(out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=float)
ok = sorted(k for k, v in merged.items() if ((v.get("cls") or {}).get("acc")) is not None)
sys.stderr.write("  表内臂: " + (", ".join(sorted(merged)) or "（空）") + "\n")
sys.stderr.write("  其中成功: " + (", ".join(ok) or "（无）") + "\n")
print(" ".join(ok))
PYEOF
)

echo "=== 2) 计算剩余臂 ==="
REM=""
for p in $ALL; do
  case " $DONE " in *" $p "*) continue;; esac
  REM="$REM $p"
done
REM=$(echo $REM)
echo "  已成功: ${DONE:-（无）}"
echo "  剩余  : ${REM:-（无）}"
if [ -z "$REM" ]; then echo "  没有剩余臂 -> 无需重启"; exit 0; fi

if [ "$DRY" = 1 ]; then
  echo "[dry-run] 到此为止：不检查链路进程、不启动链路。"
  exit 0
fi

if pgrep -f 'ablation_chain\.sh' >/dev/null 2>&1; then
  echo "[ABORT] 已有链路在跑。先 tail -30 /root/autodl-tmp/chain.log 确认它是不是卡死；"
  echo "        确实卡死再 kill 掉它的 PID，然后重跑本脚本。"
  exit 3
fi

echo "=== 3) 启动链路（ARMS=$REM） ==="
rm -f /root/autodl-tmp/chain.stop
ARMS="$REM" setsid nohup /root/autodl-tmp/ablation_chain.sh > /dev/null 2>&1 < /dev/null &
sleep 4
echo "  链路进程数: $(pgrep -c -f ablation_chain.sh)"
echo "  --- chain.log 尾部 ---"
tail -6 /root/autodl-tmp/chain.log
