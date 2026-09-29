#!/usr/bin/env bash
# ============================================================================
#  run_step.sh —— 「在可见窗口里跑一个云端步骤，然后截图」的封装。
#
#  为什么要有它：
#  起窗 / 等命令结束 / 截图 / 记日志，这套动作每次都要做一遍，散在各处极易漏步
#  （漏了 hold 就截不到结果，漏了等结束就截到半截）。封成一条命令，每次调用都一致。
#
#  用法：
#      bash outputs/screen/run_step.sh \
#          --out outputs/screenshots/04_probe.png \
#          --title "VIB-Net CLOUD step04 probe" \
#          --log  outputs/screen/step04_probe.log \
#          --banner "第 4 步 / 实测 GPU 速度" \
#          --mid 25 \
#          --hold 120 \
#          -- ssh autodl 'cd /root/autodl-tmp/vibnet-forgery-detector && PY=/root/miniconda3/bin/python bash scripts/cloud_autodl.sh probe'
#
#  退出码：0 = 截到图；非 0 = 起窗失败 / 截图失败（**不会**静默当成成功）
# ============================================================================
set -uo pipefail

# ★ 路径必须取 Windows 形式（`pwd -W` → D:/picture/...）。
#   用 `pwd` 会得到 /d/picture/...，传给 Windows 版 python.exe 时 MSYS 会把它
#   当成"相对当前盘的 POSIX 路径"再翻译一次，变成 D:\d\picture\... → 找不到文件。
#   （实测报错：can't open file 'D:\\d\\picture\\...\\vis.py'）
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
PY="${PY:-C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"

OUT=""; TITLE=""; LOG=""; BANNER=""; MID=0; HOLD=60; TIMEOUT=1800
while [ $# -gt 0 ]; do
    case "$1" in
        --out)     OUT="$2"; shift 2 ;;
        --title)   TITLE="$2"; shift 2 ;;
        --log)     LOG="$2"; shift 2 ;;
        --banner)  BANNER="$2"; shift 2 ;;
        --mid)     MID="$2"; shift 2 ;;
        --hold)    HOLD="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --)        shift; break ;;
        *)         echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

[ -z "$OUT" ]   && { echo "缺 --out"   >&2; exit 2; }
[ -z "$TITLE" ] && { echo "缺 --title" >&2; exit 2; }
[ -z "$LOG" ]   && { echo "缺 --log"   >&2; exit 2; }
[ $# -eq 0 ]    && { echo "没给要跑的命令（-- 之后）" >&2; exit 2; }

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"
# ★ 清掉旧日志与旧哨兵。
#   不清哨兵的话，等待循环会立刻命中**上一轮**留下的标记，截到一张还没开始跑的空白窗口。
: > "$LOG"
rm -f "$LOG.done"

echo ">>> 起窗: $TITLE"
"$PY" "$HERE/vis.py" launch --title "$TITLE" --log "$LOG" --banner "$BANNER" --hold "$HOLD" -- "$@" \
    || { echo "[X] 起窗失败" >&2; exit 3; }

if [ "$MID" -gt 0 ]; then
    sleep "$MID"
    echo ">>> 运行中截图（$MID 秒）"
    "$PY" "$HERE/vis.py" shot "${OUT%.png}_running.png" 2>&1 | tail -1
fi

echo ">>> 等命令结束（哨兵 $LOG.done，最多 ${TIMEOUT}s）……"
waited=0
while [ "$waited" -lt "$TIMEOUT" ]; do
    [ -f "$LOG.done" ] && { echo ">>> 命令已结束（等待 ${waited}s）"; break; }
    sleep 2
    waited=$((waited + 2))
done
if [ "$waited" -ge "$TIMEOUT" ]; then
    echo "[!] 等超时（${TIMEOUT}s）—— 命令可能还在跑，仍然截当前画面" >&2
fi
sleep 1

echo ">>> 结果截图: $OUT"
# 重试 3 次：实测偶发一次 shot 报"窗口不存在"（窗口其实还在），
# 不重试的话就会白跑一趟整个流程，而失败信息混在长输出里很容易被漏看。
rc=1
for i in 1 2 3; do
    outtxt=$("$PY" "$HERE/vis.py" shot "$OUT" 2>&1)
    rc=$?
    echo "$outtxt" | tail -1
    [ "$rc" -eq 0 ] && break
    echo "    （第 ${i} 次截图未成功，3 秒后重试）" >&2
    sleep 3
done
if [ "$rc" -ne 0 ]; then
    echo "[X] 结果截图连续 3 次失败。运行中那张仍保留：${OUT%.png}_running.png" >&2
fi
echo ">>> 日志末尾 12 行："
tail -n 12 "$LOG"
exit "$rc"
