#!/usr/bin/env bash
# ============================================================================
#  cloud_live.sh —— 云端「实时看板」，给本机那个可见控制台窗口用的。
#
#  它**永不退出**（while true），这样本机窗口不会自己关掉；
#  每 INTERVAL 秒重绘一屏，显示：时间 / GPU 占用 / 训练进程 / 最新日志末尾。
#
#  为什么要有它：
#   ssh 会话一旦结束，本机那个窗口就关闭，截图就抓不到东西了。
#   改成"循环刷新状态"，窗口就变成一块**持续在线的云端仪表盘**，
#   任何时候截图都能反映当下真实状态，而不是历史残影。
#
#  用法（在云端）：
#      bash scripts/cloud_live.sh                 # 默认 12 秒刷新
#      INTERVAL=5 bash scripts/cloud_live.sh      # 快一点
# ============================================================================
set -uo pipefail

PROJ="${PROJ:-/root/autodl-tmp/vibnet-forgery-detector}"
INTERVAL="${INTERVAL:-12}"
TRAIN_LOG="$PROJ/outputs/gpu_train_log.txt"

# ANSI 清屏+回家。比外部 clear 命令可靠（不依赖 terminfo）。
CLS=$'\033[2J\033[H'
B=$'\033[1m'; C=$'\033[1;36m'; G=$'\033[1;32m'; Y=$'\033[1;33m'; R=$'\033[1;31m'; N=$'\033[0m'

# 找出最近被写过的训练日志（不同 tag 的输出目录不一样，不能写死一个路径）
newest_log() {
    find "$PROJ/outputs" -maxdepth 3 -type f \
        \( -name '*.log' -o -name '*train*.txt' -o -name 'gpu_train_log.txt' \) \
        -mmin -60 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1
}

while true; do
    printf '%s' "$CLS"
    printf '%s\n' "${C}================================================================================${N}"
    printf '  %sVIB-Net 云端实时看板%s   （本窗口由本机经 SSH 直连，内容全部来自云端）\n' "$B" "$N"
    printf '%s\n' "${C}================================================================================${N}"
    printf '  云端时间 : %s     主机 : %s\n' "$(date '+%F %T')" "$(hostname)"
    printf '  项目目录 : %s\n' "$PROJ"

    # ---- GPU ----
    printf '\n%s[GPU]%s\n' "$Y" "$N"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader 2>/dev/null \
            | awk -F', ' '{printf "  型号=%s  利用率=%s  显存=%s / %s  温度=%s\n",$1,$2,$3,$4,$5}'
    else
        printf '  %s看不到 GPU（实例可能处于无卡模式）%s\n' "$R" "$N"
    fi

    # ---- 训练进程 ----
    # ★ 必须覆盖两种入口：scripts/train.py（正式训练）
    #   和 src.evaluation.ablation（消融训练）。
    #   只写前者时，消融跑到一半看板会显示"当前没有训练进程在跑" —— 纯误导。
    printf '\n%s[进程]%s\n' "$Y" "$N"
    TRAIN_PROC=$(pgrep -af "scripts/train\.py|src\.evaluation\.ablation|src/engine/trainer\.py" 2>/dev/null)
    if [ -n "$TRAIN_PROC" ]; then
        printf '%s\n' "$TRAIN_PROC" | head -3 | sed 's/^/  /' | cut -c1-150
    else
        printf '  当前没有训练进程在跑\n'
    fi

    # ---- 磁盘 ----
    printf '\n%s[磁盘]%s\n' "$Y" "$N"
    df -h /root/autodl-tmp 2>/dev/null | tail -1 | awk '{printf "  数据盘 %s 已用 %s/%s (%s)\n",$6,$3,$2,$5}'

    # ---- 最新日志 ----
    L="$(newest_log)"
    [ -z "$L" ] && [ -f "$TRAIN_LOG" ] && L="$TRAIN_LOG"
    printf '\n%s[最新日志]%s\n' "$Y" "$N"
    if [ -n "$L" ] && [ -f "$L" ]; then
        printf '  %s（%s，%s 前更新）%s\n' "$G" "$L" \
            "$(date -d "@$(( $(date +%s) - $(stat -c %Y "$L") ))" -u '+%H:%M:%S' 2>/dev/null || echo '?')" "$N"
        # ★ 关键进度行单独提出来置顶：else 会被下面的噪声挤出可见区。
        #   实测：数据加载器每张 PNG 都打一行 `libpng warning: iCCP ...`，
        #   屏幕只剩 12 行时，尾部全是警告，真正的 epoch/iter 进度一行都看不到。
        PL="$(grep 'epoch ' "$L" 2>/dev/null | tail -1)"
        [ -z "$PL" ] && PL="$(grep 'iter ' "$L" 2>/dev/null | tail -1)"
        [ -z "$PL" ] && PL="（还没产生进度行）"
        printf '  %s[关键]%s %s\n' "$G" "$N" "$(printf '%s' "$PL" | cut -c1-130)"
        # 行宽与行数都要压住：窗口只有 150x44，超出会折行把顶部横幅顶出屏幕。
        # 过滤纯噪声行（libpng/ICC），让这 11 行留给真正的日志。
        grep -v -e 'libpng warning' -e ': iCCP:' -e 'profile tag start not a multiple' "$L" 2>/dev/null \
            | tail -n 11 | cut -c1-134 | sed 's/^/  /'
    else
        printf '  还没找到近期活跃的日志文件\n'
    fi

    printf '\n%s（每 %s 秒刷新 · 关闭本窗口不影响云端训练）%s\n' "$G" "$INTERVAL" "$N"
    sleep "$INTERVAL"
done
