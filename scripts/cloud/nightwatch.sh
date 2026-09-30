#!/bin/bash
# =============================================================================
# 守夜脚本 v3 —— 适配「消融链路 + β 对照链路」，避免在组间把机器关掉
#
# v3（2026-09-30）：新增第二条链路 `beta_control_chain.sh`（β 生效版受控对照实验）。
#   它会在主消融链路结束后自动接力，中途还有「等主链路」的长等待期。
#   若 PAT 只认 ablation_chain.sh，主链路一结束守夜就会判定"全部结束"→ 15 分钟后
#   关机，把刚接上的 β 实验直接砍掉。故 PAT 改为同时匹配两条链路，
#   主进程 PAT 也要认得 `src.evaluation.beta_control`（它不叫 ablation）。
#
# v1 只看 `src.evaluation.ablation`：链路的组与组之间训练进程会短暂消失
# （归档、跨生成器评测、预算判断），v1 会把这当成"训练结束"→ 15 分钟后关机，
# 把还没跑完的臂全砍掉。v2 把链路进程也算"活着"。
#
# 三态判定
#   ① 训练主进程在   -> 看主日志新鲜度：> 120 分钟没写 -> 判定卡死，关机
#   ② 只有链路在     -> 看「链路日志/心跳/主日志」三者里最新的一个：
#                       > 40 分钟没有动静 -> 判定链路卡死，关机
#                       （链路每次 say() 都会刷心跳，组间间隔只有几秒）
#   ③ 两者都不在     -> 连续 3 次（15 分钟）确认 -> 关机（数据盘保留）
#
# 取消：touch /root/autodl-tmp/nightwatch.off
# 日志：/root/autodl-tmp/nightwatch.log（只在有事件时写；"只有启动行"= 正常）
# =============================================================================
LOG=/root/autodl-tmp/nightwatch.log
OFF=/root/autodl-tmp/nightwatch.off
HB=/root/autodl-tmp/chain.heartbeat
CLOG=/root/autodl-tmp/chain.log
INTERVAL=300
NEED_MISS=3
STALL_MIN=120          # 主进程活着但主日志这么久没动 -> 卡死
CHAIN_STALL_MIN=40     # 只有链路在，但所有证据都这么久没动 -> 链路卡死
PAT_MAIN='src\.evaluation\.(ablation|beta_control)'
PAT_CHAIN='(ablation_chain|beta_control_chain)\.sh'

say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

newest_age(){   # 入参为文件列表；输出"最新的那个文件距今多少秒"；全不存在输出 -1
  local newest=0 f t
  for f in "$@"; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$t" -gt "$newest" ] && newest=$t
  done
  [ "$newest" -eq 0 ] && { echo -1; return; }
  echo $(( $(date +%s) - newest ))
}

say "nightwatch v3 启动 PID=$$ 间隔=${INTERVAL}s 主日志卡死=${STALL_MIN}min 链路卡死=${CHAIN_STALL_MIN}min"
say "  活性检测：主进程='$PAT_MAIN'  链路='$PAT_CHAIN'"
miss=0
while true; do
  if [ -e "$OFF" ]; then say "检测到 $OFF -> 主动退出，不关机"; exit 0; fi

  main=0; chain=0
  pgrep -f "$PAT_MAIN"  >/dev/null 2>&1 && main=1
  pgrep -f "$PAT_CHAIN" >/dev/null 2>&1 && chain=1

  if [ "$main" = 1 ]; then
    [ "$miss" -ne 0 ] && say "训练进程恢复，miss 计数清零（$miss -> 0）"
    miss=0
    L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null || true)
    age=$(newest_age "$L")
    if [ "$age" -gt $(( STALL_MIN * 60 )) ]; then
      say "[!] 训练进程存活但主日志已 $((age/60)) 分钟未更新 -> 判定卡死，执行关机"
      sync; /usr/bin/shutdown now; exit 0
    fi
    sleep "$INTERVAL"; continue
  fi

  if [ "$chain" = 1 ]; then
    miss=0
    L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null || true)
    age=$(newest_age "$CLOG" "$HB" "$L")
    if [ "$age" -lt 0 ] || [ "$age" -gt $(( CHAIN_STALL_MIN * 60 )) ]; then
      say "[!] 链路进程存活但 $((age/60)) 分钟无任何动静 -> 判定链路卡死，执行关机"
      sync; /usr/bin/shutdown now; exit 0
    fi
    sleep "$INTERVAL"; continue
  fi

  miss=$((miss+1))
  say "未发现训练进程与链路进程（第 $miss/$NEED_MISS 次）"
  if [ "$miss" -ge "$NEED_MISS" ]; then
    say "判定训练与链路均已结束 -> 执行关机（数据盘保留，开机即恢复）"
    sync; /usr/bin/shutdown now; exit 0
  fi
  sleep "$INTERVAL"
done
