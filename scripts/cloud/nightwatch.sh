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
#   ① 训练主进程在   -> 看训练日志新鲜度（**主链路 + β 链路取最新**）：> 120 分钟没写 -> 判定卡死，关机
#   ② 只有链路在     -> 看「链路日志/心跳/主日志/β 日志」里最新的一个：
#                       > 40 分钟没有动静 -> 判定链路卡死，关机
#                       （链路每次 say() 都会刷心跳，组间间隔只有几秒）
#   ③ 两者都不在     -> 连续 3 次（15 分钟）确认 -> 关机（数据盘保留）
#
# v3.1（2026-10-01）：①与②的证据集加入 β 训练日志 glob $BLOGS（见下方说明）。
#   未修前：主链路一收尾，`.current_train_log` 就定格 ⇒ 120 分钟后守夜会**把正常跑的 β 训练杀掉**。
#
# v3.4（2026-10-05）：★ 关机前做「收尾体检」，区分「正常跑完」与「崩溃退出」。
#   实证（2026-10-04 13:36:32）：`improved_lite` 在 `trainer.py:469 _check_beta_window`
#   抛 RuntimeError 后进程 2 分钟就消失，守夜走路径③ 15 分钟后关机 ——
#   **表面现象与「正常收官」逐字相同**，人工不读日志就完全发现不了（静默失败）。
#   本版在 path③ 关机前扫一遍训练日志尾部：
#     含「[train] 完成」                -> OK
#     无完成标记但含 Traceback/Error    -> CRASH，写入 $ALERT 并在日志里打 [!!]
#     两者都无                          -> UNKNOWN（需人工看）
#   结论同时落到 $OUTCOME（一行：状态 时间 日志路径），下次开机启动横幅会回显。
#   ⚠ 本版**不改变关机时机**（照旧 3×300s），只保证失败不会被吞掉。
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
# ★ v3.2（2026-10-02）：PAT_MAIN 补上 `scripts/train.py`。
#   原版只认 `src.evaluation.ablation|beta_control`（链路内部调用训练器的形态），
#   而 `cloud_autodl.sh train` 与手工起跑走的是**裸的 `scripts/train.py`**
#   ⇒ 守夜看不见主进程 ⇒ 连续 3 次（15 min）判定"训练与链路均已结束"→ 关机，
#   **把正在正常训练的任务砍掉**。这是与 v3.1 同源的"半截匹配"类缺陷（进程 PAT 漏认）。
#   实证：2026-10-02 18:21 起跑的 lite 闸门（`scripts/train.py --config gate_lite_trainability.yaml`）
#   在修前会被误关机。
PAT_MAIN='(src\.evaluation\.(ablation|beta_control)|scripts/train\.py)'
# ★ v3.3（2026-10-03）：PAT_CHAIN 补 `lite_chain.sh`。improved 对链结束后
#   由 lite_chain.sh 接力跑 lite 档；它不在旧 PAT 里 ⇒ 等待期会被误判
#   "全部结束"而关机。
PAT_CHAIN='(ablation_chain|beta_control_chain|lite_chain)\.sh'
# ★ v3.1（2026-10-01 18:07 实证）：「卡死」判据必须把 **β 链路的训练日志**也算进去。
#   原版只读 `.current_train_log`，而那个文件**只有主消融链路会写**；β 链路跑的是
#   `src.evaluation.beta_control`，日志落在 outputs/ablation_logs/beta_*_s082_*.log。
#   于是主链路 16:05 收尾后，守夜拿那份旧日志的 mtime 当"训练日志新鲜度"：
#   16:05 + 120min = 18:05 ⇒ 18:07:55 第一次越过阈值，**把正在正常训练的 β 进程当卡死关机**
#   （当时 β 日志 18:07:51 还在写 iter 2200/2875）。即"进程 PAT 已认 β、日志 PAT 没认"的半截修复。
#   注意：下面 glob 不要加引号，让它在此处展开；无匹配时保持字面量，newest_age 里 `-f` 会跳过。
BLOGS=/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_logs/beta_*.log

# ★ v3.4（2026-10-05）收尾体检相关
ALERT=/root/autodl-tmp/ALERT_TRAIN_CRASH.txt    # 崩溃时落此文件（含日志尾部 60 行）
OUTCOME=/root/autodl-tmp/.last_run_outcome      # 一行：<OK|CRASH|UNKNOWN> <时间> <日志路径>

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

# ---- v3.4 收尾体检 ----------------------------------------------------------
pick_recent_log(){   # 输出「最可能代表刚刚结束那次训练」的日志路径（可能为空）
  local cur age
  cur=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null || true)
  if [ -n "$cur" ] && [ -f "$cur" ]; then
    age=$(newest_age "$cur")
    # 24h 内写过 ⇒ 认为它仍代表刚跑完的那一次（含 bare `scripts/train.py` 起的臂）
    if [ "$age" -ge 0 ] && [ "$age" -lt 86400 ]; then echo "$cur"; return; fi
  fi
  # 兜底：/root/autodl-tmp 下 mtime 最新的 .log
  ls -t /root/autodl-tmp/*.log 2>/dev/null | head -1
}

check_run_outcome(){   # 0=OK  1=CRASH  2=UNKNOWN；同时写 $OUTCOME / $ALERT
  local L tail_txt
  L=$(pick_recent_log)
  if [ -z "$L" ] || [ ! -f "$L" ]; then
    say "[?] 收尾体检：找不到可判读的训练日志 -> UNKNOWN"
    echo "UNKNOWN $(date '+%F %T') (no-log)" > "$OUTCOME"
    return 2
  fi
  tail_txt=$(tail -60 "$L" 2>/dev/null)
  # ★ 必须 -F：ERE 里 `[train]` 会被当字符类，静默失配
  if printf '%s\n' "$tail_txt" | grep -qF '[train] 完成'; then
    say "  收尾体检：$(basename "$L") 尾部含「[train] 完成」-> 正常收官 OK"
    echo "OK $(date '+%F %T') $L" > "$OUTCOME"
    return 0
  fi
  if printf '%s\n' "$tail_txt" | grep -qE 'Traceback|RuntimeError|Error:|Exception'; then
    {
      echo "== 训练异常退出（nightwatch v3.4 收尾体检）=="
      echo "时间：$(date '+%F %T')"
      echo "日志：$L"
      echo "判据：日志尾部无「[train] 完成」，但含异常关键字"
      echo "提示：云端此刻已按 3x300s 规则关机；本文件留存待人工读。"
      echo "---- 尾部 60 行 ----"
      printf '%s\n' "$tail_txt"
    } > "$ALERT"
    say "[!!] 收尾体检：$(basename "$L") 无「[train] 完成」且含异常 -> 判定 CRASH，已写 $ALERT"
    echo "CRASH $(date '+%F %T') $L" > "$OUTCOME"
    return 1
  fi
  say "[?] 收尾体检：$(basename "$L") 无完成标记、也无异常关键字 -> UNKNOWN，需人工看"
  echo "UNKNOWN $(date '+%F %T') $L" > "$OUTCOME"
  return 2
}

say "nightwatch v3.4 启动 PID=$$ 间隔=${INTERVAL}s 主日志卡死=${STALL_MIN}min 链路卡死=${CHAIN_STALL_MIN}min"
say "  活性检测：主进程='$PAT_MAIN'  链路='$PAT_CHAIN'"
say "  日志证据集：$(cat /root/autodl-tmp/.current_train_log 2>/dev/null) + $BLOGS"
[ -f "$OUTCOME" ] && say "  上次收尾结论：$(cat "$OUTCOME" 2>/dev/null)"
[ -f "$ALERT" ]   && say "  [!!] 存在未处理告警文件 $ALERT（上次训练异常退出）"
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
    age=$(newest_age "$L" $BLOGS)
    if [ "$age" -gt $(( STALL_MIN * 60 )) ]; then
      say "[!] 训练进程存活但所有训练日志（主 + β）均已 $((age/60)) 分钟未更新 -> 判定卡死，执行关机"
      sync; /usr/bin/shutdown now; exit 0
    fi
    sleep "$INTERVAL"; continue
  fi

  if [ "$chain" = 1 ]; then
    miss=0
    L=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null || true)
    age=$(newest_age "$CLOG" "$HB" "$L" $BLOGS)
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
    check_run_outcome || true      # ★ v3.4：崩溃信息落盘；不改关机时机
    sync; /usr/bin/shutdown now; exit 0
  fi
  sleep "$INTERVAL"
done
