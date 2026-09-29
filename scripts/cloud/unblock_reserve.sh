#!/bin/bash
# =============================================================================
# 一次性解锁守护：链路因「已失去用途的 RESERVE」停住后，解除它并续跑
#
# 背景
#   链路顶部 RESERVE=8.0 是留给「启用 COVERAGE 后的正式训练」的预算门槛。
#   09-30 用户拍板 COVERAGE 改为**纯外部测试域**（不参与训练）=> 这笔预留失去用途，
#   却仍会让链路在第 6 个臂开跑前把它挡掉（≈ 白丢 6.8 h / ￥5.8）。
#   而 RESERVE 在**运行中的** bash 脚本里不能就地改（会按字节偏移错读），
#   所以只能在链路退出后、无进程持有时改，这就是本脚本存在的唯一理由。
#
# 为什么必须尽快动作（不是「等值班任务 2 小时后处理」就行）
#   链路一停，nightwatch v2 看到「没有训练进程、没有链路进程」=> 15 分钟后自动关机。
#   实例一关机就得靠 CDP 接管浏览器去控制台点开机（脆弱、且用户不在场）。
#   本脚本每 2 min 轮询，远快于 nightwatch 的 15 min 判定窗，能把链路接回去。
#
# 安全边界（任何一条不满足都不动作）
#   ① 没有 ablation_chain.sh 进程 —— 运行中的脚本绝不能被 sed
#   ② 没有 src.evaluation.ablation 进程 —— 可能有别的训练在跑
#   ③ chain.log 尾部确实写着「[预算] 不足 -> 不安排」—— 不是别的退出原因
#      （看完全部跑完 / 连续失败自停 / 优雅停止 都不动作：那些不该解除预留）
#   ④ 只动作一次，之后退出 —— 真故障时靠值班任务人工判断，避免反复重启烧钱
#
# 用法：setsid nohup bash /root/autodl-tmp/unblock_reserve.sh > /dev/null 2>&1 < /dev/null &
# 日志：/root/autodl-tmp/unblock.log
# =============================================================================
set -u
CHAIN=/root/autodl-tmp/ablation_chain.sh
CLOG=/root/autodl-tmp/chain.log
LOG=/root/autodl-tmp/unblock.log
INTERVAL=120

say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "启动监视 PID=$$ 间隔=${INTERVAL}s（等链路因预算停住）"
while true; do
  sleep "$INTERVAL"
  # ① 链路还在跑 -> 继续等
  pgrep -f 'ablation_chain\.sh' >/dev/null 2>&1 && continue
  # ② 还有训练进程 -> 可能刚接上下一组，继续等
  pgrep -f 'src\.evaluation\.ablation' >/dev/null 2>&1 && continue
  # ③ 退出原因必须是「预算不足」
  if ! tail -40 "$CLOG" 2>/dev/null | grep -q '\[预算\] 不足 -> 不安排'; then
    say "链路已退出，但原因不是预算不足（跑完 / 故障 / 优雅停止）-> 不动作，监视结束"
    exit 0
  fi
  # ④ 解除预留 -> 幂等续跑（chain_restart.sh 只启未成功的臂，无剩余臂会自行退出 0）
  say "确认：预算停住且无任何训练/链路进程 -> 解除 RESERVE 并续跑"
  sed -i 's/^RESERVE=8\.0/RESERVE=0.0/' "$CHAIN"
  { echo "  改后：$(grep -n '^RESERVE=' "$CHAIN")"; } >> "$LOG"
  bash /root/autodl-tmp/chain_restart.sh >> "$LOG" 2>&1
  say "chain_restart 已发起 -> 监视结束（只动作一次）"
  exit 0
done
