#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""常驻 watchdog：把云端长任务的状态按固定节奏汇报出来。

为什么需要这一层
    `long-task-monitor` 技能的 `--batch-status` 是**单次检查**（跑完即退），
    它自己不循环；技能文档也写明"严格 10 分钟汇报用常驻 watchdog 实现"。
    本脚本就是那个常驻 watchdog：每 `--interval` 秒做一次
        ① 采集云端状态（scripts/cloud/collect_cloud_status.py）
        ② 调技能的 monitor.py --batch-status 出标准汇报
    两者都幂等，进程被杀后直接重启即可，不需要断点恢复（本脚本无状态）。

用法
    python scripts/cloud/watch_cloud_task.py --interval 600
日志
    outputs/monitor/watch.log（每轮一行 + 异常时的详细原因）

★ 报警退出（2026-10-01 新增）
    常驻循环本身**不会**在"链路悄悄死掉 / 主机被关机"时通知任何人，只会安静地空转。
    故本脚本在检测到**致命问题**时主动以非零码退出，借此触发任务通知、把问题顶到人面前：
        3 = 连续 N 轮采集失败（ssh 不可达：实例关机 / 网络中断）
        4 = 连续 N 轮「无任何训练/链路进程，但仍有未完成的臂」= 静默死亡
        5 = 账本余额低于阈值（默认 4 元）——归零会让在跑的臂作废且无法续跑
    退出前会把原因写进 outputs/monitor/ALERT.md。修好后直接重启本脚本即可。
    注意：本脚本**只报警、不自动重启链路**——自动重启会与另一路（每 2 h 值班自动化）
    的重启动作撞车，可能拉起两个 beta_control 写同一份产物。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SKILL_MONITOR = os.path.join(
    os.path.expanduser("~"),
    ".workbuddy", "skills", "long-tasks-monitor__skillhub", "scripts", "monitor.py",
)
RESTART_CMD = (
    "主链路 bash /root/autodl-tmp/chain_restart.sh（幂等，勿直接重跑 ablation_chain.sh）；"
    "β 接力 bash /root/autodl-tmp/beta_control_chain.sh"
)


def log(path: str, msg: str) -> None:
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], log_path: str, label: str) -> int:
    proc = subprocess.run(cmd, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(out if out.endswith("\n") or not out else out + "\n")
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-6:])
        log(log_path, f"[{label}] 非零退出 rc={proc.returncode}：{tail}")
    return proc.returncode


def write_alert(out_dir: str, code: int, title: str, detail: str) -> str:
    fp = os.path.join(out_dir, "ALERT.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(
            f"# ⚠ 云端长任务告警（code {code}）\n\n"
            f"- **时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **结论**：{title}\n\n"
            f"## 详情\n\n{detail}\n\n"
            f"---\n\n"
            f"处理建议：{RESTART_CMD}；若实例已关机，需在 AutoDL 控制台开机"
            f"（关机不会自动恢复，且关机时长仍按规则占用/停止计费）。\n"
        )
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description="云端长任务常驻汇报 watchdog")
    ap.add_argument("--interval", type=int, default=600, help="汇报间隔秒（默认 600=10 分钟）")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT, "outputs", "monitor"))
    ap.add_argument("--skill-monitor", default=DEFAULT_SKILL_MONITOR)
    ap.add_argument("--heartbeat-threshold", type=int, default=900)
    ap.add_argument("--cycles", type=int, default=0, help="0=无限循环")
    ap.add_argument("--collect-fail-limit", type=int, default=3,
                    help="连续采集失败多少轮判为实例不可达并报警退出（默认 3）")
    ap.add_argument("--dead-rounds-limit", type=int, default=2,
                    help="连续多少轮「无进程但仍有未完成臂」判为静默死亡并报警退出（默认 2）")
    ap.add_argument("--min-balance", type=float, default=4.0,
                    help="账本余额低于此值报警退出（元，默认 4.0；0 表示不检查）")
    ap.add_argument("--no-alert-exit", action="store_true",
                    help="关闭报警退出（只汇报，不主动退出）")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "watch.log")
    json_fp = os.path.join(out_dir, "cloud_status.json")
    collector = os.path.join(HERE, "collect_cloud_status.py")
    py = sys.executable

    def fatal(code: int, title: str, detail: str) -> int:
        log(log_path, f"[报警] code={code} {title}")
        fp = write_alert(out_dir, code, title, detail)
        log(log_path, f"[报警] 详情已写入 {fp}；本 watchdog 退出（修好后重启即可）")
        return code

    log(log_path, f"watchdog 启动 pid={os.getpid()} 间隔={args.interval}s "
                  f"阈值={args.heartbeat_threshold}s monitor={args.skill_monitor}")
    n = 0
    fail_streak = 0
    dead_streak = 0
    while True:
        n += 1
        rc = run([py, collector, "--out-dir", out_dir, "--quiet"], log_path, "采集")
        if rc != 0:
            fail_streak += 1
            log(log_path, f"[采集] 失败（可能是实例关机/网络中断）——本轮跳过汇报"
                          f"（连续 {fail_streak}/{args.collect_fail_limit}）")
            if not args.no_alert_exit and fail_streak >= args.collect_fail_limit:
                return fatal(
                    3,
                    f"连续 {fail_streak} 轮采集失败 —— 实例不可达",
                    f"连续 {fail_streak} 轮 ssh 采集失败（间隔 {args.interval}s ≈ "
                    f"{fail_streak * args.interval // 60} 分钟）。\n\n"
                    f"最可能是 **实例已关机**（AutoDL 关机后网关端口转发撤销，表现为 "
                    f"`Connection refused`）或网络中断。\n"
                    f"判据：ping 通 + ssh refused = 实例已关机（关机无法远程开机）。",
                )
        elif not os.path.exists(json_fp):
            log(log_path, "[汇报] 缺少状态文件，跳过")
        else:
            fail_streak = 0
            if not os.path.exists(args.skill_monitor):
                log(log_path, f"[汇报] 找不到技能脚本 {args.skill_monitor}，跳过")
            else:
                cmd = [
                    py, args.skill_monitor,
                    "--batch-status", json_fp,
                    "--report-dir", out_dir,
                    "--heartbeat-threshold", str(args.heartbeat_threshold),
                    "--progress-glob", "{dir}/progress_*.json",
                    "--dir-field", "item_dir",
                    "--restart-cmd", RESTART_CMD,
                ]
                brc = run(cmd, log_path, "汇报")
                log(log_path, f"[汇报] 第 {n} 轮完成 rc={brc}"
                              f"（0=正常/处理中，1=有失败，2=疑似中断）")

            # ---- 状态机：静默死亡 / 余额见底 ----
            try:
                with open(json_fp, encoding="utf-8") as f:
                    st = json.load(f)
                ctx = st.get("_context", {})
                vids = st.get("videos", {})
                unfinished = [a for a, v in vids.items() if v.get("status") != "done"]
                procs = ctx.get("procs") or []
                if args.no_alert_exit:
                    pass
                elif unfinished and not procs:
                    dead_streak += 1
                    log(log_path, f"[体检] 无任何训练/链路进程，但有 {len(unfinished)} 个臂未完成"
                                  f"（连续 {dead_streak}/{args.dead_rounds_limit}）：{unfinished}")
                    if dead_streak >= args.dead_rounds_limit:
                        return fatal(
                            4,
                            f"连续 {dead_streak} 轮无进程但仍有 {len(unfinished)} 个臂未完成 —— 链路静默死亡",
                            f"未完成的臂：`{', '.join(unfinished)}`\n\n"
                            f"主链路 phase=`{ctx.get('chain_phase')}` done=`{ctx.get('chain_done')}`；"
                            f"β 接力 phase=`{ctx.get('beta_phase')}` done=`{ctx.get('beta_done')}`\n\n"
                            f"远端进程列表为空 ⇒ 训练与两条链路都不在了。典型原因：\n"
                            f"① 守夜脚本误判「卡死/全部结束」执行了关机（10-01 已发生 2 次）；\n"
                            f"② 链路脚本自身崩溃退出。\n"
                            f"注意区分「真结束」与「假结束」：只有所有臂 status=done 才算真结束。",
                        )
                else:
                    dead_streak = 0
                bal = ctx.get("balance_ledger_cny")
                try:
                    balf = float(bal)
                except (TypeError, ValueError):
                    balf = None
                if (not args.no_alert_exit) and args.min_balance > 0 and balf is not None \
                        and balf < args.min_balance:
                    return fatal(
                        5,
                        f"账本余额 {balf:.2f} 元 < 阈值 {args.min_balance:.2f} 元 —— 有断供风险",
                        f"账本 = `{ctx.get('balance_ledger_ts')}` 写入的 `{bal}` 元"
                        f"（账本由每 2 h 值班自动化读真实余额刷新）。\n\n"
                        f"余额归零 → AutoDL 自动关机 → 在跑的臂作废"
                        f"（`*_last.pt` 不带优化器状态，不能续跑，只能整臂重跑）。",
                    )
            except (OSError, ValueError) as exc:  # noqa: BLE001
                log(log_path, f"[体检] 状态文件解析失败，跳过体检：{exc}")

        if args.cycles and n >= args.cycles:
            log(log_path, "达到 --cycles 上限，退出")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
