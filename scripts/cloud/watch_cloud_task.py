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
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description="云端长任务常驻汇报 watchdog")
    ap.add_argument("--interval", type=int, default=600, help="汇报间隔秒（默认 600=10 分钟）")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT, "outputs", "monitor"))
    ap.add_argument("--skill-monitor", default=DEFAULT_SKILL_MONITOR)
    ap.add_argument("--heartbeat-threshold", type=int, default=900)
    ap.add_argument("--cycles", type=int, default=0, help="0=无限循环")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "watch.log")
    json_fp = os.path.join(out_dir, "cloud_status.json")
    collector = os.path.join(HERE, "collect_cloud_status.py")
    py = sys.executable

    log(log_path, f"watchdog 启动 pid={os.getpid()} 间隔={args.interval}s "
                  f"阈值={args.heartbeat_threshold}s monitor={args.skill_monitor}")
    n = 0
    while True:
        n += 1
        rc = run([py, collector, "--out-dir", out_dir, "--quiet"], log_path, "采集")
        if rc != 0:
            log(log_path, "[采集] 失败（可能是实例关机/网络中断）——本轮跳过汇报")
        elif not os.path.exists(json_fp):
            log(log_path, "[汇报] 缺少状态文件，跳过")
        else:
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
        if args.cycles and n >= args.cycles:
            log(log_path, "达到 --cycles 上限，退出")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
