#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键查看后台长任务是否还活着。

为什么需要它
------------
单独看某个日志文件"没动"很容易误判：可能是任务真的死了，也可能只是
日志缓冲没刷出来、或者它正卡在某一步。所以这里同时看三个互相独立的证据：

  1. 进程表里还有没有这个脚本的进程（最硬的证据）
  2. 日志文件最后被写入的时间（距现在多久）
  3. 日志里的进度数字有没有推进（下载百分比 / 训练 iter）

三者一致才判定"在跑"；只有进程在、但日志长时间不动 → 判定"疑似卡死"。

用法
----
    python scripts/check_status.py            # 看全部
    python scripts/check_status.py --watch    # 每 30 秒刷新一次
    python scripts/check_status.py --watch --interval 10
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# 我们关心的长任务：脚本名 -> (说明, 日志文件, 进度解析函数名)
# --------------------------------------------------------------------------
def _parse_fetch(text: str) -> str:
    """从下载日志里取最后一条进度（日志用 \\r 刷新，需先按 \\r 切开）。"""
    tail = text[-20000:]
    lines = [s.strip() for s in tail.replace("\r", "\n").splitlines() if s.strip()]
    # 形态 1：大文件逐字节下载（fetch_datasets.py）  例： 12.3%  1.2 GB/10.0 GB  8.0 MB/s
    prog = [s for s in lines if re.search(r"\d+\.\d+%\s+[\d.]+ \w+/[\d.]+ \w+", s)]
    if prog:
        return prog[-1]
    # 形态 2：多线程逐文件下载（fetch_tamper_datasets.py）  例： 已完成 400/17737 ...
    prog2 = [s for s in lines if re.search(r"已完成\s+\d+/\d+", s)]
    if prog2:
        return prog2[-1]
    # 可能正在解压 / 切换文件，退回显示最后一行有信息量的内容
    for s in reversed(lines):
        if s.startswith(("[get", "[resume", "[skip", "[extract", "[done", "[磁盘",
                         "[err", "[FAIL", "[retry", "  完成：", "[1/", "[2/")):
            return s
    return "(还没有可解析的进度)"


def _parse_train(text: str) -> str:
    """给出最有信息量的两行：当前是哪个 epoch + 走到第几个 iter。

    ⚠ 必须按**出现先后**判断新旧，不能简单「有 iter 就用 iter」：
    每个 epoch 的开头是 `[stage...] epoch N/M ...` 行，第一次 `iter 50/...`
    要到 ~5 秒后才打。那个窗口里若直接用 last_iter，会拿上一轮的最后一条
    （如 iter 2850/2875）当进度 —— 看起来像上一轮还没跑完。
    """
    lines = [s.rstrip() for s in text.splitlines() if s.strip()]
    last_epoch = last_epoch_at = None
    last_iter = last_iter_at = None
    for i, s in enumerate(lines):
        if "epoch" in s and "val:" not in s and "global" in s:
            last_epoch, last_epoch_at = s, i
        elif re.search(r"iter \d+/\d+", s):
            last_iter, last_iter_at = s, i
    out = []
    if last_epoch:
        out.append(last_epoch)
    if last_iter and (last_epoch_at is None or last_iter_at > last_epoch_at):
        out.append(last_iter)
    return "\n         ".join(out) if out else "(还没有可解析的进度)"


def _parse_tail(text: str) -> str:
    """等待型任务（如 finalize_ablation.py）没有进度数字，直接给最后一行。"""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    return lines[-1] if lines else "(还没有输出)"


_ABLATION_LOGS = [
    "outputs/ablation_pretrained/train_log.txt",
    "outputs/ablation_random/train_log.txt",
    "outputs/ablation_pretrained/train_stdout.txt",
    "outputs/ablation_random/train_stdout.txt",
    "outputs/ablation_pretrained/cross_gen_stdout.txt",
    "outputs/ablation_random/cross_gen_stdout.txt",
]

TASKS = {
    "fetch_datasets.py": ("数据集下载(ForenSynths)",
                          ["outputs/fetch_train_log.txt",
                           "outputs/fetch_full_log.txt",
                           "outputs/fetch_val_test_log.txt"], _parse_fetch),
    "fetch_tamper_datasets.py": ("篡改数据集下载(CASIA/COVERAGE)",
                                 ["outputs/fetch_casia_log.txt"], _parse_fetch),
    # 对照实验两臂是 run_ablation.py 拉起的子进程 scripts/train.py，
    # 所以两个入口共用同一组日志候选（取最新那个，避免误报停滞）。
    # 云上（cloud_autodl.sh train 阶段）把 stdout 重定向到 outputs/gpu_train_log.txt，
    # 本地跑则写 outputs/run_default/train_log.txt（cfg: project.output_dir）。
    # 这两个都在候选里 —— 漏掉它们会让"训练正在跑"被误报成"日志文件不存在"。
    "train.py": ("模型训练(含对照实验两臂)",
                 ["outputs/gpu_train_log.txt",
                  "outputs/run_default/train_log.txt",
                  "outputs/realval_train_log.txt", *_ABLATION_LOGS], _parse_train),
    "run_ablation.py": ("对照实验驱动",
                        ["outputs/run_ablation_stdout.txt", *_ABLATION_LOGS],
                        _parse_train),
    "finalize_ablation.py": ("对照实验收尾（等待中）",
                             ["outputs/finalize_log.txt"], _parse_tail),
}

# 判定"日志停滞"的阈值（秒）。超过这个时间没有任何写入，就提示可能要看看。
STALE_AFTER = 300


def _decode(raw) -> str:
    """把子进程输出解成字符串。

    ⚠ 绝不用 subprocess.run(..., text=True)：那会按 locale 解码，中文 Windows 下
    走 GBK，一旦输出里有非 UTF-8 字节，Popen 的**读取子线程**会抛
    UnicodeDecodeError —— 异常不冒泡，主线程拿到的 stdout 直接是 None
    （表现成 "'NoneType' has no attribute 'splitlines'"）。
    → 一律取字节自己解：先 UTF-8，失败退 GBK（Linux 的 ps 输出是 UTF-8）。
    """
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def _procs_wmic():
    """Windows：wmic 能给出完整命令行。拿不到就返回 None，交给 ps 兜底。"""
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "ProcessId,WorkingSetSize,CommandLine", "/format:csv"],
            capture_output=True, timeout=30,
        )
    except Exception:                            # noqa: BLE001
        return None                              # 新版 Windows 已移除 wmic
    out = _decode(r.stdout)
    if not out.strip():
        return None

    # /format:csv 的列序是 Node,CommandLine,ProcessId,WorkingSetSize
    # （Node 固定在最前，其余属性按字母序排列）——别把后两列取反了。
    procs = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("Node,"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        cmd = ",".join(parts[1:-2]).strip()
        try:
            pid = int(parts[-2] or 0)
            ws = int(parts[-1] or 0)
        except ValueError:
            continue
        if cmd:
            procs.append((pid, ws / 1024 / 1024, cmd))
    return procs


def _procs_ps():
    """Linux（AutoDL 容器）/ Git Bash：ps -eo pid=,rss=,args=（rss 单位 KB）。"""
    try:
        r = subprocess.run(["ps", "-eo", "pid=,rss=,args="],
                           capture_output=True, timeout=30)
    except Exception:                            # noqa: BLE001
        return None
    out = _decode(r.stdout)
    procs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)               # 只切两刀，剩下整体是命令行
        if len(parts) < 3:
            continue
        cmd = parts[2].strip()
        if "python" not in cmd:                   # 与 wmic 分支口径一致：只看 python
            continue
        try:
            pid, rss_kb = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        procs.append((pid, rss_kb / 1024, cmd))
    return procs


def running_processes() -> list:
    """返回 [(pid, working_set_mb, cmdline)]，只含 python 进程。

    用 wmic 而不是 tasklist：tasklist 不给命令行，没法区分是哪个脚本。
    wmic 在新版 Windows 上可能被移除 / 容器里根本没装 —— 此时退回 `ps`，
    于是同一个脚本在本地 Windows 和云上 Linux 都能用。
    """
    for fn in (_procs_wmic, _procs_ps):
        procs = fn()
        if procs is not None:
            return procs
    print("  [warn] 无法枚举进程（wmic / ps 都不可用）")
    return []


def fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} 秒前"
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟前"
    return f"{seconds / 3600:.1f} 小时前"


def report(procs: list) -> None:
    w = 72
    print("=" * w)
    print(f" 后台任务状态   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * w)

    for script, (label, logrels, parser) in TASKS.items():
        # --- 证据 1：进程 ---
        hits = [(pid, ws) for pid, ws, cmd in procs if script in cmd]
        # 2 个进程属正常：bash 下的 wrapper + 真正的 python 解释器
        if hits:
            pids = ", ".join(str(p) for p, _ in hits)
            ws_max = max(ws for _, ws in hits)
            alive = f"✅ 在跑    进程 PID {pids}（峰值内存 {ws_max:.1f} MB）"
        else:
            alive = "❌ 没找到进程（已结束，或从未启动成功）"

        # --- 证据 2：日志新鲜度 ---
        # 一个脚本可能按用途写多个日志（如下载器分别写 train/full/casia）。
        # 必须取**最新**的那个，否则会在进程正常工作时误报"日志停滞"。
        cands = []
        for rel in logrels:
            p = os.path.join(ROOT, rel)
            if os.path.exists(p):
                cands.append((os.path.getmtime(p), rel, p))
        if cands:
            cands.sort(reverse=True)
            mtime, logrel, logpath = cands[0]
            age = time.time() - mtime
            size_kb = os.path.getsize(logpath) / 1024
            fresh = "✅" if age < STALE_AFTER else "⚠️"
            loginfo = f"{fresh} {logrel}  {size_kb:.0f} KB，最后写入 {fmt_age(age)}"
            others = [c[1] for c in cands[1:]]
            if others:
                loginfo += f"\n         （另有日志：{', '.join(others)}）"
        else:
            loginfo = f"❌ 日志文件不存在：{', '.join(logrels)}"
            age = None
            logpath = None

        # --- 证据 3：进度内容 ---
        if logpath:
            try:
                with open(logpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                prog = parser(text)
            except Exception as e:               # noqa: BLE001
                prog = f"(读取失败：{e})"
        else:
            prog = "-"

        print(f"\n【{label}】{script}")
        print(f"  {alive}")
        print(f"  {loginfo}")
        print(f"  进度 : {prog}")

        # 综合判断
        if hits and age is not None and age >= STALE_AFTER:
            print(f"  ⚠️  进程还在，但日志已 {fmt_age(age)} 没更新 —— "
                  f"可能卡在某个长步骤（如解压），也可能真的挂了。"
                  f"再等一次刷新确认。")
        elif not hits and age is not None and age < STALE_AFTER:
            print("  ℹ️  进程已退出，但日志刚刚还在写 —— 可能刚好正常结束。")

    # 孤儿进程提示（排除本脚本自己）
    known = {os.getpid()}
    for script in TASKS:
        known |= {pid for pid, _, cmd in procs if script in cmd}
    leftovers = [(pid, ws, cmd) for pid, ws, cmd in procs
                 if pid not in known and ws < 80
                 and "check_status.py" not in cmd]
    if leftovers:
        print("\n" + "-" * w)
        print("顺带一提：有 %d 个已结束脚本留下的孤儿进程（内存 <80 MB，无实际工作）："
              % len(leftovers))
        for pid, ws, cmd in leftovers[:8]:
            short = cmd.split("scripts/")[-1] if "scripts/" in cmd else cmd[:60]
            print(f"  PID {pid:>7}  {ws:5.1f} MB  {short}")

    print("\n" + "=" * w)
    if os.name == "nt":
        print(" 提示：本机所有 bash 调用都会打印一行 crashpad 的 "
              "\"CreateFile: 系统找不到指定的文件 (0x2)\"，\n"
              "       那是 Git Bash 包装层的噪音，与脚本无关 —— 连 ls 都会打印它。")


def main() -> int:
    ap = argparse.ArgumentParser(description="查看后台长任务状态")
    ap.add_argument("--watch", action="store_true", help="循环刷新")
    ap.add_argument("--interval", type=int, default=30, help="刷新间隔秒数")
    args = ap.parse_args()

    while True:
        procs = running_processes()
        os.system("cls" if os.name == "nt" else "clear")
        report(procs)
        if not args.watch:
            break
        print(f"\n（{args.interval} 秒后刷新，Ctrl+C 退出）")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n已退出。")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
