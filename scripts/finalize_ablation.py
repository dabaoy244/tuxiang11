#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照实验收尾：等实验跑完 -> 重新出报告 -> 把预训练臂的权重提升为默认权重。

为什么需要它
------------
`scripts/run_ablation.py` 跑完两臂 + 两次逐生成器评测大约要 3~4 小时（CPU）。
中途停掉会留下"报告只覆盖一臂"的半成品，而报告是中期材料的附件 A8。

本脚本做三件事，**幂等、可重复执行**：

1. 等 `run_ablation.py` 进程退出（用 wmic 查命令行，最长等 `--max-wait` 小时）；
2. 调 `run_ablation.py --report-only` 重新汇总 → 刷新
   `docs/14_对照实验_预训练权重消融.md`（拿到完整两臂数字）；
3. 把预训练臂的最佳检查点提升为默认权重 `checkpoints/vibnet_best.pt`。

为什么第 3 步要做
------------------
桌面工具（`app/modules/settings_panel.py`）与各评测脚本默认加载
`checkpoints/vibnet_best.pt`。该文件此前被一次"3 个 batch 的自检"污染过
（见 `docs/08` C19），已隔离到 `checkpoints/_probe/`。
在拿到正式三阶段训练权重之前，**预训练臂的权重是这个项目目前唯一真正训过、
并且证明了"判决边界能建立起来"的模型**，把它放回默认位置比留空更有用，
而且会在同目录写一份 `vibnet_best.README.txt` 说明它的来历与能力上限。

⚠️ 它**不是**最终交付模型（只训了 3 epoch / 1600 张图，AUC 远达不到申报指标），
只是"有一个能跑、且非退化"的权重可用。正式权重请按 `docs/15` 上云训练。

用法
----
    python scripts/finalize_ablation.py                 # 等 + 收尾
    python scripts/finalize_ablation.py --no-wait       # 不等，直接收尾（实验已结束）
    python scripts/finalize_ablation.py --max-wait 6    # 最多等 6 小时
    python scripts/finalize_ablation.py --no-promote    # 只出报告，不动权重
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORT = os.path.join(ROOT, "docs", "14_对照实验_预训练权重消融.md")
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "vibnet_best.pt")
PRETRAINED_CKPT = os.path.join(ROOT, "checkpoints", "abp_best.pt")
RANDOM_CKPT = os.path.join(ROOT, "checkpoints", "abr_best.pt")
NOTE = os.path.join(ROOT, "checkpoints", "vibnet_best.README.txt")


def log(msg: str) -> None:
    print(f"[finalize] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------
# `run_ablation.py` 的 stdout 落在这里；它每完成一个 iter 都会刷新。
# 进程是否存活有两个独立证据：① 进程表里有没有它；② 这份日志近不近。
ABLATION_STDOUT = os.path.join(ROOT, "outputs", "run_ablation_stdout.txt")
STALE_MIN = 25          # 日志超过这么多分钟没更新，视为可能已死


def _run_bytes(cmd: list) -> str | None:
    """执行命令并**按字节**取回输出，再用 errors='replace' 解码。

    必须这么做：中文 Windows 下 `subprocess.run(text=True)` 走 GBK 解码，
    wmic 输出里只要有一个非 UTF-8 字节，**读取线程就会抛 UnicodeDecodeError**，
    `stdout` 被置为 None（异常发生在子线程里，主线程只看到 None）。
    上一版就是因此 `None.splitlines()` 崩掉的 —— 见 `docs/08` C21。
    """
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=30)
    except Exception:                                        # noqa: BLE001
        return None
    if p.stdout is None:
        return None
    return p.stdout.decode("utf-8", errors="replace")


def ablation_pids() -> list:
    """查还在跑的 run_ablation.py 进程号。查不到时返回 []。"""
    out = _run_bytes(["wmic", "process", "where", "name='python.exe'",
                      "get", "ProcessId,CommandLine", "/format:csv"])
    if not out:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("Node,"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        cmd = ",".join(parts[1:-1])
        if "run_ablation.py" in cmd and "finalize_ablation" not in cmd:
            try:
                pids.append(int(parts[-1] or 0))
            except ValueError:
                pass
    return [p for p in pids if p]


def stdout_fresh_min() -> float | None:
    """run_ablation.py 的 stdout 日志多少分钟前更新过；文件不存在返回 None。"""
    if not os.path.exists(ABLATION_STDOUT):
        return None
    return (time.time() - os.path.getmtime(ABLATION_STDOUT)) / 60.0


# `run_ablation.py` 正常结束时一定会打印这一行（见其 main()）。它是比
# 「日志新不新鲜」更强的完成信号 —— 进程刚退出时日志还很"新鲜"，
# 只看 mtime 会误判成"仍在跑"，白等最多 STALE_MIN 分钟。
_DONE_MARKERS = ("总耗时",)


def stdout_says_done() -> bool:
    """日志尾部是否含"已完成"标志。文件可能很大，只读末尾 4 KB。"""
    if not os.path.exists(ABLATION_STDOUT):
        return False
    try:
        size = os.path.getsize(ABLATION_STDOUT)
        with open(ABLATION_STDOUT, "rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(m in tail for m in _DONE_MARKERS)


def still_running() -> tuple:
    """(是否仍在跑, 依据说明)。三个证据依次判定：进程表 → 完成标志 → 日志新鲜度。"""
    pids = ablation_pids()
    if pids:
        return True, f"进程存活 PID {pids}"
    if stdout_says_done():
        return False, "进程表为空，且日志已打印完成标志"
    fresh = stdout_fresh_min()
    if fresh is not None and fresh < STALE_MIN:
        # wmic 不可用（新版 Windows 已弃用）时，靠日志心跳判断
        return True, f"进程表查不到，但日志 {fresh:.1f} 分钟前仍在写"
    if fresh is not None:
        return False, f"进程表为空，且日志已 {fresh:.1f} 分钟无更新"
    return False, "进程表为空，且找不到 stdout 日志"


def wait_for_exit(max_wait_h: float, poll_s: int = 60) -> bool:
    deadline = time.time() + max_wait_h * 3600
    first = True
    while True:
        alive, why = still_running()
        if not alive:
            log(f"对照实验已结束（{why}）——进入收尾。")
            return True
        if time.time() > deadline:
            log(f"⚠️ 等待超过 {max_wait_h} 小时，实验仍在跑（{why}）。"
                f"先按当前已有结果收尾；实验结束后可再跑一次本脚本。")
            return False
        if first:
            log(f"等待对照实验结束（{why}，每 {poll_s} 秒查一次）…")
            first = False
        else:
            log(f"仍在跑（{why}）…")
        time.sleep(poll_s)


# --------------------------------------------------------------------------
def regenerate_report() -> bool:
    log("重新汇总对照实验报告 …")
    rc = subprocess.run(
        [sys.executable, "-u", os.path.join(ROOT, "scripts", "run_ablation.py"),
         "--report-only"],
        cwd=ROOT).returncode
    log(f"汇总退出码 {rc}")
    if os.path.exists(REPORT):
        log(f"报告：{REPORT}（{os.path.getsize(REPORT) / 1024:.1f} KB，"
            f"{time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(REPORT)))} 更新）")
    return rc == 0


def best_auc(arm_out: str) -> float | None:
    """从某臂的 history.json 里取最佳 val_cls_auc。"""
    p = os.path.join(ROOT, arm_out, "history.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            hist = json.load(f)
    except Exception:                                        # noqa: BLE001
        return None
    vals = [h.get("val_cls_auc") for h in hist
            if isinstance(h.get("val_cls_auc"), (int, float))]
    return max(vals) if vals else None


def promote() -> None:
    """把预训练臂的最佳权重提升为默认权重，并留下来历说明。"""
    if not os.path.exists(PRETRAINED_CKPT):
        log(f"⚠️ 找不到 {PRETRAINED_CKPT}，跳过权重提升。")
        return

    auc_pre = best_auc("outputs/ablation_pretrained")
    auc_rnd = best_auc("outputs/ablation_random")
    log(f"最佳 val AUC —— 预训练臂 {auc_pre} / 随机初始化臂 {auc_rnd}")

    if auc_pre is None:
        log("⚠️ 预训练臂没有可用指标，跳过提升（不把来路不明的权重设为默认）。")
        return
    if auc_rnd is not None and auc_pre <= auc_rnd:
        log("⚠️ 预训练臂并不优于随机初始化臂，跳过提升 —— "
            "这种情况需要先排查原因，而不是把权重设为默认。")
        return

    # 备份已有的默认权重（如果有且不是我们刚放的那个）
    if os.path.exists(DEFAULT_CKPT):
        bak = DEFAULT_CKPT + ".bak"
        shutil.copy2(DEFAULT_CKPT, bak)
        log(f"已把原有默认权重备份到 {bak}")

    shutil.copy2(PRETRAINED_CKPT, DEFAULT_CKPT)
    log(f"✅ 已把预训练臂权重提升为默认权重：{DEFAULT_CKPT}")

    with open(NOTE, "w", encoding="utf-8") as f:
        f.write(
            "checkpoints/vibnet_best.pt —— 来历说明（由 scripts/finalize_ablation.py 生成）\n"
            "==========================================================================\n\n"
            f"来源         : {os.path.relpath(PRETRAINED_CKPT, ROOT)}\n"
            "对应实验     : docs/14_对照实验_预训练权重消融.md 的「ImageNet 预训练」臂\n"
            "配置         : configs/ablation_b16_pretrained.yaml\n"
            "骨干         : ViT-B/16（90.53M 参数，FP32 362.1 MB）\n"
            "训练数据     : ForenSynths val 划分 1600 张（ProGAN 20 类保留集）\n"
            "训练轮数     : 3 epoch，batch 8，seed 3407，仅 stage1_cls_pretrain\n"
            f"最佳 val AUC : {auc_pre:.4f}\n\n"
            "== 演示时该用哪个模型 ==\n"
            "  * 演示「判决边界存在」→ 用本文件（torch 后端）。它 tn/tp 都非零，\n"
            "    真图与假图会给出不同判决；\n"
            "  * 演示「体积 ≤120MB、CPU ≥8 张/秒」→ **不要**用本文件，它是 362.1 MB。\n"
            "    改用 lite 导出件（s16 + ONNX 简化：98.7 MB / 9.46~11.22 张每秒），\n"
            "    见 outputs/cpu_benchmark.json 与 docs/09_指标可达性与骨干选型.md；\n"
            "  * 两个演示是两件事，别在同一个话术里混着说。\n\n"
            "== 能力上限（务必知悉）\n"
            "  * 这不是最终交付模型。它只训了 3 epoch / 1600 张图，"
            "AUC = %.4f，远达不到申报书 ACC≥91%% 的指标。\n"
            "  * 训练集用的是 ForenSynths 的 val 划分（官方 train 划分 70.4 GB 在受限网络下\n"
            "    未完整取得），因此**不能**声称复现了 CNNDetection 的官方协议。\n"
            "  * 它的价值在于：证明了「加载预训练权重后判决边界能够建立」"
            "（tn>0 且 tp>0），\n"
            "    而随机初始化臂只会输出单一类别（tn=0）。但两臂 AUC 只差约 0.05，\n"
            "    所以严谨表述是「预训练权重是判决边界能建立的**必要条件**，非充分条件」。\n"
            "  * 同目录下的自检产物 `checkpoints/_probe/` 里的模型**都不可用**：\n"
            "    它们要么只训了 3 个 batch，要么是随机初始化。\n"
            "    同理 `checkpoints/demo_best.pt` 也不可用 —— 实测 `train_n_batches=9`、\n"
            "    `tn=0 / fp=15 / tp=30`（全判假），是退化的冒烟测试产物，不要拿它做演示。\n\n"
            "== ★ 实测的能力边界：不要拿它去判传统篡改（PS 拼接/复制移动）==\n"
            "  本权重只在 **ForenSynths（ProGAN 生成式伪造）** 上训过。\n"
            "  在 **CASIA v2（传统篡改）** 上实测（320 张，160 真 / 160 篡改）：\n"
            "      cls AUC = 0.4920   ACC = 0.5125   —— **接近随机**\n"
            "  定位指标三个口径全为 0（该权重只训过 stage1_cls_pretrain，定位头未训练）。\n"
            "  → 演示时请用**生成式伪造（GAN/扩散）图**，不要用 PS 拼接图；\n"
            "    否则界面会给出近似随机的结论，看起来像模型坏了。\n"
            "    要覆盖传统篡改，必须按 docs/15 跑完整三阶段训练"
            "（train_sets 里同时有 ForenSynths + CASIAv2）。\n\n"
            "正式三阶段训练与全部评测流程见 docs/15_云端GPU完整训练与评测手册.md。\n"
            % (auc_pre if auc_pre is not None else float("nan")))
    log(f"已写来历说明：{NOTE}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="对照实验收尾")
    ap.add_argument("--no-wait", action="store_true", help="不等待，直接收尾")
    ap.add_argument("--max-wait", type=float, default=8.0, help="最长等待小时数")
    ap.add_argument("--no-promote", action="store_true", help="只出报告，不动权重")
    args = ap.parse_args()

    if not args.no_wait:
        wait_for_exit(args.max_wait)
    regenerate_report()
    if not args.no_promote:
        promote()
    log("收尾完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
