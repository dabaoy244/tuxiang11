#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""决策阈值扫描：用已经存盘的逐样本分数找最优判定阈值（零推理成本）。

为什么值得单独做
----------------
2026-09-29 的 ForenSynths 跨生成器评测暴露一组很刺眼的数：

    宏平均 ACC = 0.6923      宏平均 AUC = 0.8776
    真图 ACC   = 0.4692      假图 ACC   = 0.9154

**ACC 比 AUC 低 0.185** —— 排序能力（AUC）明显好于判定结果（ACC），
这只有一个解释：**0.5 这个阈值离最优位置很远**，模型系统性偏向判"伪造"，
于是把大量真实图误判为伪造（cyclegan 的真图 ACC 低到 0.0133）。

而阈值是**零成本**的：既不用重训，也不用 GPU，甚至不用重新推理 ——
只要逐样本分数还在（`--scores-out` 存下的 npz），扫一遍即可。

⚠ 两个必须一起报的口径
--------------------
1. 在**全体测试样本**上挑最优阈值，是**乐观估计**（阈值和指标用了同一批数据）。
   所以本脚本同时给出 `holdout` 结果：按图片名哈希对半劈开，
   一半挑阈值、另一半报指标 —— 这个数才是能写进论文的。
2. `ForenSynths/test` 里的 `progan` 子集与**训练集同生成器**（train 只有 ProGAN），
   其 ACC 恒为 1.0，会把宏平均**抬高**。本脚本同时给"含 progan / 不含 progan"两行。

用法
----
    python scripts/tune_decision_threshold.py \
        --scores outputs/cross_gen_20260928/scores.npz \
        --out outputs/cross_gen_20260928
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _acc(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    return float(((p >= thr).astype(int) == y).mean())


def macro_acc(names: np.ndarray, y: np.ndarray, p: np.ndarray, thr: float) -> float:
    """先按生成器算 ACC，再对生成器取平均（每个生成器等权）。"""
    vals = [_acc(y[names == g], p[names == g], thr) for g in np.unique(names)]
    return float(np.mean(vals)) if vals else float("nan")


def real_fake_acc(names: np.ndarray, y: np.ndarray, p: np.ndarray, thr: float,
                  keep: np.ndarray | None = None):
    m = np.ones_like(y, dtype=bool) if keep is None else keep
    out = {}
    for lab, key in ((0, "real"), (1, "fake")):
        sel = m & (y == lab)
        out[key] = float(((p[sel] >= thr).astype(int) == y[sel]).mean()) if sel.any() else float("nan")
    return out


def sweep(names, y, p, keep, grid):
    best = (None, -1.0)
    curve = []
    for t in grid:
        a = macro_acc(names[keep], y[keep], p[keep], float(t))
        curve.append((float(t), a))
        if a > best[1]:
            best = (float(t), a)
    return best, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="eval_cross_generator.py --scores-out 产出的 npz")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--drop-seen-generators", nargs="*", default=["progan"],
                    help="训练时见过的生成器（默认 progan），不计入严格跨生成器宏平均")
    args = ap.parse_args()

    d = np.load(args.scores, allow_pickle=True)
    names, y, p = d["names"], d["y"].astype(int), d["probs"].astype(float)
    gens = list(np.unique(names))
    dropped = [g for g in args.drop_seen_generators if g in gens]
    keep_cross = ~np.isin(names, dropped)

    grid = np.arange(0.02, 0.99 + 1e-9, args.step)

    L = []
    L.append("# 决策阈值扫描（零推理成本）")
    L.append("")
    L.append(f"- 分数来源：`{args.scores}`")
    L.append(f"- 样本数 {len(y)}，生成器 {len(gens)} 个"
             f"{'，其中 ' + str(dropped) + ' 与训练集同生成器' if dropped else ''}")
    L.append("")

    # ---- 1. 现状 ----
    L.append("## 一、阈值 0.5 的现状")
    L.append("")
    L.append("| 口径 | 宏平均 ACC | 真图 ACC | 假图 ACC |")
    L.append("|---|---|---|---|")
    a_all = macro_acc(names, y, p, 0.5)
    rf_all = real_fake_acc(names, y, p, 0.5)
    L.append(f"| 13 生成器（含 progan） | **{a_all:.4f}** | {rf_all['real']:.4f} | {rf_all['fake']:.4f} |")
    a_cross = macro_acc(names[keep_cross], y[keep_cross], p[keep_cross], 0.5)
    rf_cross = real_fake_acc(names, y, p, 0.5, keep_cross)
    L.append(f"| 严格跨生成器（排除 {','.join(dropped) if dropped else '无'}） | "
             f"**{a_cross:.4f}** | {rf_cross['real']:.4f} | {rf_cross['fake']:.4f} |")
    L.append("")

    # ---- 2. 全样本扫（乐观） ----
    (bt_all, ba_all), _ = sweep(names, y, p, np.ones_like(y, bool), grid)
    (bt_cross, ba_cross), _ = sweep(names, y, p, keep_cross, grid)
    L.append("## 二、扫阈值（同一批数据选阈值 + 报指标 ⇒ **乐观，仅供看上限**）")
    L.append("")
    L.append("| 口径 | 最优阈值 | 宏平均 ACC | 相比 0.5 |")
    L.append("|---|---|---|---|")
    L.append(f"| 含 progan | {bt_all:.3f} | **{ba_all:.4f}** | {ba_all - a_all:+.4f} |")
    L.append(f"| 严格跨生成器 | {bt_cross:.3f} | **{ba_cross:.4f}** | {ba_cross - a_cross:+.4f} |")
    L.append("")

    # ---- 3. 半劈 holdout（诚实） ----
    # 按路径哈希分半，避免用同一批样本既挑阈值又报指标。
    key = None
    for k in ("paths", "names"):
        if k in d:
            key = d[k]
            break
    if key is not None:
        # ⚠️ 不能用内置 `hash()` 分半：CPython 对 str 的 hash 每个进程都随机化
        #   （PYTHONHASHSEED），同一份 npz 两次运行会劈出不同的两半，
        #   "诚实 holdout" 的结论就不可复现 —— 而这个数是要写进论文的。
        #   改用 md5，跨进程 / 跨机器稳定。
        import hashlib

        h = np.array([int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % 2
                      for x in key])
        h = h.astype(bool)
        L.append("## 三、半劈 holdout（★这个数才能写进论文）")
        L.append("")
        L.append("> 按样本标识哈希对半劈开：A 半选阈值，B 半报指标。")
        L.append("> 这样阈值没有见过 B 半的标签，指标是**无偏**的。")
        L.append("")
        L.append("| 方向 | 选出的阈值 | B 半宏平均 ACC（含 progan） | B 半（严格跨生成器） |")
        L.append("|---|---|---|---|")
        rows = []
        for name_a, name_b, ma, mb in (("A→B", "A 选 → B 报", h, ~h),
                                       ("B→A", "B 选 → A 报", ~h, h)):
            (tb, _), _ = sweep(names, y, p, ma, grid)
            ab = macro_acc(names[mb], y[mb], p[mb], tb)
            ab_c = macro_acc(names[mb & keep_cross], y[mb & keep_cross],
                             p[mb & keep_cross], tb)
            L.append(f"| {name_a} | {tb:.3f} | {ab:.4f} | {ab_c:.4f} |")
            rows.append({"dir": name_a, "thr": tb, "acc_all": ab, "acc_cross": ab_c})
        mean_all = float(np.mean([r["acc_all"] for r in rows]))
        mean_cross = float(np.mean([r["acc_cross"] for r in rows]))
        L.append(f"| **平均** | — | **{mean_all:.4f}** | **{mean_cross:.4f}** |")
        L.append("")
        L.append(f"→ 诚实结论：仅调阈值可把严格跨生成器宏平均 ACC "
                 f"从 **{a_cross:.4f} 提到约 {mean_cross:.4f}**"
                 f"（{mean_cross - a_cross:+.4f}），**不需要任何重训**。")
        L.append("")

    # ---- 4. 逐生成器最优阈值 ----
    L.append("## 四、逐生成器：0.5 vs 各自最优（看偏差是否系统性一致）")
    L.append("")
    L.append("| 生成器 | n | 0.5 阈值 ACC | 最优阈值 | 最优 ACC | 真图ACC@0.5 | 假图ACC@0.5 |")
    L.append("|---|---|---|---|---|---|---|")
    for g in gens:
        m = names == g
        a0 = _acc(y[m], p[m], 0.5)
        (bt, ba), _ = sweep(names[m], y[m], p[m], np.ones(m.sum(), bool), grid)
        rf = real_fake_acc(names[m], y[m], p[m], 0.5)
        L.append(f"| {g} | {int(m.sum())} | {a0:.4f} | {bt:.3f} | {ba:.4f} | "
                 f"{rf['real']:.4f} | {rf['fake']:.4f} |")
    L.append("")
    L.append("> 若各生成器最优阈值**落点分散**，说明单一全局阈值救不了，"
             "要的是校准（温度缩放 / Platt scaling）或修训练目标；"
             "若**落点集中偏高**，说明模型有固定的'偏向判伪造'偏差，"
             "一个全局阈值即可大幅改善。")
    L.append("")
    L.append(f"*由 `scripts/tune_decision_threshold.py` 生成 · "
             f"{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}*")

    os.makedirs(args.out, exist_ok=True)
    md = os.path.join(args.out, "threshold_sweep.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    js = os.path.join(args.out, "threshold_sweep.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({
            "n": int(len(y)), "generators": gens, "dropped": dropped,
            "acc05_all": a_all, "acc05_cross": a_cross,
            "best_all": {"thr": bt_all, "acc": ba_all},
            "best_cross": {"thr": bt_cross, "acc": ba_cross},
        }, f, ensure_ascii=False, indent=2, default=float)
    print("\n".join(L))
    print(f"\n[thr] 报告 {md}\n[thr] 明细 {js}")


if __name__ == "__main__":
    main()
