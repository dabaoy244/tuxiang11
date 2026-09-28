#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标口径回归测试 —— 用手算样例把**分类**与**定位**两类指标的定义钉死。

为什么要这个测试
----------------
本项目有两处指标口径会造成"数字看着正常、其实取错"的事故：

1. **mIoU 有三种定义**，同一模型能差一倍以上，而申报指标写的是
   「CASIAv2 篡改定位 mIoU ≥ 56%」——不写清是哪一种，等于没有指标；
2. **AUC 在并列分数上会算错**。退化模型常输出**饱和同分**（整批都是 0.0 或 1.0），
   若用"排序后阶梯曲线 + 梯形积分"，全同分时 AUC 会算成 **0.0**（应为 0.5）——
   而且这个偏差**方向对我们有利**（会夸大与预训练臂的差距），更不能含糊。

只要有人改了 `src/engine/metrics.py` 或 `scripts/eval_cross_generator.py` 的度量实现，
这个测试就会失败。**它是把"指标定义"从口头约定变成可执行断言的手段。**

覆盖范围
--------
| 组 | 对象 | 关键断言 |
|---|---|---|
| A | 定位指标（三种 mIoU 口径） | 数值手算对拍；`仅篡改图 > 池化 > 全部非空图`；空掩码图**被跳过而非记 IoU=1** |
| B | 分类指标（AUC / AP） | 完美可分=1.0、完全反向=0.0、**全同分=0.5**、全判一类=0.5、单类=nan |

用法
----
    python scripts/test_metric_definitions.py      # 退出码 0 = 全部通过
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_cross_generator import metrics as cls_metrics   # noqa: E402
from src.engine.metrics import ClassificationMetrics               # noqa: E402
from src.engine.metrics import LocalizationMetrics                # noqa: E402

FAILS = []
H = W = 8


def chk(name: str, got, want, tol: float = 1e-6) -> None:
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = False
    print(f"  {'✅' if ok else '❌'} {name:34s} got={got}  want={want}")
    if not ok:
        FAILS.append(name)


def chk_true(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'✅' if cond else '❌'} {name:34s} {detail}")
    if not cond:
        FAILS.append(name)


# ==========================================================================
def part_a_localization() -> None:
    """4 张 8×8 图，每张 IoU 都能手算。"""
    print("=" * 70)
    print("A. 定位指标：三种 mIoU 口径")
    print("=" * 70)

    prob = np.zeros((4, 1, H, W), dtype=np.float32)
    gt = np.zeros((4, 1, H, W), dtype=np.float32)
    # 图0：篡改（左半 32 px），预测也覆盖左半 → inter=32 union=32 IoU=1.0
    gt[0, 0, :, :4] = 1.0
    prob[0, 0, :, :4] = 0.9
    # 图1：真实（GT 全零），预测全零 → union=0，两种逐图口径都**跳过**
    # 图2：真实（GT 全零），误报 2×2=4 px → inter=0 union=4 IoU=0.0
    prob[2, 0, :2, :2] = 0.8
    # 图3：篡改 4 px（左上 2×2），预测覆盖整张 64 px → inter=4 union=64 IoU=0.0625
    gt[3, 0, :2, :2] = 1.0
    prob[3, 0, :, :] = 0.9

    m = LocalizationMetrics(threshold=0.5)
    m.update(prob, gt)
    r = m.compute()

    print("  像素级混淆矩阵（全局汇总）")
    chk("tp", m.tp, 36)
    chk("fp", m.fp, 64)
    chk("fn", m.fn, 0)
    chk("tn", m.tn, 156)
    chk("n_tampered", m.n_tampered, 2)

    print("  三种口径")
    chk("miou（池化）", r["miou"], 36 / 100)
    chk("miou_tampered_only", r["miou_tampered_only"], (1.0 + 0.0625) / 2)
    chk("miou_per_sample", r["miou_per_sample"], (1.0 + 0.0 + 0.0625) / 3)

    print("  口径之间的排序（这个不等关系必须成立）")
    chk_true("tampered_only > 池化 > per_sample",
             r["miou_tampered_only"] > r["miou"] > r["miou_per_sample"],
             f"({r['miou_tampered_only']:.4f} > {r['miou']:.4f} > "
             f"{r['miou_per_sample']:.4f})")

    print("  关键不变量：全零预测的真实图必须被**跳过**，而不是记成 IoU=1")
    m2 = LocalizationMetrics(threshold=0.5)
    m2.update(np.zeros((1, 1, H, W), dtype=np.float32),
              np.zeros((1, 1, H, W), dtype=np.float32))
    r2 = m2.compute()
    chk("miou_per_sample（应无样本）", r2["miou_per_sample"], 0.0)
    chk("miou_tampered_only（应无样本）", r2["miou_tampered_only"], 0.0)
    chk_true("逐图列表为空（跳过而非记 1）",
             len(m2.per_sample_iou) == 0 and m2.n_tampered == 0)
    print()


# ==========================================================================
def part_b_classification() -> None:
    """分类指标：AUC 必须做并列校正，AP 必须能到 1.0。"""
    print("=" * 70)
    print("B. 分类指标：AUC / AP")
    print("=" * 70)

    y = np.array([0, 0, 0, 1, 1, 1])

    def auc_of(s):
        return cls_metrics(y, np.asarray(s, dtype=float))["auc"]

    def ap_of(s):
        return cls_metrics(y, np.asarray(s, dtype=float))["ap"]

    print("  AUC 边界情形（★ 前四条是必须成立的）")
    chk("完美可分 → 1.0", auc_of([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), 1.0)
    chk("完全反向 → 0.0", auc_of([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), 0.0)
    chk("全同分 → 0.5", auc_of([0.5] * 6), 0.5)
    chk("全同分且饱和=1.0 → 0.5", auc_of([1.0] * 6), 0.5)
    chk("全判一类（分数全 0） → 0.5", auc_of([0.0] * 6), 0.5)

    print("  部分并列（退化模型的典型形态：一半饱和）")
    y6 = np.array([0] * 6 + [1] * 6)
    s6 = np.array([0.0] * 6 + [1.0] * 6)
    chk("真图全 0 / 假图全 1 → 1.0", cls_metrics(y6, s6)["auc"], 1.0)

    print("  单类（只有一个类别时无定义）")
    chk_true("单类 → nan", np.isnan(cls_metrics(np.ones(4),
                                               np.array([0.1, 0.2, 0.3, 0.4]))["auc"]))

    print("  AP（标准定义，按不同分数值分组；★ 完美可分必须 = 1.0）")
    chk("完美可分 → 1.0", ap_of([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), 1.0)
    chk("完美可分（同分饱和） → 1.0", ap_of([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 1.0)
    chk_true("AP 不出现 > 1", ap_of([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) <= 1.0 + 1e-9)

    print("  ★ AP 的判别性边界：分数全同时 AP 必须 = 正类占比（不是 0、也不是 1）")
    chk("6 张全同分（正类占 3/6=0.5） → 0.5", ap_of([0.5] * 6), 0.5)
    chk("全判同一类（分数全 0） → 0.5", ap_of([0.0] * 6), 0.5)
    y10 = np.array([0] * 8 + [1] * 2)
    chk("8 真 2 假全同分 → 0.2", cls_metrics(y10, np.zeros(10))["ap"], 0.2)
    chk("8 真 2 假完美可分 → 1.0", cls_metrics(y10, np.r_[np.zeros(8), np.ones(2)])["ap"], 1.0)

    print("  对照：不做并列校正的梯形法 —— 存下来只是为了显式暴露这个偏差")
    chk("梯形法在全同分上给出 0.0（**错的**）",
        cls_metrics(y, np.full(6, 0.5))["auc_trapz"], 0.0)
    print()

    # ---------------------------------------------------------------
    print("  ★★ 同一指标必须只有一套定义（跨入口一致性）")
    print("     本项目曾在 `src/engine/metrics.py` 与 `scripts/eval_cross_generator.py`")
    print("     各留一份 AUC/AP 实现，**同名字段 `ap` 下是两套不同定义**：")
    print("     一个 11 点插值、一个逐正例求和，两张表的数字因此不可比。")
    rng = np.random.RandomState(3407)
    cases = [
        ("真实分布",      np.array([0] * 80 + [1] * 20), rng.rand(100)),
        ("一半饱和",      np.array([0] * 50 + [1] * 50), np.r_[np.zeros(50), rng.rand(50)]),
        ("全判假（退化）", np.array([0] * 50 + [1] * 50), np.zeros(100)),
        ("全同分",        np.array([0] * 50 + [1] * 50), np.full(100, 0.5)),
    ]
    for tag, yy, ss in cases:
        cm = ClassificationMetrics()
        cm.update(np.asarray(ss, float), np.asarray(yy, int))
        got_main = cm.compute()
        got_cross = cls_metrics(yy, ss)
        same_auc = abs(got_main["auc"] - got_cross["auc"]) < 1e-12
        same_ap = abs(got_main["ap"] - got_cross["ap"]) < 1e-12
        chk_true(f"主评测 vs 跨生成器 AUC 一致（{tag}）", same_auc,
                 f"{got_main['auc']:.6f} vs {got_cross['auc']:.6f}")
        chk_true(f"主评测 vs 跨生成器 AP 一致（{tag}）", same_ap,
                 f"{got_main['ap']:.6f} vs {got_cross['ap']:.6f}")

    print("  ★ 退化模型的判别性统计量（不用 ACC —— 它与模型无关）")
    cm = ClassificationMetrics()
    cm.update(np.zeros(100), np.array([0] * 50 + [1] * 50))
    d = cm.compute()
    chk_true("全判假 → tn>0 且 tp==0（判决退化）", d["tn"] == 50 and d["tp"] == 0,
             f"tn={d['tn']} fp={d['fp']} fn={d['fn']} tp={d['tp']}")
    chk("全判假时 ACC 恰为 0.5（**故不可用**）", d["acc"], 0.5)
    print()


# ==========================================================================
def main() -> int:
    part_a_localization()
    part_b_classification()
    print("=" * 70)
    if FAILS:
        print(f"❌ 失败 {len(FAILS)} 项：{FAILS}")
        return 1
    print("✅ 全部通过 —— 分类与定位指标的口径已被手算样例钉死。")
    print("   写论文/中期材料时：")
    print("   * mIoU 报 `miou_tampered_only` 并注明口径，不要挑最高的那个；")
    print("   * AUC 必须是并列校正版（全同分 → 0.5）；AP 必须是分组版（全同分 → 正类占比）；")
    print("   * 两者都只从 `src/engine/metrics.py` 取，不要在脚本里再抄一份。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
