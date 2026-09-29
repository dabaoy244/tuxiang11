#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `outputs/coverage_ext/external_COVERAGE_*.json` 汇总成一张外部域基准表。

外部域 COVERAGE 是**零样本**评测（权重全程没见过它），所以这张表回答的是
「模型能不能迁移到『没训练过的伪造类型』」，与训练域内部的 test 口径要分开报。

口径红线（脚本会强制写进表头，避免以后误并列）：
  · `miou_tampered_only` 才是跨数据集可比的定位指标（只在 GT 有前景的图上平均）；
  · `miou`(池化) / `miou_per_sample` / dice / pixel_acc 受「100 张全零 GT 真实图」
    影响，**不可**与 CASIAv2 的同名指标并列。

用法：
    python scripts/report_coverage_rows.py            # 读默认目录
    python scripts/report_coverage_rows.py --out outputs/coverage_ext
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: 报告里的行序：先正式权重，再 full 臂，最后其余臂
TAG_ORDER = ["formal0928", "ckpt0928", "full", "no_vib", "no_cross_attention",
             "no_edge", "no_learnable_mask", "no_phase", "no_grad_stop"]
TAG_LABEL = {
    "formal0928": "09-28 正式权重", "ckpt0928": "09-28 正式权重（云端 checkpoints/）",
    "full": "full（完整模型）",
    "no_vib": "−VIB", "no_cross_attention": "−CS-CAM 跨支路注意力",
    "no_edge": "−边缘监督", "no_learnable_mask": "−可学习频域掩码",
    "no_phase": "−相位支路", "no_grad_stop": "−梯度阻断",
}


def g(v, n=4):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/coverage_ext")
    ap.add_argument("--pattern", default="external_COVERAGE_*.json")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    files = sorted(glob.glob(os.path.join(out_dir, args.pattern)))
    if not files:
        print(f"❌ 没找到 {args.out}/{args.pattern}；先跑 scripts/eval_external_domain.py")
        return 1

    rows = []
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        tag = os.path.basename(p)[len("external_COVERAGE_"):-len(".json")]
        rows.append((tag, d, os.path.relpath(p, ROOT)))

    def key(t):
        return (TAG_ORDER.index(t) if t in TAG_ORDER else 99, t)

    rows.sort(key=lambda r: key(r[0]))

    dom = rows[0][1]
    n_tot, n_real, n_fake = dom["n_total"], dom["n_real"], dom["n_fake"]

    L = []
    L.append(f"# 外部域独立基准行：COVERAGE（零样本跨数据集，{n_real} 真 + {n_fake} 篡改 = {n_tot} 张）")
    L.append("")
    L.append("- 定义：COVERAGE 是**复制-移动**篡改数据集，与训练域（ForenSynths 生成图 / "
             "CASIAv2 拼接·复制移动）**不同源**，且全程**未参与任何训练**。")
    L.append("- 因此本表是**零样本跨数据集**结果，回答「迁移到没见过的伪造类型还行不行」，"
             "**不可**与训练域内部 test 指标并列。")
    L.append("- 定位口径：★ 只报 `miou_tampered_only`（在 GT 真有前景的图上逐图平均，"
             "与 ManTra-Net / SPAN / CAT-Net 可比）。")
    L.append("- ⚠️ 池化 `miou` / `miou_per_sample` / Dice / PixelAcc 受「COVERAGE 真实图"
             "自带全零掩码文件」影响（这些图会被计入定位池、误报被罚），"
             "**跨数据集不可比**，故不列。")
    L.append("")
    L.append("| 权重 | 样本 | 真伪 ACC | AUC | AP | F1 | 真实图召回 | 篡改图召回 | 定位 mIoU★ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tag, d, rel in rows:
        cls = d.get("cls") or {}
        loc = d.get("loc") or {}
        nr = d.get("n_real") or 0
        tn, fp = cls.get("tn", 0), cls.get("fp", 0)
        real_rec = (tn / (tn + fp)) if (tn + fp) else float("nan")
        L.append(
            f"| {TAG_LABEL.get(tag, tag)} | {d.get('n_total')} "
            f"| **{g(cls.get('acc'))}** | **{g(cls.get('auc'))}** "
            f"| {g(cls.get('ap'))} | {g(cls.get('f1'))} "
            f"| {g(real_rec)} | {g(cls.get('recall'))} "
            f"| **{g(loc.get('miou_tampered_only'))}** |")
    L.append("")
    L.append("> 随机基线：ACC 0.5 / AUC 0.5 / mIoU 0.0。"
             "AUC 落在 0.5 附近 = 该权重在 COVERAGE 上**没有可用的鉴别信号**。")
    L.append("")

    md = "\n".join(L)
    print(md)

    md_path = args.md or os.path.join(out_dir, "coverage_rows.md")
    with open(os.path.join(ROOT, md_path) if not os.path.isabs(md_path) else md_path,
              "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"已写出：{os.path.relpath(md_path, ROOT) if not os.path.isabs(md_path) else md_path}")

    # 明细溯源
    print("\n逐行来源：")
    for tag, _, rel in rows:
        print(f"  {tag:22s} {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
