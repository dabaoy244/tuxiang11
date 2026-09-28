#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把「第三个测试域」的几组评测汇总成一张结论表。

为什么单独写
------------
1. **指标必须只有一处实现**：AUC / AP 一律从 `src/engine/metrics.py` 取，
   本脚本不自己算，避免又出现「同名不同义」（见 docs/08 的 D23）。
2. **必须验证「处理前 / 后」是同一批样本**：如果两个变体的样本集不同，
   前后差值里就混进了「样本不同」这个变量，归因不成立（见 docs/08 的 D28）。
   这里用**文件名主干集合**做断言，不靠"我记得是同一份"。
3. **结论表要能追溯**：每个数字都带上是哪个 `.npz` 文件算出来的。

用法
----
    python scripts/report_third_domain.py            # 读 outputs/cross_gen_scores/*.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine.metrics import average_precision, roc_auc  # noqa: E402

# 变体顺序即报告里的行顺序；标签是给人看的
VARIANT_LABEL = {
    "raw": "统一处理前（PNG/JPEG 原样，硬链接）",
    "norm": "统一处理后（512×512 JPEG q=90）",
}
ARM_LABEL = {"pretrained": "预训练骨干", "random": "随机初始化骨干"}


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {
        "names": d["names"], "y": d["y"].astype(int), "probs": d["probs"].astype(float),
        "ckpt": str(d["ckpt"]), "path": os.path.relpath(path, ROOT),
    }
    # `paths` 是后来才加进 npz 的；老文件没有，标记为 None 并退化为"查目录"。
    out["paths"] = ([str(p) for p in d["paths"]] if "paths" in d.files else None)
    return out


def file_stems(paths) -> list:
    """只取文件名主干：raw 变体是 .png/.jpeg，norm 变体是 .jpg，扩展名必然不同。"""
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def dir_stems(root: str) -> list:
    """直接列目录，得到该评测域用到的全部样本主干（0_real 与 1_fake 合并）。"""
    out = []
    for sub in ("0_real", "1_fake"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp",
                                                  ".tif", ".tiff"):
                out.append(os.path.splitext(f)[0])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总第三个测试域（GenImage SDv1.4 子集）的结果")
    ap.add_argument("--scores-dir", default="outputs/cross_gen_scores")
    ap.add_argument("--pattern", default="sdv14_*.npz")
    ap.add_argument("--out-md", default="outputs/gensynth_eval/third_domain_summary.md")
    ap.add_argument("--out-json", default="outputs/gensynth_eval/third_domain_summary.json")
    ap.add_argument("--variant-dir", nargs="*", default=[
        "raw=data/Datasets/GenImageSDv14_raw/test/sdv14raw",
        "norm=data/Datasets/GenImageSDv14/test/sdv14",
        "lp=data/Datasets/GenImageSDv14_lp/test/sdv14lp",
    ], help="变体名=评测域目录，用于跨变体比对样本集（形如 norm=data/Datasets/xxx/test/gen）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, args.scores_dir, args.pattern)))
    if not files:
        print(f"❌ 没找到 {args.scores_dir}/{args.pattern}")
        return 1

    runs = {}
    for p in files:
        key = os.path.splitext(os.path.basename(p))[0]
        # 形如 sdv14_<variant>_<arm>
        parts = key.split("_")
        variant, arm = parts[-2], parts[-1]
        runs[(variant, arm)] = load(p)

    var_dir = {}
    for spec in args.variant_dir:
        if "=" in spec:
            k, v = spec.split("=", 1)
            var_dir[k] = os.path.join(ROOT, v)

    # ---- 样本一致性检查（before/after 必须是同一批样本，见 docs/08 D28）
    print("=" * 74)
    print("样本一致性检查（before/after 必须是同一批样本，见 docs/08 D28）")
    print("=" * 74)
    bad = []
    variants = sorted({v for v, _ in runs})
    for variant in variants:
        arms = sorted(a for vv, a in runs if vv == variant)
        ref_y = None
        for arm in arms:
            r = runs[(variant, arm)]
            y = r["y"]
            extra = ""
            if r["paths"] is not None:
                f0 = file_stems(r["paths"])
                extra = f"  文件 {len(f0)} 个 / 唯一主干 {len(set(f0))}"
                if len(f0) != len(set(f0)):
                    bad.append(f"{variant}/{arm}: 同一批次里出现重复文件主干")
            else:
                extra = "  （旧 npz，无文件清单）"
            if ref_y is None:
                ref_y, ref_arm = y, arm
            elif not np.array_equal(y, ref_y):
                bad.append(f"{variant}: {arm} 与 {ref_arm} 的标签向量不一致"
                           f"（同变体内两臂本应走同一份评测计划）")
            print(f"  {variant:5s} {arm:11s} n={len(y):5d} 假样本 {int(y.sum())}{extra}")

    # ---- 跨变体：优先用 npz 里的文件清单，其次退化为直接列目录
    if len(variants) > 1:
        base = {}
        for v in variants:
            arm = sorted(a for vv, a in runs if vv == v)[0]
            paths = runs[(v, arm)]["paths"]
            if paths is not None:
                base[v] = file_stems(paths)
                src = "npz 内的文件清单"
            elif v in var_dir and os.path.isdir(var_dir[v]):
                base[v] = dir_stems(var_dir[v])
                src = f"目录 {os.path.relpath(var_dir[v], ROOT)}"
            else:
                print(f"  ⚠ 变体 {v} 无法取得样本清单（旧 npz 且未提供 --variant-dir {v}=...）")
                continue
            print(f"  [{v}] 样本主干 {len(base[v])} 个（来源：{src}）")
        ref_v = next(iter(base))
        for v in base:
            if v == ref_v:
                continue
            if base[v] != base[ref_v]:
                only_a = sorted(set(base[ref_v]) - set(base[v]))[:3]
                only_b = sorted(set(base[v]) - set(base[ref_v]))[:3]
                bad.append(f"变体 {v} 与 {ref_v} 的样本主干集合不一致"
                           f"（仅 {ref_v} 有 {only_a}；仅 {v} 有 {only_b}）")
            else:
                print(f"  ✅ 变体 {v} 与 {ref_v} 是同一批样本（{len(base[v])} 个主干逐项相同）")
    if bad:
        print("\n❌ 样本一致性检查未通过：")
        for b in bad:
            print(f"   - {b}")
        print("   前后差值无法归因到清洗动作，报告不应给出 before/after 对比。")
        return 1
    print("  ✅ 通过")

    # ---- 指标
    rows = []
    for (variant, arm), r in sorted(runs.items()):
        y, s = r["y"], r["probs"]
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        pred = (s >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({
            "variant": variant, "arm": arm,
            "n_real": n0, "n_fake": n1,
            "acc": float((tp + tn) / len(y)),
            "real_acc": float(tn / n0) if n0 else float("nan"),
            "fake_acc": float(tp / n1) if n1 else float("nan"),
            "auc": float(roc_auc(y, s)),
            "ap": float(average_precision(y, s)),
            "tn": tn, "tp": tp,
            "ckpt": r["ckpt"], "scores": r["path"],
        })

    md = []
    md.append("# 第三个测试域：GenImage SDv1.4 子集（同伴提供数据改造）\n")
    md.append(f"- 样本：{rows[0]['n_real']} 真 + {rows[0]['n_fake']} 假 = "
              f"{rows[0]['n_real'] + rows[0]['n_fake']} 张，"
              f"交集 14 个 ImageNet 类别，**逐类真假数量相等**")
    md.append("- AUC / AP 取自 `src/engine/metrics.py` 的**唯一实现**"
              "（并列分数秩平均；AP 为分组 ΣP·ΔR）")
    md.append("- 分数原始文件：`outputs/cross_gen_scores/sdv14_*.npz`\n")
    md.append("| 变体 | 骨干 | ACC | 真图ACC | 假图ACC | tn(真判真) | AUC | AP |")
    md.append("|---|---|---|---|---|---|---|---|")
    for v in variants:
        for arm in sorted({a for vv, a in runs if vv == v}):
            r = next(x for x in rows if x["variant"] == v and x["arm"] == arm)
            md.append(f"| {VARIANT_LABEL.get(v, v)} | {ARM_LABEL.get(arm, arm)} "
                      f"| {r['acc']:.4f} | {r['real_acc']:.4f} | {r['fake_acc']:.4f} "
                      f"| {r['tn']} | **{r['auc']:.4f}** | {r['ap']:.4f} |")
    md.append("")

    # ---- 关键论断：把「清洗带来多少变化」算出来
    if {"raw", "norm"} <= set(variants):
        md.append("## 清洗动作的净效果（同一批样本，只有处理方式不同）\n")
        md.append("| 骨干 | raw AUC | norm AUC | Δ | raw 真图ACC | norm 真图ACC |")
        md.append("|---|---|---|---|---|---|")
        for arm in sorted({a for vv, a in runs}):
            rr = [x for x in rows if x["variant"] == "raw" and x["arm"] == arm]
            nn = [x for x in rows if x["variant"] == "norm" and x["arm"] == arm]
            if not rr or not nn:
                continue
            rr, nn = rr[0], nn[0]
            md.append(f"| {ARM_LABEL.get(arm, arm)} | {rr['auc']:.4f} | {nn['auc']:.4f} "
                      f"| {nn['auc'] - rr['auc']:+.4f} | {rr['real_acc']:.4f} "
                      f"| {nn['real_acc']:.4f} |")
        md.append("")

    md.append("> 口径提醒：本域是**整图 AI 生成**（`kind: gensynth`，无掩码），")
    md.append("> 只能用于真伪分类，**不能**支撑定位指标；且只有一个生成器（`sdv4`），")
    md.append("> 它检验的是**跨数据源 / 跨内容分布**泛化，不是跨伪造类型泛化。")
    md.append("> 数据出处是 GenImage 的 **train split**（同伴确认）——")
    md.append("> **只做零样本测试域，禁止用于训练**，否则是数据泄漏。")
    md.append("> 残余平凡线索（统一处理后）：文件级 `每像素字节数` AUC = 0.6374；")
    md.append("> 像素级最高 `Laplacian 方差` AUC = 0.6399。详见 `outputs/audit_sdv14_normalized.txt`。")

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out_md)) or ".", exist_ok=True)
    with open(os.path.join(ROOT, args.out_md), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(ROOT, args.out_json), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "variants": variants}, f,
                  ensure_ascii=False, indent=2, default=float)

    print("\n" + "\n".join(md))
    print(f"\n已写出：\n  {args.out_md}\n  {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
