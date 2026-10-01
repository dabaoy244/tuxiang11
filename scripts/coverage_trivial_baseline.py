"""COVERAGE 定位指标的『平凡基线』探针。

目的：判定 `miou_tampered_only`（full = 0.1106）到底是"弱但有信号"还是"也就等于零"。
做法：同一批 91 张篡改图上，算几种不需要任何模型能力的预测的 IoU 均值：
  A. 全零预测      -> IoU = 0（这是"完全不预测"的下界）
  B. 全图预测为篡改 -> IoU = a（a = GT 面积占比）
  C. 居中正方框（面积 = GT 面积）-> 看"猜中间"能拿多少
  D. 随机位置同面积框（解析期望）-> IoU = a / (2 - a)
只要模型的 0.1106 不显著高于 C/D，就不能说"定位能力零样本迁移了"。
"""
import glob
import os

import numpy as np
from PIL import Image

MASK_DIR = os.path.join("data", "external", "COVERAGE", "test", "mask")


def main() -> None:
    files = sorted(glob.glob(os.path.join(MASK_DIR, "*t.png")))
    print(f"篡改掩码数: {len(files)}")

    rows = []
    for f in files:
        m = np.array(Image.open(f).convert("L"))
        gt = m > 127
        h, w = gt.shape
        total = h * w
        a = float(gt.sum()) / total
        if gt.sum() == 0:
            continue  # 只统计真有前景的

        # C. 居中正方框，面积对齐 GT
        side = int(round((a * total) ** 0.5))
        box = np.zeros_like(gt)
        cy, cx = h // 2, w // 2
        y0, y1 = max(0, cy - side // 2), min(h, cy + side // 2)
        x0, x1 = max(0, cx - side // 2), min(w, cx + side // 2)
        box[y0:y1, x0:x1] = True
        inter = float(np.logical_and(box, gt).sum())
        union = float(np.logical_or(box, gt).sum())
        iou_center = inter / union if union else 0.0

        rows.append((a, iou_center, a / (2.0 - a) if a < 2 else 0.0, gt.shape))

    a_arr = np.array([r[0] for r in rows])
    c_arr = np.array([r[1] for r in rows])
    d_arr = np.array([r[2] for r in rows])

    print(f"有效样本 n = {len(rows)}")
    print(f"GT 面积占比 a : mean={a_arr.mean():.4f}  median={np.median(a_arr):.4f} "
          f"p10={np.percentile(a_arr, 10):.4f}  p90={np.percentile(a_arr, 90):.4f}  max={a_arr.max():.4f}")
    print()
    print("平凡基线 IoU（同一 91 张篡改图上的均值）:")
    print(f"  A 全零预测        : 0.0000")
    print(f"  B 全图预测        : {a_arr.mean():.4f}")
    print(f"  C 居中框(面积对齐) : {c_arr.mean():.4f}   中位={np.median(c_arr):.4f}")
    print(f"  D 随机框(解析期望) : {d_arr.mean():.4f}")
    print()
    print(f"参照：模型 full 实测 miou_tampered_only = 0.1106")
    print(f"     模型 Dice = 0.1925  -> 反推 IoU = {0.1925 / (2 - 0.1925):.4f}（自洽校验）")
    print()
    best_trivial = max(a_arr.mean(), c_arr.mean(), d_arr.mean())
    print(f"最强的平凡基线 = {best_trivial:.4f}")
    if 0.1106 > best_trivial:
        print(f"⇒ 模型比最强平凡基线高 {0.1106 - best_trivial:+.4f}")
    else:
        print(f"⇒ ★ 模型【没有】超过最强平凡基线（差 {0.1106 - best_trivial:+.4f}）")


if __name__ == "__main__":
    main()
