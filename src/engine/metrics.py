"""评测指标（申报书 6.3 评价指标）

真伪检测：ACC / Precision / Recall / F1 / AUC / AP(mAP)
篡改定位：mIoU / Pixel Acc / Dice / F1
推理性能：由 deploy/benchmark 脚本统计（CPU fps / GPU fps / 模型体积 MB）
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


# ==========================================================================
# ★ 全仓库唯一的 AUC / AP 实现。
#
# 为什么必须唯一：这两个指标曾经在 `src/engine/metrics.py` 和
# `scripts/eval_cross_generator.py` 各有一份实现，**同名字段 `ap` 下是两套不同定义**
# （一个 11 点插值、一个逐正例求和），导致主实验表和跨生成器表的数字不可比；
# 更糟的是两者在**并列分数**（退化模型的典型输出：整批 0.0 或 1.0）上都会偏低。
# 现在两个调用方都从这里取，`scripts/test_metric_definitions.py` 用边界对拍钉死。
# ==========================================================================
def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC 曲线下面积（Mann–Whitney U / 秩平均），对并列分数稳健。

    并列必须用**平均秩**：退化模型常整批输出饱和同分，
    若按"排序后梯形积分"算，全同分会被算成 0.0（正确值 0.5），
    从而**系统性压低退化臂的 AUC、反而放大对照实验的优势**。
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    pos, neg = (y == 1), (y == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1)
    # 并列值取平均秩
    i = 0
    while i < sp.size:
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """AP = Σ_n (R_n − R_{n−1}) · P_n，**按不同分数值分组**（VOC2010+ / COCO 口径）。

    为什么必须分组：PR 曲线的一个点是**一个阈值**，而并列分数共用一个阈值。
    若按"逐正例求和"，同分样本被强行排出先后，precision 会在并列段内人为波动，
    把 AP 算低 —— 全同分时有 50 正 50 负会得到 0.3118，而正确答案是正类占比 0.5。
    分组后：全同分 → 只有一个阈值 → AP = 正类占比（正确）；完美可分 → 1.0。
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys, sp = y[order], p[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    # 只在分数值发生变化处取点（并列共用一个阈值）
    last = np.r_[sp[1:] != sp[:-1], True]
    tp, fp = tp[last], fp[last]
    rec = tp / n_pos
    prec = tp / np.maximum(tp + fp, 1)
    d_rec = np.diff(np.r_[0.0, rec])
    return float(np.sum(d_rec * prec))


# ==========================================================================
class ClassificationMetrics:
    """整图真伪二分类指标累加器。正类 = 伪造(label=1)。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.y_true: List[np.ndarray] = []
        self.y_prob: List[np.ndarray] = []

    def update(self, logits_or_prob: np.ndarray, labels: np.ndarray) -> None:
        """接受 logits 或概率，自动识别（logits 可能为负或和不为 1）。"""
        arr = np.asarray(logits_or_prob, dtype=np.float64)
        if arr.ndim == 2:
            e = np.exp(arr - arr.max(axis=1, keepdims=True))
            prob = (e / e.sum(axis=1, keepdims=True))[:, 1]
        else:
            prob = arr
        self.y_prob.append(prob.ravel())
        self.y_true.append(np.asarray(labels).ravel().astype(np.int64))

    # ------------------------------------------------------------------
    @property
    def _all(self):
        if not self.y_true:
            return np.array([]), np.array([])
        return np.concatenate(self.y_true), np.concatenate(self.y_prob)

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        y, p = self._all
        if y.size == 0:
            return {}
        pred = (p >= threshold).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())

        acc = (tp + tn) / max(1, y.size)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)

        auc = roc_auc(y, p)
        ap = average_precision(y, p)
        return {
            "acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": auc, "ap": ap, "mAP": ap,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": int(y.size),
        }

    # ------------------------------------------------------------------
    # 保留这两个静态方法名作为**薄转发**，避免外部按旧名调用时静默失效；
    # 真正的实现已上移到模块级的 `roc_auc` / `average_precision`（全仓库唯一）。
    @staticmethod
    def _auc(y: np.ndarray, p: np.ndarray) -> float:
        return roc_auc(y, p)

    @staticmethod
    def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
        return average_precision(y, p)


# ==========================================================================
class LocalizationMetrics:
    """像素级篡改定位指标累加器。mask 为 {0,1}，只统计含有效 GT 的样本。"""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.dice_num = self.dice_den = 0.0
        self.per_sample_iou: List[float] = []
        self.per_sample_iou_tampered: List[float] = []
        self.n = 0
        self.n_tampered = 0

    def update(self, prob: np.ndarray, gt: np.ndarray) -> None:
        """prob/gt: (B,1,H,W) 或 (B,H,W)。"""
        p = np.asarray(prob)
        g = np.asarray(gt)
        if p.ndim == 4:
            p = p[:, 0]
        if g.ndim == 4:
            g = g[:, 0]
        pred = (p >= self.threshold).astype(np.uint8)
        gt_b = (g > 0.5).astype(np.uint8)

        self.tp += int(((pred == 1) & (gt_b == 1)).sum())
        self.fp += int(((pred == 1) & (gt_b == 0)).sum())
        self.fn += int(((pred == 0) & (gt_b == 1)).sum())
        self.tn += int(((pred == 0) & (gt_b == 0)).sum())
        self.dice_num += 2.0 * float(((pred == 1) & (gt_b == 1)).sum())
        self.dice_den += float(pred.sum()) + float(gt_b.sum())

        for i in range(pred.shape[0]):
            inter = float(((pred[i] == 1) & (gt_b[i] == 1)).sum())
            union = float(((pred[i] == 1) | (gt_b[i] == 1)).sum())
            if union > 0:
                self.per_sample_iou.append(inter / union)
            # 只在"这张图真的有篡改区域"时统计 —— 这是篡改定位文献的标准口径
            if gt_b[i].sum() > 0:
                self.per_sample_iou_tampered.append(inter / union if union > 0 else 0.0)
                self.n_tampered += 1
            self.n += 1

    def compute(self) -> Dict[str, float]:
        eps = 1e-9
        iou = self.tp / max(eps, self.tp + self.fp + self.fn)
        dice = (2 * self.tp) / max(eps, 2 * self.tp + self.fp + self.fn)
        pixel_acc = (self.tp + self.tn) / max(eps, self.tp + self.tn + self.fp + self.fn)
        f1 = (2 * self.tp) / max(eps, 2 * self.tp + self.fp + self.fn)
        return {
            "miou": float(iou),
            "miou_per_sample": float(np.mean(self.per_sample_iou)) if self.per_sample_iou else 0.0,
            # 仅在含篡改区域的图上求平均（与 ManTra-Net / SPAN / CAT-Net 等可比）
            "miou_tampered_only": (
                float(np.mean(self.per_sample_iou_tampered))
                if self.per_sample_iou_tampered else 0.0
            ),
            "n_tampered": int(self.n_tampered),
            "pixel_acc": float(pixel_acc),
            "dice": float(dice),
            "f1": float(f1),
            "n_pixels": int(self.tp + self.tn + self.fp + self.fn),
        }


# ==========================================================================
def psnr_free_summary() -> Dict[str, str]:
    return {"note": "指标口径说明：acc/precision/recall/f1 基于阈值 0.5。"
                    "定位 mIoU 有三个口径，务必标明报的是哪一个："
                    "① miou = 全数据集像素汇总后算 IoU（池化，被大区域图主导，最宽松）；"
                    "② miou_tampered_only = 只在含篡改区域的图上逐图算 IoU 再平均"
                    "（篡改定位文献标准口径，与 ManTra-Net/SPAN/CAT-Net 可比）；"
                    "③ miou_per_sample = 在所有「预测或 GT 非空」的图上平均"
                    "（含被误报的真实图，IoU 记 0，最严格）。"}
