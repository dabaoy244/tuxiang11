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
    # ★★★ 2026-10-11：**硬闸 —— 只接受二值标签**。
    #   为什么必须加：本函数按 pos=(y==1) / neg=(y==0) 取两侧，但**秩是在全量样本上排的**。
    #   若喂进多类标签（如 0..3 的四类），y=2/3 的样本**既不计入分子也不计入分母，
    #   却参与了排名** ⇒ 分子被抬高、分母不变 ⇒ **静默算出 AUC > 1**。
    #   四类判别训练（2026-10-10）就是这么被坑的：20 轮里有 8 轮 val_cls_auc > 1
    #   （最大 1.1605），而训练照跑、best.pt 照选 —— 全程没有任何报错。
    #   多类请用 `MulticlassMetrics`（OvR 宏平均）。
    _u = np.unique(y)
    if _u.size and not np.all(np.isin(_u, (0, 1))):
        raise ValueError(
            f"roc_auc 只接受二值标签 {{0,1}}，实收 {_u[:8].tolist()}"
            f"（共 {_u.size} 种值）。多类标签会让本函数**静默**产出 AUC>1。"
            f"多类请改用 MulticlassMetrics 的 macro_auc_ovr。")
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
    # ★ 同 roc_auc：只接受二值标签。多类标签下 `y==1` 只挑出一类、
    #   而 `cumsum(y==0)` 把 2/3 类也算成"负" ⇒ PR 曲线静默走样。
    _u = np.unique(y)
    if _u.size and not np.all(np.isin(_u, (0, 1))):
        raise ValueError(
            f"average_precision 只接受二值标签 {{0,1}}，实收 {_u[:8].tolist()}。"
            f"多类请改用 MulticlassMetrics。")
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
class MulticlassMetrics:
    """**多类**分类指标累加器（四类判别用：0 真实 / 1 AI整幅生成 / 2 AI局部编辑 / 3 传统篡改）。

    ★ 为什么**不**复用 `ClassificationMetrics`（2026-10-11 的教训）：
      后者是二分类实现 —— `update` 只取 `softmax(...)[:, 1]`、`roc_auc` 又按
      `pos=(y==1)` / `neg=(y==0)` 取两侧。把四类标签喂进去，AUC 会**静默超过 1**
      （实测 20 轮里 8 轮 >1，最大 1.1605），而训练照跑、best.pt 照选、**全程无报错**。
      所以多类必须走独立实现，并在 `roc_auc`/`average_precision` 里加硬闸拦住误用。

    主口径与二分类保持一致：**宏平均（先逐类算再平均）+ 阈无关**。
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None) -> None:
        self.C = int(num_classes)
        self.names = list(class_names) if class_names else \
            ['类%d' % i for i in range(self.C)]
        self.reset()

    def reset(self) -> None:
        self._y: List[np.ndarray] = []
        self._p: List[np.ndarray] = []

    def update(self, logits_or_prob: np.ndarray, labels: np.ndarray) -> None:
        """logits_or_prob: (N,C)。logits 或概率自动识别。"""
        arr = np.asarray(logits_or_prob, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.C:
            raise ValueError(
                f"MulticlassMetrics 期望 (N,{self.C})，收到 {arr.shape}")
        # 识别：概率非负且行和≈1；否则当 logits 走 softmax
        rs = arr.sum(axis=1, keepdims=True)
        is_prob = bool(np.all(arr >= -1e-9) and
                       np.allclose(rs.ravel(), 1.0, atol=1e-3))
        if is_prob:
            p = np.clip(arr, 0.0, 1.0)
        else:
            e = np.exp(arr - arr.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
        self._p.append(p)
        y = np.asarray(labels).ravel().astype(np.int64)
        bad = y[(y < 0) | (y >= self.C)]
        if bad.size:
            raise ValueError(
                f"标签越界：期望 0..{self.C - 1}，实收 {sorted(set(bad.tolist()))[:8]}。"
                f"多类标签必须 0-based（交叉熵同此约定）。")
        self._y.append(y)

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, object]:
        if not self._y:
            return {}
        y = np.concatenate(self._y)
        P = np.concatenate(self._p, axis=0)
        pred = P.argmax(axis=1)
        n = int(y.size)

        cm = np.zeros((self.C, self.C), dtype=np.int64)
        for t, q in zip(y, pred):
            cm[t, q] += 1

        per: Dict[str, Dict[str, float]] = {}
        f1s, aucs = [], []
        for c in range(self.C):
            tp = int(cm[c, c])
            fn = int(cm[c, :].sum() - tp)
            fp = int(cm[:, c].sum() - tp)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
            # OvR AUC：该类为正、其余为负（阈无关）
            yb = (y == c).astype(np.int64)
            auc = roc_auc(yb, P[:, c]) if 0 < int(yb.sum()) < n else float("nan")
            per[self.names[c]] = {
                "n": int((y == c).sum()), "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1, "auc_ovr": auc,
            }
            if not np.isnan(f1):
                f1s.append(f1)
            if not np.isnan(auc):
                aucs.append(auc)

        acc = float((pred == y).mean())
        return {
            "n": n,
            "acc": acc,
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "macro_auc_ovr": float(np.mean(aucs)) if aucs else float("nan"),
            "confusion": cm.tolist(),
            "per_class": per,
            "class_names": self.names,
        }

    def format(self) -> str:
        r = self.compute()
        if not r:
            return "(无样本)"
        L = ["多类指标（n=%d）  acc=%.4f  **macro_f1=%.4f**  **macro_auc_ovr=%.4f**"
             % (r["n"], r["acc"], r["macro_f1"], r["macro_auc_ovr"]),
             "%-16s %7s %7s %8s %8s %8s %9s" %
             ("类", "n", "tp", "P", "R", "F1", "AUC(OvR)")]
        for nm in r["class_names"]:
            d = r["per_class"][nm]
            L.append("%-16s %7d %7d %8.4f %8.4f %8.4f %9.4f"
                     % (nm, d["n"], d["tp"], d["precision"], d["recall"],
                        d["f1"], d["auc_ovr"]))
        L.append("混淆矩阵（行=真值，列=预测）：")
        L.append("           " + "".join("%9s" % n[:8] for n in r["class_names"]))
        for i, nm in enumerate(r["class_names"]):
            L.append("%-10s " % nm[:8] +
                     "".join("%9d" % v for v in r["confusion"][i]))
        return "\n".join(L)


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

    #: ★ 像素级 AUC 的直方图分箱数。
    #:   为什么用直方图而不是把所有像素存下来：CASIAv2 test 一跑就是 1263×224² ≈ 6.3e7 个像素，
    #:   存 float32 要 250MB+；分箱后只占 2×65536 个计数（≈1MB），且**误差上界 ~1/65536 ≈ 1.5e-5**
    #:   —— 比我们关心的效应量（1~2pp）小三个数量级。
    #:   同一箱内的像素按"并列"处理、各给 0.5 权重，这与本文件分类 AUC 的**秩平均**约定一致，
    #:   也保住"分数全同 ⇒ AUC=0.5"这条不变量（见 `scripts/test_metric_definitions.py`）。
    _PAUC_BINS = 65536

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.dice_num = self.dice_den = 0.0
        self.per_sample_iou: List[float] = []
        self.per_sample_iou_tampered: List[float] = []
        self.n = 0
        self.n_tampered = 0
        # 像素级 AUC 的两个直方图（正类=篡改像素、负类=真实像素）
        self._pauc_pos = np.zeros(self._PAUC_BINS, dtype=np.int64)
        self._pauc_neg = np.zeros(self._PAUC_BINS, dtype=np.int64)

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

        # ★ 像素级 AUC 累加（不依赖阈值；与 ManTra-Net / SPAN 的 pixel-level AUC 同义）
        flat_p = np.clip(np.asarray(p, dtype=np.float64).ravel(), 0.0, 1.0)
        flat_g = np.asarray(g, dtype=np.float64).ravel() > 0.5
        bins = np.minimum((flat_p * (self._PAUC_BINS - 1)).astype(np.int64),
                          self._PAUC_BINS - 1)
        self._pauc_pos += np.bincount(bins[flat_g], minlength=self._PAUC_BINS)
        self._pauc_neg += np.bincount(bins[~flat_g], minlength=self._PAUC_BINS)

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
            # ★ 像素级 AUC（阈无关）—— ManTra-Net / SPAN 的**第一位指标**，本仓库此前缺这一项。
            #   与分类 AUC 同约定：并列按秩平均（同箱各给 0.5）⇒ 分数全同 = 0.5。
            "pixel_auc": self._pixel_auc(),
        }

    def _pixel_auc(self) -> float:
        """由两个直方图算像素级 AUC = P(随机取一个篡改像素, 其分数 > 随机一个真实像素)。"""
        pos, neg = self._pauc_pos, self._pauc_neg
        P, N = int(pos.sum()), int(neg.sum())
        if P == 0 or N == 0:
            return float("nan")          # 没有篡改像素（或没有真实像素）⇒ AUC 无定义
        cum_neg_below = np.concatenate(([0], np.cumsum(neg)))[:-1]   # 严格低于该箱的负类数
        num = float((pos * (cum_neg_below + 0.5 * neg)).sum())
        return num / (P * N)


# ==========================================================================
def psnr_free_summary() -> Dict[str, str]:
    return {"note": "指标口径说明：acc/precision/recall/f1 基于阈值 0.5。"
                    "定位 mIoU 有三个口径，务必标明报的是哪一个："
                    "① miou = 全数据集像素汇总后算 IoU（池化，被大区域图主导，最宽松）；"
                    "② miou_tampered_only = 只在含篡改区域的图上逐图算 IoU 再平均"
                    "（篡改定位文献标准口径，与 ManTra-Net/SPAN/CAT-Net 可比）；"
                    "③ miou_per_sample = 在所有「预测或 GT 非空」的图上平均"
                    "（含被误报的真实图，IoU 记 0，最严格）。"
                    "另：pixel_auc = 像素级 AUC（阈无关，直方图分箱实现，误差 ~1.5e-5），"
                    "是 ManTra-Net / SPAN 报的第一位指标；与 mIoU 不是一回事，两者都要报。"}
