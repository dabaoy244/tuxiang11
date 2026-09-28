"""1.4 不确定性加权多阶段联合损失函数

对应申报书公式 (26)(29)(30)(31)：

    L_VIB = -(1/N) * sum_i log q(y_i | z_i) + beta(t) * KL                (26)
           其中 KL 已按公式 (25) 裁剪到 [0,10]

    L_edge = BCE(M_edge_pred, M_edge_gt)                                  (29)

    L' = (L - mu_L) / sigma_L          # Z-score 归一化，滑动窗口 100 批次      (30)

    L_total = 1/(2*sigma1^2) * L'_VIB  + 1/(2*sigma2^2) * L'_BCE
            + 1/(2*sigma3^2) * L'_Dice + 1/(2*sigma4^2) * L'_edge
            + log(sigma1*sigma2*sigma3*sigma4)                            (31)

----------------------------------------------------------------------
实现细节说明
----------------------------------------------------------------------
1. 数值稳定的参数化：令 s_i = log(sigma_i^2)，则
      1/(2 sigma_i^2) = 0.5 * exp(-s_i)
      log(sigma1*sigma2*sigma3*sigma4) = 0.5 * sum_i s_i
   于是 L_total = sum_i [ 0.5*exp(-s_i) * L'_i + 0.5*s_i ]，全程无除法、
   无开方，且梯度有界（这是 Kendall 2018 的原始写法）。
2. Z-score 归一化用"滑动窗口统计量"，且**统计量 detach**（只让分子回传梯度）。
   否则归一化的分母也会被优化，导致损失尺度失控。
   窗口默认 100 个批次，用 deque 实现，同时用 EMA 兜底（窗口未满时）。
3. 若 `uncertainty_weighting=False`，退化为固定权重版（消融基线）。
4. `terms` 控制当前阶段参与计算的损失项（三阶段训练按阶段启用）。
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.mobile_unetv2 import EdgeHead


# ==========================================================================
# 基础损失项
# ==========================================================================
def vib_loss(cls_logits: torch.Tensor, labels: torch.Tensor,
             kl: torch.Tensor, beta: float) -> torch.Tensor:
    """公式 (26)：交叉熵（= -log q(y|z)）+ beta * KL。"""
    ce = F.cross_entropy(cls_logits, labels)
    return ce + beta * kl


def bce_loss(mask_logits: torch.Tensor, mask_gt: torch.Tensor,
             valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """逐像素 BCE。`valid` (B,) 为 bool 时，只对含有效掩码的样本求平均。

    混合批次（ForenSynths 无掩码 + CASIA 有掩码）必须用 valid 过滤，
    否则补零掩码会把定位支路教成"全部预测为真实"。
    """
    per_sample = F.binary_cross_entropy_with_logits(
        mask_logits, mask_gt, reduction="none"
    ).flatten(1).mean(dim=1)
    if valid is not None:
        if valid.sum() == 0:
            return mask_logits.sum() * 0.0
        return per_sample[valid].mean()
    return per_sample.mean()


def dice_loss(mask_logits: torch.Tensor, mask_gt: torch.Tensor, smooth: float = 1.0,
              valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """1 - Dice 系数（逐样本计算后取均值，避免大目标主导）。"""
    prob = torch.sigmoid(mask_logits)
    b = prob.shape[0]
    p = prob.reshape(b, -1)
    g = mask_gt.reshape(b, -1)
    inter = (p * g).sum(dim=1)
    denom = p.sum(dim=1) + g.sum(dim=1)
    dice = (2 * inter + smooth) / (denom + smooth)
    per_sample = 1.0 - dice
    if valid is not None:
        if valid.sum() == 0:
            return mask_logits.sum() * 0.0
        return per_sample[valid].mean()
    return per_sample.mean()


def edge_loss(edge_logits: torch.Tensor, edge_gt: torch.Tensor,
              valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """公式 (29)：边缘二元交叉熵。"""
    per_sample = F.binary_cross_entropy_with_logits(
        edge_logits, edge_gt, reduction="none"
    ).flatten(1).mean(dim=1)
    if valid is not None:
        if valid.sum() == 0:
            return edge_logits.sum() * 0.0
        return per_sample[valid].mean()
    return per_sample.mean()


# ==========================================================================
# 公式 (30) Z-score 归一化
# ==========================================================================
class ZScoreNormalizer:
    """按损失项维护滑动窗口统计量，实现 L' = (L - mu_L) / sigma_L。

    注意：窗口未满时用 Welford 在线均值/方差兜底，保证第一步就能归一化。
    """

    def __init__(self, terms: Iterable[str], window: int = 100, eps: float = 1e-6):
        self.terms = list(terms)
        self.window = max(2, window)
        self.eps = eps
        self.history: Dict[str, deque] = {t: deque(maxlen=self.window) for t in self.terms}
        self._ema_mean: Dict[str, float] = {t: None for t in self.terms}
        self._ema_var: Dict[str, float] = {t: None for t in self.terms}
        self._count: int = 0

    @torch.no_grad()
    def _update(self, name: str, value: float) -> None:
        self.history[name].append(value)
        # EMA 兜底（alpha=0.05），窗口满后 EMA 与窗口统计量混合
        if self._ema_mean[name] is None:
            self._ema_mean[name] = value
            self._ema_var[name] = 0.0
        else:
            a = 0.05
            diff = value - self._ema_mean[name]
            self._ema_mean[name] += a * diff
            self._ema_var[name] = (1 - a) * (self._ema_var[name] + a * diff * diff)

    def stats(self, name: str) -> tuple:
        """返回 (mu, sigma)。

        某个损失项**从未在当前阶段被更新过**时（例如阶段一只训练分类，
        bce/dice/edge 的统计量为空），退化为恒等归一化 (0, 1)，
        而不是抛异常 —— 否则三阶段训练在第一次验证时就会崩。
        """
        hist = self.history.get(name)
        if hist and len(hist) >= max(4, self.window // 4):
            t = torch.tensor(list(hist), dtype=torch.float32)
            return float(t.mean()), float(t.std(unbiased=False).clamp_min(self.eps))
        if self._ema_mean.get(name) is None:
            return 0.0, 1.0
        return float(self._ema_mean[name]), float(max(self._ema_var[name] ** 0.5, self.eps))

    def normalize(self, name: str, value: torch.Tensor, update: bool = True) -> torch.Tensor:
        """返回归一化后的损失（保留 value 的梯度，统计量分离）。"""
        if update and self.training_enabled:
            self._update(name, float(value.detach()))
        mu, sigma = self.stats(name)
        return (value - mu) / sigma

    training_enabled: bool = True

    def state_dict(self) -> dict:
        return {f"hist_{k}": list(v) for k, v in self.history.items()}

    def load_state_dict(self, sd: dict) -> None:
        for k, v in sd.items():
            if k.startswith("hist_"):
                name = k[len("hist_"):]
                if name in self.history:
                    self.history[name] = deque(v, maxlen=self.window)


# ==========================================================================
# 公式 (31) 同方差不确定性加权
# ==========================================================================
class UncertaintyWeighting(nn.Module):
    """可学习不确定性参数。s_i = log(sigma_i^2) 初始化为 0（即 sigma=1，等权起步）。

    ⚠ 为什么必须钳位 s_i
    --------------------
    L_total = Σ [ 0.5*exp(-s_i)*L'_i + 0.5*s_i ]，把 s_i 看成自变量求导：
        dL/ds_i = -0.5*exp(-s_i)*L'_i + 0.5 = 0  =>  s_i* = ln(L'_i)
    **只有当 L'_i > 0 时才有极小值**。而公式 (30) 的 Z-score 归一化把每项损失
    中心化到 0 附近，约一半批次会出现 L'_i < 0；此时 dL/ds_i 恒为正，
    即 L 关于 s_i 单调递增 → 最优解是 s_i → -∞，损失无下界，
    训练会表现为「日志里的 total loss 一路向负无穷漂、权重发散」。
    这就是申报书公式 (30) 与 (31) 直接串联时的数学不一致之处。

    这里给 s_i 加一个对称区间钳位作为工程保护：越界后 clamp 的梯度为 0，
    参数自动停住，训练不会发散。默认 [-4, 4] 对应 sigma ∈ [0.135, 7.39]，
    权重 0.5*exp(±4) ∈ [0.009, 27.3]，覆盖了正常需要的动态范围。
    若要彻底消除不一致，应把加权作用在**未归一化的非负原始损失**上，
    只把 Z-score 用于日志展示（见 docs/08）。
    """

    def __init__(self, terms: List[str], s_min: float = -4.0, s_max: float = 4.0):
        super().__init__()
        self.terms = list(terms)
        self.log_var = nn.Parameter(torch.zeros(len(self.terms)))
        self.s_min, self.s_max = float(s_min), float(s_max)
        self._clamped = False

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ⚠ 调用方必须保证 losses 非空。本函数早先以 `total = 0.0` 起手，
        #   一旦一项都没匹配上就会把 Python 浮点当成 total 返回，
        #   调用方 `loss.backward()` 直接 AttributeError（见 docs/08 D40）。
        #   这里改成 None 起手 + 末尾显式报错，把"空输入"从一个**静默的浮点**
        #   变成**带调用栈的错误**，同时把"整批无监督"的正常业务情形
        #   收敛到 MultiTaskLoss.forward 里统一处理（返回零张量 + empty=True）。
        total = None
        detail = {}
        for i, t in enumerate(self.terms):
            if t not in losses:
                continue
            s_raw = self.log_var[i]
            s = torch.clamp(s_raw, self.s_min, self.s_max)
            if not self._clamped and float(s_raw.detach()) != float(s.detach()):
                self._clamped = True      # 只提示一次，避免刷屏
            w = 0.5 * torch.exp(-s)
            reg = 0.5 * s
            term = w * losses[t] + reg
            total = term if total is None else total + term
            detail[f"w_{t}"] = float(w.detach())
            detail[f"sigma_{t}"] = float(torch.exp(0.5 * s).detach())
        if total is None:
            raise ValueError(
                f"UncertaintyWeighting 没有匹配到任何损失项：terms={self.terms}，"
                f"实际传入={sorted(losses)}。空字典请由 MultiTaskLoss 处理"
                f"（应返回零张量并把 empty 置 True），不要传到这里。")
        return {"total": total, "detail": detail, "clamped": self._clamped}

    @torch.no_grad()
    def weights(self) -> Dict[str, float]:
        return {t: float(0.5 * torch.exp(-torch.clamp(self.log_var[i],
                                                      self.s_min, self.s_max)))
                for i, t in enumerate(self.terms)}


# ==========================================================================
# 空监督批次的安全零张量
# ==========================================================================
def zero_loss(outputs: Dict[str, object]) -> torch.Tensor:
    """返回一个**挂在计算图上**的 0 张量（可安全 backward，梯度全零）。

    为什么必须挂在图上而不是用 `torch.zeros(())`
    --------------------------------------------
    本函数的唯一用途是"这一批没有任何可用的监督项"时充当 total loss。
    调用方（Trainer）紧接着要执行 `loss.backward()`，所以返回值必须
    有 `grad_fn`；直接返回 Python 浮点 `0.0` 或 `torch.zeros(())` 都会在
    `backward()` 处炸掉（前者 AttributeError，后者 RuntimeError）。

    也不能图省事返回 `None` 让调用方判空 —— 那要求每一处调用点都记得判，
    漏一处就是又一次"训练跑到一半突然死"，而这类死法在云端要花 100 分钟才发现。
    所以这里坚持"永远返回张量"，把判定收敛到 `empty` 这一个布尔标志上。
    """
    for v in outputs.values():
        if isinstance(v, torch.Tensor):
            return v.sum() * 0.0
    # 理论上到不了这里（outputs 至少含 cls_logits）；仅作兜底
    return torch.zeros((), requires_grad=True)


# ==========================================================================
# 组合损失
# ==========================================================================
class MultiTaskLoss(nn.Module):
    """多任务联合损失：归一化 -> 不确定性加权（或固定权重）。"""

    def __init__(self, cfg: dict):
        super().__init__()
        lcfg = cfg["loss"]
        self.terms = list(lcfg.get("terms", ["vib", "bce", "dice", "edge"]))
        self.norm_mode = lcfg.get("normalization", "zscore")
        self.normalizer = ZScoreNormalizer(self.terms, lcfg.get("zscore_window", 100))
        self.use_uncertainty = lcfg.get("uncertainty_weighting", True)
        self.fixed_weights = lcfg.get("weights", {})
        self.dice_smooth = lcfg.get("dice_smooth", 1.0)

        if self.use_uncertainty:
            self.weighting = UncertaintyWeighting(
                self.terms,
                s_min=float(lcfg.get("uncertainty_s_min", -4.0)),
                s_max=float(lcfg.get("uncertainty_s_max", 4.0)))
        else:
            self.weighting = None

    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask_gt: Optional[torch.Tensor],
        beta: float = 0.0,
        active_tasks: Optional[List[str]] = None,
        edge_gt: Optional[torch.Tensor] = None,
        update_norm: bool = True,
        mask_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """计算总损失。

        Args:
            outputs: 模型前向输出
            labels: (B,) 0=真实 1=伪造
            mask_gt: (B,1,H,W) 像素级 GT，None 时跳过定位相关损失
            beta: 当前轮次的 beta(t)（公式 24）
            active_tasks: 当前阶段启用的任务，如 ["cls"] / ["loc","edge"] / 全部
            edge_gt: 预先算好的边缘 GT；为 None 时用 EdgeHead.canny_edge 在线提取
            update_norm: 是否更新归一化统计量（验证时置 False）
            mask_valid: (B,) bool，标记哪些样本的掩码真实有效（混合批次必需）
        """
        active = set(active_tasks) if active_tasks else {"cls", "loc", "edge"}
        raw: Dict[str, torch.Tensor] = {}
        skipped: Dict[str, str] = {}

        if "cls" in active:
            raw["vib"] = vib_loss(outputs["cls_logits"], labels, outputs["kl"], beta)
        else:
            skipped["vib"] = "本阶段未启用分类任务"

        if mask_gt is not None and "loc" in active and (
            mask_valid is None or bool(mask_valid.any())
        ):
            raw["bce"] = bce_loss(outputs["mask_logits"], mask_gt, mask_valid)
            raw["dice"] = dice_loss(outputs["mask_logits"], mask_gt, self.dice_smooth, mask_valid)
        else:
            skipped["bce"] = skipped["dice"] = "本阶段未启用定位任务或数据无掩码"

        if ("edge" in active and "edge_logits" in outputs and mask_gt is not None
                and (mask_valid is None or bool(mask_valid.any()))):
            if edge_gt is None:
                edge_gt = EdgeHead.canny_edge(mask_gt)
            raw["edge"] = edge_loss(outputs["edge_logits"], edge_gt, mask_valid)
        else:
            skipped["edge"] = "本阶段未启用边缘任务 / 无掩码 / 模型未含边缘头"

        # ---- 公式 (30) 归一化
        normed: Dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            if self.norm_mode == "zscore":
                normed[k] = self.normalizer.normalize(k, v, update=update_norm)
            else:
                normed[k] = v

        # ---- 公式 (31) 加权
        #
        # ★ 空监督批次（2026-09-25 云上 stage2 第 1 个 epoch 实测踩到）
        #   阶段二只启用 loc/edge，这两个任务的监督**全部来自像素掩码**；
        #   而训练集是 ForenSynths(整图真伪，无掩码) + CASIAv2/COVERAGE(带掩码)
        #   的混合集，掩码样本占比约 13%，batch=16 时约有 10% 的批次
        #   **一张带掩码的样本都没有**（batch 内 iid 抽样，(1-0.133)^16≈10%）。
        #   此时 raw 为空。旧实现里 UncertaintyWeighting.forward 的 `total = 0.0`
        #   一直没被替换成张量，于是 total 是个 Python 浮点，
        #   训练循环走到 `loss.backward()` 抛
        #       AttributeError: 'float' object has no attribute 'backward'
        #   —— 修复早停 bug 后**第一次真正跑进 stage2**，就撞上了它，
        #      整个训练在 103 分钟处直接死掉。
        #   现在：total 恒为张量（backward 安全、梯度全零），
        #   并用 empty=True 让训练循环跳过参数更新、且不把 0 计入平均损失。
        if not normed:
            total = zero_loss(outputs)
            weight_detail = {}
        elif self.use_uncertainty:
            out = self.weighting(normed)
            total = out["total"]
            weight_detail = out["detail"]
        else:
            total = zero_loss(outputs)
            weight_detail = {}
            for k, v in normed.items():
                w = float(self.fixed_weights.get(k, 1.0))
                total = total + w * v
                weight_detail[f"w_{k}"] = w

        return {
            "total": total,
            # 本批次是否"没有任何启用中的监督项"（如整批无掩码）——
            # 训练循环必须据此跳过 opt.step()，否则这一步是纯噪声。
            "empty": not raw,
            "raw": {k: float(v.detach()) for k, v in raw.items()},
            "normed": {k: float(v.detach()) for k, v in normed.items()},
            "weights": weight_detail,
            "skipped": skipped,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def weights(self) -> Dict[str, float]:
        if self.weighting is not None:
            return self.weighting.weights()
        return {k: float(v) for k, v in self.fixed_weights.items()}


def build_loss(cfg: dict) -> MultiTaskLoss:
    return MultiTaskLoss(cfg)
