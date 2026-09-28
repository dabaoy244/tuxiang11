"""1.2 分层 VIB 信息瓶颈特征提纯模块

对应申报书公式 (18)~(26)：

    L_VIB = -E_q(z|x)[log q(y|z)] + beta * KL(q(z|x) || r(z))          (19)
    mu    = MLP_mu(F_fusion)              in R^256                      (20)
    sigma = Softplus(MLP_sigma(F_fusion)) in R^256                      (21)
    Softplus(x) = log(1 + e^x)                                          (22)
    z     = mu + eps * sigma,  eps ~ N(0,1)     (重参数技巧)             (23)
    beta(t) = 0            , t < 20
            = 0.1*(t-20)/20, 20 <= t < 40                               (24)
            = 0.1          , t >= 40
    KL = clip(KL(q(z|x) || N(0,1)), 0, 10)                              (25)

三重稳定训练策略：beta 退火 / KL 散度裁剪 / 梯度裁剪(L2<=5.0，见 engine/trainer)。
"分层" 的含义：VIB 只挂在**分类支路的最后一层**，不作用于底层共享特征与定位支路，
从根本上避免压缩细粒度空间信息。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
def beta_schedule(epoch: int, beta_max: float = 0.1,
                  warm_start: int = 20, warm_end: int = 40) -> float:
    """公式 (24) 的分段 beta 退火（此处实为"分段升温"）。"""
    if epoch < warm_start:
        return 0.0
    if epoch < warm_end:
        return beta_max * (epoch - warm_start) / max(1, (warm_end - warm_start))
    return beta_max


# --------------------------------------------------------------------------
class HierarchicalVIB(nn.Module):
    """分层变分信息瓶颈：512 -> 256 -> (mu, log_sigma)，重参数采样得 z ∈ R^256。

    返回 (z, kl, mu, sigma)，KL 已按公式 (25) 裁剪到 [kl_min, kl_max]。

    ------------------------------------------------------------------
    工程要点 1：σ 的初始化
    ------------------------------------------------------------------
    KL(N(μ,σ²) || N(0,1)) = 0.5 * Σ (μ² + σ² − log σ² − 1)。
    若把 σ 初始化得很小（例如 0.1），−log σ² 项会让 KL 起步就有几百，
    一上来就被裁到上限，反向梯度直接被 clamp 吃掉 —— 表现为"VIB 完全不工作"。
    正确做法：σ 初值取 1（此时 μ=0、σ=1 时 KL=0，是 KL 的全局最小值），
    再让模型自己学。Softplus(b) = 1  =>  b = log(e^1 − 1) ≈ 0.5413。

    ------------------------------------------------------------------
    工程要点 2：KL 裁剪不能杀掉梯度（与公式 25 的重要澄清）
    ------------------------------------------------------------------
    `torch.clamp` 在区间外梯度为 0。若 KL 常态性地 > 10，公式 (25) 就会把
    VIB 正则项变成"常数"，等于**静默关闭信息瓶颈**，而且论文里完全看不出来。
    本实现提供两种口径，默认选更稳妥的组合：
      * kl_reduction = "mean"（对 batch 和 latent 维度都取均值）
        —— 数值落在 0.0x ~ 几 的量级，此时 [0,10] 的裁剪是"安全阀"而非"常态"，
           与公式 (25) 的字面语义一致；
      * kl_reduction = "sum"（对 latent 维求和、对 batch 取均值）
        —— 即 Alemi 原始 VIB 写法，数值常在几十到几百，[0,10] 会常态触发；
      * kl_clip_grad_through = True（默认）：前向仍是硬裁剪（数值满足公式 25），
           反向按恒等传递，保证 KL 始终有梯度、VIB 不会静默失效。
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, latent_dim: int = 256,
                 kl_clip_min: float = 0.0, kl_clip_max: float = 10.0,
                 kl_reduction: str = "mean", kl_clip_grad_through: bool = True):
        super().__init__()
        # 第一层：512 -> 256（公式 (20)(21) 共用的第一层 MLP）
        self.fc_shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # 第二层分别输出 mu 与 sigma（sigma 用 Softplus 保证非负）
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)
        self.softplus = nn.Softplus()

        self.kl_clip_min = kl_clip_min
        self.kl_clip_max = kl_clip_max
        self.kl_reduction = kl_reduction
        self.kl_clip_grad_through = kl_clip_grad_through
        self.latent_dim = latent_dim

        # 初始化：μ 初值为 0；σ 初值为 1（Softplus(b)=1 => b=log(e-1)≈0.5413），
        # 这样 KL 从 0 起步、稳步增长，避免"一上来就被裁到上限"。
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.normal_(self.fc_sigma.weight, std=0.01)
        nn.init.constant_(self.fc_sigma.bias, math.log(math.e - 1.0))

    # ------------------------------------------------------------------
    def encode(self, f_fusion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """F_fusion 可以是 (B,512)、也可以是 (B,512,14,14)（内部自动 GAP）。"""
        if f_fusion.dim() == 4:
            f_fusion = f_fusion.mean(dim=(2, 3))
        h = self.fc_shared(f_fusion)
        mu = self.fc_mu(h)                                   # 公式 (20)
        sigma = self.softplus(self.fc_sigma(h)) + 1e-4       # 公式 (21)(22)
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor,
                       sample: bool = True) -> torch.Tensor:
        """公式 (23) 重参数技巧；eval 阶段可用 sample=False 取 mu（确定性推理）。"""
        if not sample:
            return mu
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """KL(N(mu,sigma^2) || N(0,1))。

        kl_reduction="mean" -> 对 batch 与 latent 维都取均值（默认，配合 [0,10] 裁剪）
        kl_reduction="sum"  -> 对 latent 维求和、对 batch 取均值（Alemi 原始写法）
        """
        kl = 0.5 * (mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + 1e-8) - 1.0)
        if self.kl_reduction == "sum":
            return kl.sum(dim=-1).mean()
        return kl.mean()

    def _clip_kl(self, kl: torch.Tensor) -> torch.Tensor:
        """公式 (25) KL 裁剪 [0,10]。

        数值上做硬裁剪（满足公式语义）；`kl_clip_grad_through=True` 时反向
        按恒等传递，避免 KL 常态化超上限后 VIB 静默失效。
        """
        clipped = torch.clamp(kl, self.kl_clip_min, self.kl_clip_max)
        if self.kl_clip_grad_through:
            clipped = kl + (clipped - kl).detach()
        return clipped

    def forward(self, f_fusion: torch.Tensor, sample: bool = True,
                beta: float = 0.0) -> dict:
        mu, sigma = self.encode(f_fusion)
        z = self.reparameterize(mu, sigma, sample)
        kl_raw = self.kl_divergence(mu, sigma)
        kl = self._clip_kl(kl_raw)          # 公式 (25)
        return {"z": z, "mu": mu, "sigma": sigma, "kl_raw": kl_raw, "kl": kl,
                "beta": beta, "kl_term": beta * kl}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sanity_check(self) -> dict:
        self.eval()
        x = torch.randn(4, 512, 14, 14)
        out = self.forward(x, sample=True, beta=0.1)
        return {
            "mu": tuple(out["mu"].shape),
            "sigma": tuple(out["sigma"].shape),
            "z": tuple(out["z"].shape),
            "sigma_min/max": (float(out["sigma"].min()), float(out["sigma"].max())),
            "kl_raw": float(out["kl_raw"]),
            "kl_clipped": float(out["kl"]),
            "beta(10)/(30)/(50)": (
                beta_schedule(10), beta_schedule(30), beta_schedule(50)
            ),
        }
