"""1.1 (3) 通道-空间交叉注意力自适应特征融合模块 (CS-CAM)

对应申报书公式 (13)~(17)：

    F_spa'  = Conv1x1(F_spa)   in R^512                 (13)
    F_freq' = Conv1x1(F_freq)  in R^512                 (14)
    F_spa''  = F_spa'  * CA(F_spa')  * SA(F_spa')       (15)
    F_freq'' = F_freq' * CA(F_freq') * SA(F_freq')      (16)
    F_fusion = F_spa'' + F_freq''  in R^512             (17)

其中 CA 为通道注意力（GAP + GMP -> 共享 MLP），SA 为空间注意力
（沿通道轴 GAP + GMP -> 拼接 -> 7x7 卷积 -> Sigmoid）。

----------------------------------------------------------------------
实现说明
----------------------------------------------------------------------
* 公式里 512 维写成向量，实际是 (B,512,14,14) 的特征图，1x1 卷积即通道映射。
* 通道注意力 MLP 的降维比 r=16（512 -> 32 -> 512），与 CBAM 一致。
* 空间注意力卷积核取 7x7（CBAM 原文设定），申报书未给核大小，此处按惯例取 7；
  若审阅要求严格，改 `spatial_kernel=3` 即可。
* 消融：`use_cross_attention=False` 时退化为 1x1 卷积后直接相加（无注意力加权）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """通道注意力 CA(·)：GAP/GMP -> 共享 MLP -> Sigmoid -> (B,C,1,1)。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared_mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.shared_mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """空间注意力 SA(·)：沿通道轴 GAP/GMP -> 拼接 -> 卷积 -> Sigmoid -> (B,1,H,W)。"""

    def __init__(self, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(2, 1, kernel, 1, pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


# --------------------------------------------------------------------------
class CSCAM(nn.Module):
    """通道-空间交叉注意力融合模块。"""

    def __init__(self, spa_dim: int = 896, freq_dim: int = 256, dim: int = 512,
                 use_cross_attention: bool = True, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.proj_spa = nn.Sequential(
            nn.Conv2d(spa_dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
        )
        self.proj_freq = nn.Sequential(
            nn.Conv2d(freq_dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
        )
        if use_cross_attention:
            self.ca_spa, self.sa_spa = ChannelAttention(dim, reduction), SpatialAttention(spatial_kernel)
            self.ca_freq, self.sa_freq = ChannelAttention(dim, reduction), SpatialAttention(spatial_kernel)
        self.dim = dim

    def forward(self, f_spa: torch.Tensor, f_freq: torch.Tensor) -> torch.Tensor:
        """入参为特征图 (B,896,14,14) / (B,256,14,14)，返回 (B,512,14,14)。"""
        p_spa = self.proj_spa(f_spa)      # 公式 (13)
        p_freq = self.proj_freq(f_freq)   # 公式 (14)

        if self.use_cross_attention:
            a_spa = p_spa * self.ca_spa(p_spa) * self.sa_spa(p_spa)     # 公式 (15)
            a_freq = p_freq * self.ca_freq(p_freq) * self.sa_freq(p_freq)  # 公式 (16)
        else:
            a_spa, a_freq = p_spa, p_freq

        return a_spa + a_freq             # 公式 (17)  F_fusion

    @torch.no_grad()
    def sanity_check(self) -> dict:
        self.eval()
        f_spa = torch.randn(2, 896, 14, 14)
        f_freq = torch.randn(2, 256, 14, 14)
        out = self.forward(f_spa, f_freq)
        return {"F_spa": tuple(f_spa.shape), "F_freq": tuple(f_freq.shape),
                "F_fusion": tuple(out.shape), "期望": "(2,512,14,14)"}
