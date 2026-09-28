"""1.1 (2) 频域特征提取分支 —— 幅值-相位联合表征 + 可学习频域掩码

对应申报书公式 (4)~(12)：

    F(u,v)      = DFT(I(x,y))                                   (4)
    A(u,v)      = |F_shift(u,v)|        P(u,v) = arg(F_shift)   (5)(6)
    M(u,v)      = sigmoid(Conv2(Conv1(A)))  in [0,1]            (7)  可学习频域掩码
    A' = A ⊙ M , P' = P ⊙ M                                     (8)(9)
    F_amp   = GAP(CNN_amp(A'))   in R^128                       (10)
    F_phase = GAP(CNN_phase(P')) in R^128                       (11)
    F_freq  = Concat(F_amp, F_phase) in R^256                   (12)

----------------------------------------------------------------------
实现要点（工程细节，直接影响能否复现）
----------------------------------------------------------------------
1. 输入先转灰度（3 通道平均），保证 DFT 得到单通道二维频谱；
   若直接用 RGB，则幅值/相位需要分别对 3 通道处理，参数量 x3，工程上不划算。
2. 2D-DFT 等价于 `torch.fft.fft2(x, norm='ortho')`，移频用 `torch.fft.fftshift`。
3. 幅值谱动态范围极大（对数尺度上 1e-3 ~ 1e3），必须先做对数压缩
   `log(1+A)` 再送入掩码网络，否则 Sigmoid 掩码会被少数高频点主导。
4. 相位 [-pi, pi] 直接送入网络；为便于 OpenVINO 导出，用
   `atan2(sin, cos)` 而不是 `angle`（前者算子更通用，且避免复数类型）。
   实现上：`phase = torch.atan2(imag, real)`，全程只用实数张量。
5. 掩码网络只有 2 层、通道 1->32->1，输出 (B,1,224,224)，广播到 A、P 上。
6. 双路特征提取网络把 224x224 下采样到 14x14（stride 4 + 4），保证与
   空域分支空间尺寸一致；对 (B,128,14,14) 做 GAP 即得公式 (10)(11) 的 128 维向量。

消融开关（对应申报书消融实验）：
    use_phase=False          -> 仅用幅值，F_freq in R^128（验证"频域信息利用率提升 100%"）
    use_learnable_mask=False -> 用固定中频掩码替代可学习掩码
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dft import DFT2D


# --------------------------------------------------------------------------
class LearnableFrequencyMask(nn.Module):
    """2 层卷积生成自适应频域掩码（公式 7）。

    Conv(1->32, 5x5, s1, p2) -> ReLU -> Conv(32->1, 5x5, s1, p2) -> Sigmoid
    """

    def __init__(self, hidden: int = 32, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(1, hidden, kernel, 1, pad)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, 1, kernel, 1, pad)
        self.act = nn.ReLU(inplace=True)

    def forward(self, log_amp: torch.Tensor) -> torch.Tensor:
        m = self.conv2(self.act(self.bn1(self.conv1(log_amp))))
        return torch.sigmoid(m)          # (B,1,H,W) ∈ [0,1]


class FixedBandMask(nn.Module):
    """固定中频掩码（消融基线）：保留归一化半径 r∈[0.15, 0.55] 的环形频带。"""

    def __init__(self, low: float = 0.15, high: float = 0.55):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, log_amp: torch.Tensor) -> torch.Tensor:
        b, _, h, w = log_amp.shape
        yy = torch.linspace(-1, 1, h, device=log_amp.device).view(-1, 1).expand(h, w)
        xx = torch.linspace(-1, 1, w, device=log_amp.device).view(1, -1).expand(h, w)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        m = ((r >= self.low) & (r <= self.high)).float()
        return m.view(1, 1, h, w).expand(b, 1, h, w)


class FreqPathNet(nn.Module):
    """单路频域特征提取：2 层 3x3 卷积 32->128，两次 stride=2 下采样到 14x14。

    对应公式 (10)(11) 中 CNN_amp / CNN_phase（结构相同）。
    """

    def __init__(self, in_ch: int = 1, channels: List[int] = (32, 128), out_hw: int = 14):
        super().__init__()
        chs = [in_ch] + list(channels)
        layers = []
        for i in range(len(channels)):
            layers += [
                nn.Conv2d(chs[i], chs[i + 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.out_hw = out_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)
        return F.adaptive_avg_pool2d(f, self.out_hw)     # (B,128,14,14)


# --------------------------------------------------------------------------
class FrequencyBranch(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        f = cfg["model"]["freq"]
        self.use_phase = f.get("use_phase", True)
        self.use_learnable_mask = f.get("use_learnable_mask", True)
        hidden = f.get("mask_hidden", 32)
        kernel = f.get("mask_kernel", 5)
        chs = f.get("feat_channels", [32, 128])

        # 2D-DFT：训练用 torch FFT（最快），导出前切到 matmul（ONNX 可导出）
        size = cfg["data"].get("image_size", 224)
        self.dft = DFT2D(size, size, mode=f.get("dft_mode", "torch"))

        self.mask_net = (
            LearnableFrequencyMask(hidden, kernel) if self.use_learnable_mask else FixedBandMask()
        )
        self.amp_net = FreqPathNet(1, chs, out_hw=14)
        self.amp_dim = chs[-1]
        if self.use_phase:
            self.phase_net = FreqPathNet(1, chs, out_hw=14)
            self.phase_dim = chs[-1]
        else:
            self.phase_net = None
            self.phase_dim = 0
        self.out_dim = self.amp_dim + self.phase_dim    # 256 (或 128)

    # ------------------------------------------------------------------
    def set_dft_mode(self, mode: str) -> None:
        """切换 DFT 实现：'torch'（训练）/ 'matmul'（导出）。"""
        self.dft.mode = mode

    def to_amp_phase(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """公式 (4)(5)(6)：2D-DFT -> fftshift -> 分离幅值谱 / 相位谱（全实数运算）。

        输入先转灰度（3 通道平均），保证得到单通道二维频谱。
        """
        gray = x.mean(dim=1, keepdim=True)                   # (B,1,H,W) 转灰度
        return self.dft(gray)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amp, phase = self.to_amp_phase(x)                    # (B,1,224,224)
        log_amp = torch.log1p(amp)                           # 动态范围压缩
        mask = self.mask_net(log_amp)                        # (B,1,224,224) ∈ [0,1]  公式(7)
        amp_enh = amp * mask                                 # 公式 (8)
        amp_map = self.amp_net(amp_enh)                      # (B,128,14,14)

        if self.use_phase:
            phase_enh = phase * mask                         # 公式 (9)
            phase_map = self.phase_net(phase_enh)            # (B,128,14,14)
            f_freq = torch.cat([amp_map, phase_map], dim=1)  # (B,256,14,14)  公式(12)
        else:
            f_freq = amp_map                                 # (B,128,14,14)
        return f_freq

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_with_debug(self, x: torch.Tensor) -> dict:
        amp, phase = self.to_amp_phase(x)
        log_amp = torch.log1p(amp)
        mask = self.mask_net(log_amp)
        out = {
            "amp": amp, "phase": phase, "mask": mask,
            "F_freq": self.forward(x),
        }
        return out

    @torch.no_grad()
    def sanity_check(self, img_size: int = 224) -> dict:
        self.eval()
        x = torch.randn(2, 3, img_size, img_size)
        amp, phase = self.to_amp_phase(x)
        f = self.forward(x)
        return {
            "A(u,v)": tuple(amp.shape),
            "P(u,v)": tuple(phase.shape),
            "F_freq 特征图": tuple(f.shape),
            "F_freq(GAP) 向量维度": tuple(f.mean(dim=(2, 3)).shape),
            "use_phase": self.use_phase,
            "use_learnable_mask": self.use_learnable_mask,
        }
