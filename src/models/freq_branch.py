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


def _radius(h: int, w: int, device) -> torch.Tensor:
    """归一化半径图（fftshift 后：低频在中心，r=0 是直流，r=1 触及各轴奈奎斯特）。

    坐标用 linspace(-1, 1)，因此 r 的取值范围是 [0, √2]（角上最大）。
    """
    yy = torch.linspace(-1, 1, h, device=device).view(-1, 1).expand(h, w)
    xx = torch.linspace(-1, 1, w, device=device).view(1, -1).expand(h, w)
    return torch.sqrt(xx ** 2 + yy ** 2)


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
        r = _radius(h, w, log_amp.device)
        m = ((r >= self.low) & (r <= self.high)).float()
        return m.view(1, 1, h, w).expand(b, 1, h, w)


class IdentityMask(nn.Module):
    """无掩码（恒等）：掩码 ≡ 1，等价于"做了 FFT 但不做任何频率选择"。

    对应 SE-DFNet（宁夏师范大学学报 2026）的频域流：固定 FFT + 对数幅度谱 + CNN，
    全文没有频域掩码。本项目把它作为"有频域、但无频率选择"这一档对照。
    """

    def forward(self, log_amp: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(log_amp)


class FixedHighPassMask(nn.Module):
    """固定高通硬掩码：保留归一化半径 r > cutoff 的高频，低频置 0。

    对应 FAD-Net（Machine Vision and Applications 2026, 37:146）的 FFM：
    原文在 DCT 上"把左上角低频区置 0、保留右下角高频区"，低频/高频位置固定、
    掩码不参与梯度。这里搬到 fftshift 后的频谱上——低频在**中心**，故保留中心圆外。
    """

    def __init__(self, cutoff: float = 0.5):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, log_amp: torch.Tensor) -> torch.Tensor:
        b, _, h, w = log_amp.shape
        m = (_radius(h, w, log_amp.device) > self.cutoff).float()
        return m.view(1, 1, h, w).expand(b, 1, h, w)


class StaticLearnableBandMask(nn.Module):
    """固定频带基 + **静态**（与输入无关）可学残差。

    对应 Luo et al. (Applied Soft Computing 164 (2024) 111992) 的 FAD 式(3)：
        y = D⁻¹{ D(x) ⊙ [ f_base + σ(f_w) ] }
    原文明确：`f_base` 是**固定的二值频带**，`f_w` 是**可学习滤波器**，且
    `σ 被归一化到 (−1,1)`；该设计更早源自 Qian et al. (ECCV 2020) "Thinking in Frequency"。

    ★ 本实现对原文的忠实点与取舍（必须写清，否则对照不成立）：
      - 忠实：掩码 = `clamp(f_base + residual, 0, 1)`，`residual = 2σ(f_w) − 1 ∈ (−1,1)`
        （与原文"σ 归一化到 (−1,1)"一致）；`f_w` 是 `nn.Parameter`，
        **与输入无关** —— 这正是与"输入条件化"对照的那个变量。
      - 取舍：原文用 3 个 `f_base` 产出 **3 张分解图**分别送网络；本分支的接口是
        **单个乘性掩码**，故取 `f_base` = 固定中频带（与 `FixedBandMask` 同形）作为单一基图。
        因此这里测的是"**静态可学** vs **输入条件化**"，不是"三分支分解 vs 单掩码"。
    """

    def __init__(self, low: float = 0.15, high: float = 0.55):
        super().__init__()
        self.low, self.high = low, high
        # 静态可学残差：形状 (1,1,H,W)，但 H/W 在建构造时未知 —— 用 lazy 参数
        self.f_w = nn.Parameter(torch.zeros(1, 1, 224, 224))
        self._size = None

    def _base(self, h: int, w: int, device) -> torch.Tensor:
        r = _radius(h, w, device)
        m = torch.where((r >= self.low) & (r <= self.high),
                        torch.ones_like(r), torch.full_like(r, 0.2))
        return m

    def forward(self, log_amp: torch.Tensor) -> torch.Tensor:
        b, _, h, w = log_amp.shape
        if self._size != (h, w):
            with torch.no_grad():
                self.f_w.data = F.interpolate(self.f_w.data, size=(h, w), mode="bilinear",
                                              align_corners=False)
            self._size = (h, w)
        base = self._base(h, w, log_amp.device)                    # (h,w) 固定基图
        residual = 2.0 * torch.sigmoid(self.f_w) - 1.0             # ∈ (-1,1) 静态
        m = torch.clamp(base.view(1, 1, h, w) + residual, 0.0, 1.0)
        return m.expand(b, 1, h, w)


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
#: 掩码档位注册表 —— 顺序即"频率选择能力"的阶梯（供 P0 对照实验逐档训练）
#:   有无频域 → 有频域无选择 → 固定选择 → 静态可学选择 → 输入条件化选择
MASK_MODES = {
    "none": IdentityMask,                     # 固定 FFT，不做任何频率选择（SE-DFNet 式）
    "fixed_band": FixedBandMask,              # 固定中频环带（软）
    "fixed_highpass": FixedHighPassMask,      # 固定高通硬掩码（FAD-Net 式）
    "static_learnable": StaticLearnableBandMask,   # 固定频带 + 静态可学残差（Luo 2024 / Qian 2020 式）
    "learnable": LearnableFrequencyMask,      # 输入条件化逐点掩码（本项目）
}

#: 哪些档位的掩码**随输入变化** —— 这是"输入条件化"的判定依据，单测直接读它
MASK_INPUT_DEPENDENT = {
    "none": False,
    "fixed_band": False,
    "fixed_highpass": False,
    "static_learnable": False,
    "learnable": True,
}


class FrequencyBranch(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        f = cfg["model"]["freq"]
        self.use_phase = f.get("use_phase", True)
        self.use_learnable_mask = f.get("use_learnable_mask", True)
        hidden = f.get("mask_hidden", 32)
        kernel = f.get("mask_kernel", 5)
        chs = f.get("feat_channels", [32, 128])

        # 掩码档位：mask_mode 优先；未给则按老的布尔开关回退（保持零回归）
        mode = f.get("mask_mode")
        if mode is None:
            mode = "learnable" if self.use_learnable_mask else "fixed_band"
        if mode not in MASK_MODES:
            raise ValueError(f"未知 mask_mode={mode!r}，可选：{sorted(MASK_MODES)}")
        self.mask_mode = mode

        # 2D-DFT：训练用 torch FFT（最快），导出前切到 matmul（ONNX 可导出）
        size = cfg["data"].get("image_size", 224)
        self.dft = DFT2D(size, size, mode=f.get("dft_mode", "torch"))

        if mode == "learnable":
            self.mask_net = LearnableFrequencyMask(hidden, kernel)
        elif mode == "fixed_highpass":
            self.mask_net = FixedHighPassMask(f.get("highpass_cutoff", 0.5))
        elif mode == "fixed_band":
            self.mask_net = FixedBandMask(f.get("band_low", 0.15), f.get("band_high", 0.55))
        elif mode == "static_learnable":
            self.mask_net = StaticLearnableBandMask(f.get("band_low", 0.15),
                                                    f.get("band_high", 0.55))
        else:                                    # "none"
            self.mask_net = IdentityMask()

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
            "mask_mode": self.mask_mode,
            "掩码随输入变化": MASK_INPUT_DEPENDENT[self.mask_mode],
        }


# --------------------------------------------------------------------------
class ZeroFreqBranch(nn.Module):
    """无频域通路（P0 对照实验的**下限臂**）。

    接口与 `FrequencyBranch` 完全一致（`out_dim` / `forward`），但：
      - **不构造** amp_net / phase_net / mask_net ⇒ 不占任何参数、不产生梯度；
      - forward 返回**全零**的 (B, out_dim, 14, 14)，使融合层看到"没有频域信息"。

    为什么用零张量而不是改融合层结构：这样空间分支与融合层的**参数、形状、
    初始化方式**与其它臂逐位一致，对照只差"频域信息是否存在"这一个变量。
    零输入经融合的交叉注意力后得到的是常量项（softmax 均匀），因此等价于
    "把频域那一半输入永久掐掉"。
    """

    def __init__(self, cfg: dict):
        super().__init__()
        f = cfg["model"]["freq"]
        chs = f.get("feat_channels", [32, 128])
        self.mask_mode = "nofreq"
        self.use_phase = f.get("use_phase", True)
        self.out_dim = chs[-1] * (2 if self.use_phase else 1)
        self._out_hw = 14

    def set_dft_mode(self, mode: str) -> None:      # 保持接口一致，空实现
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return x.new_zeros((b, self.out_dim, self._out_hw, self._out_hw))

    @torch.no_grad()
    def forward_with_debug(self, x: torch.Tensor) -> dict:
        return {"F_freq": self.forward(x), "mask": None}

    def extra_repr(self) -> str:
        return f"out_dim={self.out_dim}, mask_mode=no-freq-branch"


def build_freq_branch(cfg: dict) -> nn.Module:
    """按 `model.freq.use_freq` 选择：真实频域分支，或"无频域"下限臂。"""
    if cfg["model"]["freq"].get("use_freq", True):
        return FrequencyBranch(cfg)
    return ZeroFreqBranch(cfg)
