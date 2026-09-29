"""可导出的 2D-DFT 实现（申报书公式 (4)(5)(6)）

----------------------------------------------------------------------
为什么需要这个模块（这是一个**必须在早期解决的部署问题**）
----------------------------------------------------------------------
频域分支按公式 (4) 直接用 `torch.fft.fft2` 写是最自然的，但：

    torch.onnx.export(..., opset_version=17)  ->  UnsupportedOperatorError:
    Exporting the operator 'aten::fft_fft2' to ONNX opset version 17 is not supported

**PyTorch 的 FFT 算子无法导出到 ONNX opset 17**（ONNX 有独立的 `DFT` 算子，
但 PyTorch 没有为 `fft_fft2` 注册对应的符号函数）。
也就是说：如果直接用 `torch.fft`，你训练完的模型**根本导不出 ONNX，
后面 OpenVINO 加速、桌面工具全部卡死**。这个问题往往在项目后期才暴露，
届时已经来不及改结构。

解决办法：把 2D-DFT 写成**固定的矩阵乘法**。
DFT 本质是线性变换：`F = W_H · x · W_W^T`，其中

    W_H[h, k] = exp(-j·2π·h·k / H) / √H
    W_W[v, l] = exp(-j·2π·v·l / W) / √W        （norm='ortho' 归一化）

对于实数输入 x，令 A = Wre·x、B = Wim·x，则

    Re(F) = A·Wre^T − B·Wim^T
    Im(F) = A·Wim^T + B·Wre^T

`fftshift` 也可以**烘焙进矩阵的行序**（把行按 (h + H//2) % H 排列），
于是整条链路只剩 matmul / sqrt / atan2 —— 全部是 ONNX opset 17 支持的算子。

**额外好处：CPU 上更快。** 224×224 的矩阵乘法一次约 11M MAC，全流程 6 次约 67M MAC，
折算约 1~3 ms（Intel i7），比 OpenVINO 里没有高效实现的 FFT 算子更可控。

两种模式：
    dft_mode="torch"  ：训练用（GPU 上 FFT 更快，且数值最权威）
    dft_mode="matmul" ：导出/部署用（数值与 torch 模式等价，误差 ~1e-5）

验证等价性：`python -m src.models.dft`
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ==========================================================================
def build_dft_matrices(n: int, shift: bool = True, ortho: bool = True
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造 1D DFT 的实部/虚部矩阵。

    Args:
        n: 变换长度
        shift: 是否把 fftshift 烘焙进行序（行索引 r 对应频率槽 (r + n//2) % n）
        ortho: 是否使用 1/sqrt(n) 正交归一化（与 torch.fft.fft2(norm='ortho') 一致）

    Returns:
        (Wre, Wim)，均为 (n, n) 的 float32 张量
    """
    k = torch.arange(n, dtype=torch.float64).view(1, -1)          # 输入索引
    rows = torch.arange(n, dtype=torch.float64).view(-1, 1)       # 频率索引
    if shift:
        rows = (rows + n // 2) % n
    angle = -2.0 * math.pi * rows * k / n
    scale = (1.0 / math.sqrt(n)) if ortho else 1.0
    return (torch.cos(angle) * scale).float(), (torch.sin(angle) * scale).float()


# ==========================================================================
class DFT2D(nn.Module):
    """2D-DFT + fftshift + 幅相分离（公式 (4)(5)(6)），支持 ONNX 导出。

    forward 输入 (B,1,H,W) 实数张量，输出 (amp, phase) 均为 (B,1,H,W)。
    """

    def __init__(self, height: int = 224, width: int = 224, mode: str = "torch",
                 shift: bool = True, ortho: bool = True):
        super().__init__()
        assert mode in ("torch", "matmul"), f"未知 dft_mode: {mode}"
        self.mode = mode
        self.height, self.width = height, width
        self.shift, self.ortho = shift, ortho

        wre_h, wim_h = build_dft_matrices(height, shift, ortho)
        wre_w, wim_w = build_dft_matrices(width, shift, ortho)
        # persistent=False：不写进 checkpoint（可随时重建，省 1.6MB 权重体积）
        self.register_buffer("Wre_H", wre_h, persistent=False)
        self.register_buffer("Wim_H", wim_h, persistent=False)
        self.register_buffer("Wre_W", wre_w, persistent=False)
        self.register_buffer("Wim_W", wim_w, persistent=False)

    # ------------------------------------------------------------------
    def to_matmul(self) -> "DFT2D":
        """切到可导出模式（导出 ONNX 前必须调用）。"""
        self.mode = "matmul"
        return self

    def to_torch(self) -> "DFT2D":
        self.mode = "torch"
        return self

    # ------------------------------------------------------------------
    def _fft_shift_torch(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.shift:
            return spec
        return torch.fft.fftshift(spec, dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B,1,H,W)  ->  (amp (B,1,H,W), phase (B,1,H,W))"""
        if self.mode == "torch":
            spec = torch.fft.fft2(x, dim=(-2, -1), norm="ortho" if self.ortho else "backward")
            spec = self._fft_shift_torch(spec)
            real, imag = spec.real, spec.imag
        else:
            # F = Wre_H·x·Wre_W^T - Wim_H·x·Wim_W^T + j(Wre_H·x·Wim_W^T + Wim_H·x·Wre_W^T)
            a = torch.matmul(self.Wre_H, x)                 # (B,1,H,W)
            b = torch.matmul(self.Wim_H, x)
            real = torch.matmul(a, self.Wre_W.t()) - torch.matmul(b, self.Wim_W.t())
            imag = torch.matmul(a, self.Wim_W.t()) + torch.matmul(b, self.Wre_W.t())

        amp = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        phase = torch.atan2(imag, real)                     # (-pi, pi]
        return amp, phase


# ==========================================================================
def _self_test() -> None:
    """验证 matmul 模式与 torch FFT 模式的数值等价性。

    注意：相位比较要用**圆周差** `wrap(Δφ)`，因为 atan2 的值域 (-π, π] 有分支割线，
    在幅值趋近 0 的点上（相位本身无意义）Δφ 会跳到 ±2π。
    """
    torch.manual_seed(0)
    ok = True
    for size in (64, 224):
        x = torch.randn(2, 1, size, size)
        d_t = DFT2D(size, size, mode="torch")
        d_m = DFT2D(size, size, mode="matmul")
        a1, p1 = d_t(x)
        a2, p2 = d_m(x)
        amp_err = float((a1 - a2).abs().max())
        d_phi = p1 - p2
        d_phi = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi   # 圆周差
        mask = a1 > 1e-2
        phase_err = float(d_phi[mask].abs().max()) if bool(mask.any()) else 0.0
        print(f"  size={size:4d}  幅值最大误差={amp_err:.3e}  "
              f"相位最大圆周误差={phase_err:.3e}")
        ok &= amp_err < 1e-4 and phase_err < 1e-3
    print("matmul 模式与 torch FFT 模式数值等价 [OK]" if ok else "[x] 数值不等价")
    assert ok


if __name__ == "__main__":
    print("DFT2D 自检：")
    _self_test()
