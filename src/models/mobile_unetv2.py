"""1.3 (3) 定位支路 —— Mobile-UNetv2 定位头 + Canny 边缘监督分支

对应申报书 1.3(3) 与公式 (28)(29)：

    M_pred = sigmoid(Mobile-UNetv2(F_fusion)) in [0,1]^(224x224)       (28)
    L_edge = BCE(M_edge_pred, M_edge_gt),  M_edge_gt = Canny(M_gt)     (29)

Mobile-UNetv2 结构（申报书原文）：
  编码器：4 个倒残差块，步长 1/2/2/2，通道 64 -> 128 -> 256 -> 512，ReLU6
  解码器：3 个上采样块（双线性 2x 上采样 -> 与编码器对应层拼接 -> 3x3 卷积融合）
  输出层：1x1 卷积 -> 1 通道 -> Sigmoid -> 224x224x1

输入：F_fusion (B, 512, 14, 14)（**未经 VIB 压缩**，避免细粒度信息丢失）
尺寸流：(14,14) -> 14 -> 7 -> 4 -> 2 -> 上采样 4 -> 7 -> 14 -> 224
（解码器上采样使用 `interpolate(size=skip.shape)` 对齐，规避奇数尺寸误差）

★ 所有双线性上采样都必须经由 `_bilinear_upsample_fp32()`（强制 fp32）：
  autocast 下 `upsample_bilinear2d` 会跟随输入的 bf16 执行，而它的**反向**核在
  CUDA 上走未向量化的 atomicAdd 慢路径 —— 单次实测 **2.9 s**，fp32 只要 ~7 ms
  （相差 400 倍），足以让 stage2 单步慢 14 倍。详见该函数说明与 docs/08 D43。

参数量约 3M 量级，满足申报书"轻量、适合端侧"的要求。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear_upsample_fp32(x: torch.Tensor, size) -> torch.Tensor:
    """双线性上采样，**强制在 fp32 下执行**，返回 fp32 结果。

    ★ 2026-09-27 云上实测（RTX 3080 Ti / torch 2.3.0+cu121，探针见
      scripts/diag_bwd2.py 与 docs/08 的 D43）：

      `F.interpolate(mode="bilinear")` **不在 autocast 的 fp32 提升名单里**。
      当输入是 bf16（autocast 下 conv 的输出就是 bf16）时，它按 bf16 执行，
      而 CUDA 的 `aten::upsample_bilinear2d_backward` 对 half/bf16 走的是
      **未向量化的 atomicAdd 慢路径** —— 单次调用实测 **2.9 s**，而 fp32
      只要 ~7 ms，相差 **400 倍**（torch.profiler 直接点名该算子占反向 97.19%）。

      后果：本文件的 14x14 -> 224x224 上采样一步就吃掉整个反向的 97%，
      使 stage2（启用定位任务）比 stage1（只用分类，定位支路反向被 autograd
      剪掉）慢 8 倍以上，而且**不报任何错**，只表现为"慢得莫名其妙"。

    这里显式转 fp32 再上采样，把最坏路径换掉。下游 conv 在 autocast 下会
    自动转回 bf16，因此数值只比原来更精确（fp32 上采样 ⊃ bf16 上采样），
    与申报书公式 (28)(29) 的实现等价，不影响语义。
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块：1x1 逐点(升维) -> 3x3 深度卷积 -> 1x1 逐点(降维)，ReLU6。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: float = 2.0):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        hidden = max(hidden, out_ch)
        self.use_skip = (stride == 1 and in_ch == out_ch)
        layers: List[nn.Module] = []
        if hidden != in_ch:
            layers += [nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return x + out if self.use_skip else out


class UpBlock(nn.Module):
    """上采样块：双线性插值 2x -> 与编码器特征拼接 -> 3x3 卷积融合。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 上采样走 fp32（见 _bilinear_upsample_fp32 的说明），再转回 skip 的
        # dtype 以保证 cat 成立 —— bf16 下这里同样会命中慢路径，只是尺寸小。
        x = _bilinear_upsample_fp32(x, skip.shape[-2:]).to(skip.dtype)
        return self.conv(torch.cat([x, skip], dim=1))


# --------------------------------------------------------------------------
class MobileUNetV2(nn.Module):
    def __init__(self, in_dim: int = 512, out_size: int = 224,
                 encoder_channels: Tuple[int, ...] = (64, 128, 256, 512),
                 encoder_strides: Tuple[int, ...] = (1, 2, 2, 2),
                 decoder_channels: Tuple[int, ...] = (256, 128, 64),
                 refine_channels: int = 32):
        super().__init__()
        self.out_size = out_size
        ec = list(encoder_channels)

        # ---- 编码器：4 个倒残差块
        self.enc1 = InvertedResidual(in_dim, ec[0], encoder_strides[0], expand_ratio=1.0)
        self.enc2 = InvertedResidual(ec[0], ec[1], encoder_strides[1], expand_ratio=2.0)
        self.enc3 = InvertedResidual(ec[1], ec[2], encoder_strides[2], expand_ratio=2.0)
        self.enc4 = InvertedResidual(ec[2], ec[3], encoder_strides[3], expand_ratio=2.0)

        # ---- 解码器：3 个上采样块 + 与编码器对应层拼接
        dc = list(decoder_channels)
        self.dec3 = UpBlock(ec[3], ec[2], dc[0])   # 2x2 -> 4x4
        self.dec2 = UpBlock(dc[0], ec[1], dc[1])   # -> 7x7
        self.dec1 = UpBlock(dc[1], ec[0], dc[2])   # -> 14x14

        # ---- 输出层：上采样到 224 -> 3x3 卷积 -> 1x1 卷积 -> 1 通道
        self.refine_channels = refine_channels
        self.refine = nn.Sequential(
            nn.Conv2d(dc[2], refine_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(refine_channels, 1, 1)
        # 供边缘监督头复用（= refine 的输出通道数）
        self.feature_dim = refine_channels

    def forward(self, f_fusion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (mask_logits (B,1,224,224), 解码器特征 (B,64,224,224))。"""
        e1 = self.enc1(f_fusion)     # (B, 64, 14, 14)
        e2 = self.enc2(e1)           # (B,128,  7,  7)
        e3 = self.enc3(e2)           # (B,256,  4,  4)
        e4 = self.enc4(e3)           # (B,512,  2,  2)

        d3 = self.dec3(e4, e3)       # (B,256,  4,  4)
        d2 = self.dec2(d3, e2)       # (B,128,  7,  7)
        d1 = self.dec1(d2, e1)       # (B, 64, 14, 14)

        # ★ 必须走 fp32（见 _bilinear_upsample_fp32）：bf16 下这一次上采样的
        #   反向要 2.9 s，占整个 stage2 反向的 97%（2026-09-27 实测）。
        feat = _bilinear_upsample_fp32(d1, (self.out_size, self.out_size))
        feat = self.refine(feat)     # (B, 32, 224, 224)
        logits = self.out_conv(feat)  # (B, 1, 224, 224)  公式 (28) 的 logits
        return logits, feat


class EdgeHead(nn.Module):
    """边缘监督分支：输入定位支路解码特征，1x1 卷积输出 224x224x1 边缘预测（公式 29）。"""

    def __init__(self, in_dim: int = 32, mid: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)

    @staticmethod
    def canny_edge(mask: torch.Tensor, low: int = 50, high: int = 150) -> torch.Tensor:
        """从 GT 掩码提取边缘 M_edge_gt（申报书：Canny 低阈值 50 / 高阈值 150）。

        纯 PyTorch 实现（Sobel 梯度 + 双阈值 + 非极大值抑制的简化版），
        避免训练时引入 OpenCV 依赖与 CPU-GPU 同步开销。
        若数据管线已离线生成边缘图，直接用离线结果更好。
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        m = mask.float()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=m.device, dtype=m.dtype).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        pad = F.pad(m, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(pad, kx)
        gy = F.conv2d(pad, ky)
        # 注意：不要把 eps 加在 sqrt 里面。
        # 若写成 sqrt(gx²+gy²+1e-8)，全零掩码会得到 mag≡1e-4、amax=1e-4，
        # 归一化后 mag≡1.0，于是"整幅图都是边缘"——这是极易踩的坑。
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        scale = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        mag = mag / scale
        lo, hi = low / 255.0, high / 255.0
        strong = (mag >= hi).float()
        weak = ((mag >= lo) & (mag < hi)).float()
        # 弱边缘沿 3x3 邻域找到强边缘则保留（滞后阈值）
        dilated_strong = F.max_pool2d(strong, 3, 1, 1)
        edge = torch.clamp(strong + weak * dilated_strong, 0, 1)
        return edge


# --------------------------------------------------------------------------
class LocalizationBranch(nn.Module):
    """定位支路整体：Mobile-UNetv2 定位头 + 可选边缘监督头。"""

    def __init__(self, cfg: dict):
        super().__init__()
        loc = cfg["model"]["localization"]
        self.head = MobileUNetV2(
            in_dim=cfg["model"]["fusion"]["dim"],
            out_size=cfg["data"]["image_size"],
            encoder_channels=tuple(loc.get("encoder_channels", [64, 128, 256, 512])),
            encoder_strides=tuple(loc.get("encoder_strides", [1, 2, 2, 2])),
        )
        self.use_edge_head = loc.get("use_edge_head", True)
        self.edge_head = EdgeHead(self.head.feature_dim) if self.use_edge_head else None

    def forward(self, f_fusion: torch.Tensor) -> dict:
        mask_logits, feat = self.head(f_fusion)
        out = {
            "mask_logits": mask_logits,
            "mask_prob": torch.sigmoid(mask_logits),
            "loc_feat": feat,
        }
        if self.edge_head is not None:
            edge_logits = self.edge_head(feat)
            out["edge_logits"] = edge_logits
            out["edge_prob"] = torch.sigmoid(edge_logits)
        return out

    def num_millions(self) -> float:
        n = sum(p.numel() for p in self.head.parameters())
        if self.edge_head is not None:
            n += sum(p.numel() for p in self.edge_head.parameters())
        return n / 1e6
