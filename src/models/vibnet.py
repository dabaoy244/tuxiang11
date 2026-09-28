"""整体模型装配：空频双分支轻量 VIB-Net

数据流（对应申报书 图表2 技术路线）：

    输入 x (B,3,224,224)
      ├─ 空域分支 SpatialBranch        -> F_spa  (B,896,14,14) + phi(x) (B,768)
      └─ 频域分支 FrequencyBranch      -> F_freq (B,256,14,14)
                     ↓
          CS-CAM 交叉注意力融合        -> F_fusion (B,512,14,14)          [1.1(3)]
                     ↓
        ┌────────────┴─────────────┐
     GS1 梯度停止层             GS2 梯度停止层                          [1.3(1)]
        ↓                          ↓
   分层 VIB (仅分类支路)       定位支路 Mobile-UNetv2 + 边缘头            [1.2 / 1.3(3)]
        ↓                          ↓
   MLP 分类头 256→128→2        224x224 篡改掩码 + 边缘图
        ↓                          ↓
   整图真伪概率 y_cls           像素级定位 M_pred

损失（1.4）：L_total = Σ 0.5*exp(-s_i)*L'_i + 0.5*Σ s_i
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .cs_cam import CSCAM
from .freq_branch import FrequencyBranch
from .gradient_stop import GradientStopLayer
from .mobile_unetv2 import LocalizationBranch
from .spatial_branch import SpatialBranch
from .vib import HierarchicalVIB

# 可冻结/解冻的模块组名
MODULE_GROUPS = ("spatial", "freq", "fusion", "vib", "cls_head", "localization")


class VIBNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        m, fcfg, lcfg = cfg["model"], cfg["model"]["fusion"], cfg["model"]["localization"]

        # ---- 1.1 空频双分支
        self.spatial = SpatialBranch(cfg)
        self.freq = FrequencyBranch(cfg)

        # ---- 1.1(3) 交叉注意力融合
        self.fusion = CSCAM(
            spa_dim=self.spatial.out_dim,
            freq_dim=self.freq.out_dim,
            dim=fcfg.get("dim", 512),
            use_cross_attention=fcfg.get("use_cross_attention", True),
        )
        fusion_dim = fcfg.get("dim", 512)

        # ---- 1.3(1) 梯度停止层
        use_gs = lcfg.get("use_grad_stop", True)
        self.gs_cls = GradientStopLayer(stop=lcfg.get("gs_stop_cls", False), name="GS1_cls")
        self.gs_loc = GradientStopLayer(stop=lcfg.get("gs_stop_loc", False), name="GS2_loc")
        if not use_gs:      # 消融：移除梯度隔离
            self.gs_cls.stop = self.gs_loc.stop = False
        self.use_grad_stop = use_gs

        # ---- 1.2 分层 VIB（仅分类支路最后一层）
        vcfg = m["vib"]
        self.vib = HierarchicalVIB(
            in_dim=fusion_dim,
            hidden_dim=vcfg.get("hidden_dim", 256),
            latent_dim=vcfg.get("latent_dim", 256),
            kl_clip_min=vcfg.get("kl_clip_min", 0.0),
            kl_clip_max=vcfg.get("kl_clip_max", 10.0),
            kl_reduction=vcfg.get("kl_reduction", "mean"),
            kl_clip_grad_through=vcfg.get("kl_clip_grad_through", True),
        )
        self.use_vib = vcfg.get("use_vib", True)

        # ---- 分类头：MLP 256 -> 128 -> 2（公式 27）
        h = m["head"]
        latent_dim = vcfg.get("latent_dim", 256)
        in_dim_cls = latent_dim if self.use_vib else fusion_dim
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim_cls, h.get("cls_hidden", 128)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h.get("cls_hidden", 128), h.get("num_classes", 2)),
        )

        # ---- 1.3(3) 定位支路
        self.localization = LocalizationBranch(cfg)

    # ==================================================================
    def forward(
        self,
        x: torch.Tensor,
        sample_vib: bool = True,
        beta: float = 0.0,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # 1.1 空频双分支
        f_spa, cls_token = self.spatial(x)          # (B,896,14,14), (B,768)
        f_freq = self.freq(x)                       # (B,256,14,14)

        # 1.1(3) 融合
        f_fusion = self.fusion(f_spa, f_freq)       # (B,512,14,14)

        # ---- 分类支路（经 GS1 + VIB）
        f_cls_in = self.gs_cls(f_fusion)
        vib_out = self.vib(f_cls_in, sample=sample_vib, beta=beta)
        z = vib_out["z"] if self.use_vib else f_cls_in.mean(dim=(2, 3))
        cls_logits = self.cls_head(z)               # (B,2)

        # ---- 定位支路（经 GS2，保留细粒度）
        f_loc_in = self.gs_loc(f_fusion)
        loc_out = self.localization(f_loc_in)

        out: Dict[str, torch.Tensor] = {
            "cls_logits": cls_logits,
            "cls_prob": torch.softmax(cls_logits, dim=-1),
            "mask_logits": loc_out["mask_logits"],
            "mask_prob": loc_out["mask_prob"],
            "kl": vib_out["kl"],
            "kl_raw": vib_out["kl_raw"],
            "mu": vib_out["mu"],
            "sigma": vib_out["sigma"],
            "beta": torch.tensor(float(beta)),
        }
        if "edge_logits" in loc_out:
            out["edge_logits"] = loc_out["edge_logits"]
            out["edge_prob"] = loc_out["edge_prob"]

        if return_intermediate:
            out.update({
                "F_spa": f_spa, "F_freq": f_freq, "F_fusion": f_fusion,
                "freq_mask": self.freq.mask_net(
                    torch.log1p(self.freq.to_amp_phase(x)[0])
                ),
                "cls_token": cls_token, "latent_z": z,
            })
        return out

    # ==================================================================
    # 参数管理：按模块组冻结 / 解冻（支撑三阶段训练）
    # ==================================================================
    def set_requires_grad(self, groups: Iterable[str], requires_grad: bool) -> None:
        groups = list(groups)
        if "all" in groups:
            for p in self.parameters():
                p.requires_grad_(requires_grad)
            return
        for g in groups:
            if g not in MODULE_GROUPS:
                raise ValueError(f"未知模块组 {g}，可选：{MODULE_GROUPS}")
            module = getattr(self, g)
            for p in module.parameters():
                p.requires_grad_(requires_grad)

    def apply_stage(self, freeze: List[str], train: List[str]) -> None:
        """按阶段配置设置 requires_grad。train 优先于 freeze。

        ★ 末尾必须**重放骨干的冻结策略**（2026-09-26 修）：
        `CLIPViTBackbone.__init__` 里已经按申报书 1.1(1) 把前
        `freeze_first_n_layers` 层冻好了，但本函数第一句
        `set_requires_grad(["all"], True)` 会把它们**全部重新解冻**。
        而三个阶段（含 stage3 的 `train: ["all"]`）都要让 `spatial` 参与训练，
        于是"冻结前 6 层"在**整个训练里都不生效**，且**不报错**：
        启动时打印的"冻结前 6/12 层，可训练参数 42.53M / 85.80M"是真的，
        进 stage1 后日志却变成"可训练参数 86.84M / 总计 90.53M" ——
        只有把相隔几十行的两个数字对起来看才会发现。
        连带后果：docs/00、docs/04、docs/08-C5 里"分类 ACC 卡 50% 就先检查
        `freeze_first_n_layers`"这条排查建议是**死的**（照做也不会有任何变化）。

        想恢复"全骨干微调"（不做任何冻结），改配置即可：
        `model.spatial.unfreeze_last_n_layers: 12`（本策略按
        `total - unfreeze_last_n` 计算冻结层数，等于全解冻）。
        """
        self.set_requires_grad(["all"], True)
        if freeze:
            self.set_requires_grad(freeze, False)
        if train and "all" not in train:
            self.set_requires_grad(["all"], False)
            self.set_requires_grad(train, True)
        # 最后重放骨干策略；显式 freeze "spatial" 时不重放（那表示用户想整支冻结）
        if "spatial" not in (freeze or []):
            self.spatial.backbone._apply_freeze_policy(tag=" [阶段重放]")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ==================================================================
    def set_dft_mode(self, mode: str) -> None:
        """切换频域分支的 DFT 实现。

        * "torch"  —— 训练阶段用，GPU 上 FFT 更快；
        * "matmul" —— **导出 ONNX 前必须切换**，否则会因
          `aten::fft_fft2` 不受 opset 17 支持而导出失败。
        两种实现数值等价（误差 ~1e-5），详见 src/models/dft.py。
        """
        self.freq.set_dft_mode(mode)

    # ==================================================================
    def summary(self) -> Dict[str, float]:
        def n(mod):
            return sum(p.numel() for p in mod.parameters()) / 1e6

        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        info = {
            "spatial(M)": n(self.spatial),
            "freq(M)": n(self.freq),
            "fusion(M)": n(self.fusion),
            "vib(M)": n(self.vib),
            "cls_head(M)": n(self.cls_head),
            "localization(M)": n(self.localization),
            "total(M)": total,
            "trainable(M)": trainable,
            "fp32_size(MB)": total * 4,
        }
        return info

    @torch.no_grad()
    def sanity_check(self, batch: int = 2) -> Dict[str, object]:
        """端到端维度自检：核对申报书各公式的输出维度。"""
        was_training = self.training
        self.eval()
        size = self.cfg["data"]["image_size"]
        x = torch.randn(batch, 3, size, size)
        out = self.forward(x, sample_vib=False, beta=0.1, return_intermediate=True)
        self.train(was_training)
        return {
            "输入 x": tuple(x.shape),
            "phi(x)/cls_token": tuple(out["cls_token"].shape),
            "F_spa (公式3)": tuple(out["F_spa"].shape),
            "F_freq (公式12)": tuple(out["F_freq"].shape),
            "F_fusion (公式17)": tuple(out["F_fusion"].shape),
            "latent z (公式23)": tuple(out["latent_z"].shape),
            "cls_logits (公式27)": tuple(out["cls_logits"].shape),
            "mask_prob (公式28)": tuple(out["mask_prob"].shape),
            "edge_prob (公式29)": tuple(out.get("edge_prob", torch.empty(0)).shape),
            "kl (公式25)": float(out["kl"]),
            "参数量": self.summary(),
        }


# ======================================================================
def build_model(cfg: dict) -> VIBNet:
    """工厂函数：从配置字典构建模型。"""
    model = VIBNet(cfg)
    return model


def load_config(path: str) -> dict:
    """读取 YAML 配置，并把相对路径解析为项目根目录下的绝对路径。"""
    import os

    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for key in ("output_dir", "ckpt_dir"):
        if key in cfg.get("project", {}):
            cfg["project"][key] = os.path.join(root, cfg["project"][key])
    if "data" in cfg and "root" in cfg["data"] and not os.path.isabs(cfg["data"]["root"]):
        cfg["data"]["root"] = os.path.join(root, cfg["data"]["root"])
    if "data" in cfg and "demo" in cfg["data"] and "root" in cfg["data"]["demo"]:
        if not os.path.isabs(cfg["data"]["demo"]["root"]):
            cfg["data"]["demo"]["root"] = os.path.join(root, cfg["data"]["demo"]["root"])
    return cfg
