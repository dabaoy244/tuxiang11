"""1.1 (1) 空域特征提取分支 —— "全局语义特征 + 局部纹理特征"

对应申报书公式 (1)(2)(3)：

    phi(x)   = CLIP-ViT-B/16(x)[:, 0, :]            in R^768      (1)  全局语义(CLS token)
    F_local  = GAP(DS-Conv3(x))                     in R^128      (2)  3 层深度可分离卷积
    F_spa    = Concat(phi(x), F_local)              in R^896      (3)  通道拼接

----------------------------------------------------------------------
实现说明（重要，写论文/答辩时会用到）
----------------------------------------------------------------------
公式 (1)(3) 给出的是 **向量** 形式，但 1.3(3) 定位支路要求
"512 维空频融合特征图（尺寸为 14x14x512）"。二者必须统一，本实现的做法是：

  * 全局语义：取 CLIP-ViT-B/16 的 patch token（去掉 CLS token）→ (B, 196, 768)
    → 重排为 (B, 768, 14, 14) 的空间特征图；CLS token 单独保留，
    作为整图真伪判别的"全局语义向量"（与公式 (1) 一致）。
  * 局部纹理：DS-Conv3 直接在 224x224 原图上做，输出 (B, 128, 224, 224)，
    自适应平均池化到 14x14 → (B, 128, 14, 14)，与全局特征空间对齐。
  * 通道拼接：(B, 896, 14, 14) 即 F_spa 的 "特征图" 形式；
    对 (B, 896, 14, 14) 做 GAP 即得到公式 (3) 的 896 维向量。
    验证：`model.spatial_branch.sanity_check()`。

这样既满足公式 (1)(2)(3)，又满足 1.3(3) 对空间分辨率的要求。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: 仓库根目录。用于把配置里写的相对路径（如 pretrained/xxx.pt）解析成绝对路径 ——
#: 训练脚本可能从任意工作目录启动，只靠相对路径会找不到文件。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
# 深度可分离卷积
# --------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """深度卷积 + 逐点卷积，3x3 / stride 1 / pad 1（申报书 1.1(1) 局部纹理特征提取）。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class LocalTextureNet(nn.Module):
    """3 层深度可分离卷积网络：通道 32 -> 64 -> 128，输出空间特征图。

    对 224x224 输入，三层 stride=1 保持分辨率，最后池化到 14x14。
    （等价于公式 (2) 的 DS-Conv3；GAP 得到 128 维向量。）
    """

    def __init__(self, in_ch: int = 3, channels: List[int] = (32, 64, 128), out_hw: int = 14):
        super().__init__()
        chs = [in_ch] + list(channels)
        self.blocks = nn.Sequential(
            *[
                DepthwiseSeparableConv(chs[i], chs[i + 1], kernel=3, stride=1, padding=1)
                for i in range(len(channels))
            ]
        )
        self.out_hw = out_hw
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.blocks(x)                       # (B, 128, H, W)
        return F.adaptive_avg_pool2d(f, self.out_hw)   # (B, 128, 14, 14)


# --------------------------------------------------------------------------
# CLIP-ViT-B/16 骨干
# --------------------------------------------------------------------------
class _CLIPVisionV4Layout(nn.Module):
    """把 transformers>=5 的 `CLIPVisionModel` 包一层，使其 **state_dict 键名**
    与 transformers 4.x 完全一致（多一层 `vision_model.` 前缀）。

    为什么需要（2026-09-29 本机实测踩到）
    ------------------------------------
    transformers 5.x 去掉了 `CLIPVisionModel -> vision_model(CLIPVisionTransformer)`
    这层嵌套，视觉塔直接挂在根上，键名从
        vision_model.embeddings.class_embedding
    变成
        embeddings.class_embedding
    权重本身**完全没变**，只是命名少了一层。而本项目云端训练产出的 checkpoint
    是 4.x 布局的，于是在 transformers 5.x 的机器上加载时：

        load_state_dict(..., strict=False)  → missing 150 / unexpected 199

    整个空域骨干（85.8M 参数）**保持随机初始化且不报错**，
    ACC 恒为 0.5、三种 mIoU 全为 0 —— 看起来像"模型没训好"，其实是键名没对上。
    这里补一层包装即可两边通用，无需降级 transformers。

    注意：只补 `vision_model` 这一层，forward 语义不变（v4 的 CLIPVisionTransformer
    与 v5 的 CLIPVisionModel 都返回 `last_hidden_state`）。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.vision_model = model

    def forward(self, pixel_values: torch.Tensor = None, **kw):  # type: ignore[assignment]
        return self.vision_model(pixel_values=pixel_values, **kw)


def _as_v4_layout(model: nn.Module) -> nn.Module:
    """transformers 4.x 布局原样返回；5.x 布局补一层 `vision_model.` 包装。"""
    if hasattr(model, "vision_model"):
        return model
    print("[CLIPViTBackbone] 检测到 transformers>=5 的扁平布局，"
          "已补 `vision_model.` 包装以对齐 4.x 的键名（权重内容无差异）。")
    return _CLIPVisionV4Layout(model)


class CLIPViTBackbone(nn.Module):
    """CLIP-ViT-B/16：冻结前 6 层，仅微调后 6 层（申报书 1.1(1)）。

    - 优先从本地目录加载权重（离线场景），其次从 HuggingFace 名称加载；
    - 若两者都不可用，可退回 `TinyViTFallback`（结构同构的轻量随机初始化骨干），
      用于无网络/无 GPU 环境下跑通全流程（指标不可用于论文，仅用于验证代码正确性）。
    """

    def __init__(
        self,
        pretrained_name: str = "openai/clip-vit-base-patch16",
        local_dir: str | None = None,
        freeze_first_n: int = 6,
        unfreeze_last_n: int = 6,
        allow_fallback: bool = True,
        force_fallback: bool = False,
        variant: str = "b16",
        weights_path: str | None = None,
    ):
        super().__init__()
        self.variant = variant if variant in BACKBONE_PRESETS else "b16"
        hidden, depth, heads, mlp_ratio = BACKBONE_PRESETS[self.variant]
        self.preset = dict(hidden=hidden, depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.hidden_dim = hidden
        self.grid = 14                     # 224 / 16 = 14
        self.num_patches = self.grid * self.grid   # 196
        self.is_fallback = False
        self.freeze_first_n = freeze_first_n
        self.unfreeze_last_n = unfreeze_last_n
        self.pretrained_loaded = False

        clip = None
        if not force_fallback and self.variant == "b16":
            clip = self._load_clip(pretrained_name, local_dir, allow_fallback)

        if clip is None:
            # 未取得 CLIP 权重：按档位构造离线骨干（b16 档同样走这里，保证可导出 ONNX）
            self.clip = TinyViTFallback(patch=16, img=224, **self.preset)
            self.is_fallback = True
            # ★ 若有本地转换好的预训练权重，装进去 —— 否则骨干是随机初始化，
            #   在几千~几万张图上根本训不起来（实测 5 epoch 后 acc 恒 0.5）。
            #   见 scripts/fetch_pretrained_backbone.py
            self._load_local_weights(weights_path)
        else:
            self.clip = clip
        self._apply_freeze_policy()

    # ------------------------------------------------------------------
    def _load_local_weights(self, weights_path: str | None) -> None:
        """把本地 state_dict 装进兜底骨干。

        用 strict=True 而不是 strict=False：strict=False 会**静默跳过**没对上的
        键，表面上"加载成功"，实际骨干一半还是随机的 —— 这种失败极难察觉。
        键不匹配就直接抛错，把问题暴露在启动阶段。
        """
        if not weights_path:
            return
        p = weights_path if os.path.isabs(weights_path) \
            else os.path.join(ROOT_DIR, weights_path)
        if not os.path.exists(p):
            print(f"[CLIPViTBackbone] ⚠️ 指定的骨干权重不存在：{p} —— "
                  f"骨干将为随机初始化，训练不会收敛。"
                  f"先跑 scripts/fetch_pretrained_backbone.py。")
            return
        blob = torch.load(p, map_location="cpu", weights_only=True)
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        try:
            self.clip.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"骨干权重键不匹配：{p}\n{e}\n"
                f"请确认 backbone_variant 与权重档位一致"
                f"（b16 权重需要 backbone_variant: b16）。") from e
        self.pretrained_loaded = True
        print(f"[CLIPViTBackbone] ✅ 已加载预训练骨干权重点 {p} "
              f"（{len(sd)} 个张量）")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_clip(name: str, local_dir: str | None, allow_fallback: bool):
        try:
            from transformers import CLIPVisionModel  # 延迟导入，避免强依赖
        except Exception:
            if not allow_fallback:
                raise
            print("[CLIPViTBackbone] 未安装 transformers，使用 TinyViT 兜底骨干。")
            return None

        # 先尝试"只用本地缓存/本地目录"，避免无网络时卡在下载上；
        # 若本地没有权重，且显式设置 VIB_NET_ALLOW_DOWNLOAD=1，才联网下载。
        attempts = []
        if local_dir:
            attempts.append((local_dir, True))
        attempts.append((name, True))
        if os.environ.get("VIB_NET_ALLOW_DOWNLOAD") == "1":
            attempts.append((name, False))

        for ckpt, local_only in attempts:
            try:
                # 统一成 transformers 4.x 的键名布局，否则 transformers 5.x 下
                # 加载本项目 checkpoint 会静默错配（见 _CLIPVisionV4Layout 说明）
                return _as_v4_layout(
                    CLIPVisionModel.from_pretrained(ckpt, local_files_only=local_only))
            except Exception as e:  # noqa: BLE001
                print(f"[CLIPViTBackbone] 加载 {ckpt} (local_only={local_only}) 失败："
                      f"{type(e).__name__}")
        print("[CLIPViTBackbone] 未取得 CLIP 预训练权重。"
              "如需联网下载：先 export VIB_NET_ALLOW_DOWNLOAD=1；"
              "或把权重放到 clip_local_dir；或用 backbone: tiny_vit 跑通流程。")
        if allow_fallback:
            # ⚠️ 不要说"指标无意义" —— 兜底骨干**就是这个仓库训练的架构**
            #    （b16 档同样是它），紧接着 _load_local_weights() 会尝试装入
            #    backbone_weights。是否"指标有意义"取决于有没有那一步，不取决于这里。
            print("[CLIPViTBackbone] 使用离线兜底骨干（结构等价于当前档位）。"
                  "若配置了 model.spatial.backbone_weights 且加载成功，"
                  "指标是有效的；否则骨干为随机初始化，指标无意义。")
            return None
        raise RuntimeError("无法加载 CLIP 权重")

    # ------------------------------------------------------------------
    def _apply_freeze_policy(self, tag: str = "") -> None:
        """冻结前 freeze_first_n 层，解冻后 unfreeze_last_n 层（含 embedding 之外的层）。

        ⚠ 只在构造函数里调一次是**不够的** —— `VIBNet.apply_stage()` 会把
        `requires_grad` 整体重置，所以每个阶段开始时都要重放一次
        （见 `VIBNet.apply_stage` 的注释）。`tag` 只用于让日志能区分
        "构造时"和"阶段重放"两次调用。
        """
        if self.is_fallback:
            # 兜底骨干是随机初始化的，指标本来就没有参考意义，不套用冻结策略。
            # 但也别**静默**跳过：线上核对"有没有 [阶段重放] 这行"时，
            # 静默会让"策略没生效"和"骨干是兜底的"两种完全不同的情况长得一样。
            if not getattr(self, "_fallback_freeze_noted", False):
                self._fallback_freeze_noted = True
                print("[CLIPViTBackbone] 兜底骨干：不做层冻结（随机初始化，指标无参考意义）")
            for p in self.clip.parameters():
                p.requires_grad_(True)
            return

        # 先全部冻结
        for p in self.clip.parameters():
            p.requires_grad_(False)

        layers = self.clip.vision_model.encoder.layers
        total = len(layers)
        n_unfreeze = min(self.unfreeze_last_n, total)
        for layer in layers[total - n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # 后置 LayerNorm 与视觉投影头参与微调
        if hasattr(self.clip.vision_model, "post_layernorm"):
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad_(True)
        if getattr(self.clip, "vision_model", None) is not None and hasattr(
            self.clip.vision_model, "pre_layrnorm"
        ):
            for p in self.clip.vision_model.pre_layrnorm.parameters():
                p.requires_grad_(True)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[CLIPViTBackbone]{tag} 冻结前 {total - n_unfreeze}/{total} 层，"
            f"可训练参数 {n_train/1e6:.2f}M / {n_total/1e6:.2f}M"
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (全局语义向量 cls (B,768), patch token 特征图 (B,768,14,14))。"""
        out = self.clip(pixel_values=x)
        tokens = out.last_hidden_state              # (B, 1+N, 768)
        cls_token = tokens[:, 0, :]                 # 公式 (1): phi(x) in R^768
        patch_tokens = tokens[:, 1:, :]             # (B, 196, 768)

        b, n, c = patch_tokens.shape
        g = int(n ** 0.5)
        fmap = patch_tokens.transpose(1, 2).reshape(b, c, g, g)   # (B, 768, 14, 14)
        return cls_token, fmap


class _ManualTransformerBlock(nn.Module):
    """手写 Transformer 编码块（pre-LN）。

    为什么不直接用 `nn.TransformerEncoder`：
    它的 `aten::_transformer_encoder_layer_fwd` 是**融合算子**，
    `torch.onnx.export` 在 opset 17 下无法导出（报 UnsupportedOperatorError）。
    而申报书 3.1 明确要求导出 ONNX opset 17，因此这里用最基础的
    matmul / softmax / layer_norm 手写一遍，保证整条部署链路可用。
    """

    def __init__(self, dim: int = 768, heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.heads
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B,h,N,d)
        attn = torch.softmax((q * self.scale) @ k.transpose(-1, -2), dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(y)
        x = x + self.mlp(self.norm2(x))
        return x


#: 骨干规模档位预设（hidden, depth, heads, mlp_ratio）
#:   b16 —— 申报书原始设定，对应 CLIP-ViT-B/16 结构，87.5M 参数
#:   s16 —— ViT-S/16 规模（384 维），22.5M 参数，精度/体积折中
#:   ti16 —— ViT-Ti/16 规模（192 维），5.9M 参数，**离线可构造、体积最小**
#: 注意：预训练权重缺失时这些档位是随机初始化的，只用于「跑通流程 + 报告体积」，
#:       真正用于精度实验请把对应预训练权重放到 clip_local_dir。
BACKBONE_PRESETS: Dict[str, Tuple[int, int, int, float]] = {
    "b16": (768, 12, 12, 4.0),
    "s16": (384, 12, 6, 4.0),
    "ti16": (192, 12, 3, 4.0),
    "ti8": (192, 8, 3, 4.0),
    # ↓ 仅用于加载"引入档位概念之前"训练出的旧 checkpoint（768 维 / 6 层 / mlp 2.0）。
    #   不建议用于新训练：它不是任何真实 ViT 的规模，只是为了旧权重还能被加载和复现。
    "legacy_vit768_6": (768, 6, 8, 2.0),
}

#: 仅用于兼容旧权重的档位，切换时会额外提示
LEGACY_VARIANTS = frozenset({"legacy_vit768_6"})


class TinyViTFallback(nn.Module):
    """离线可构造的 ViT 骨干：输出接口与 CLIP-ViT-B/16 完全一致（CLS + N patch tokens）。

    两个用途：
      1. **兜底**：无网络 / 无预训练权重时让整条流水线可跑通；
      2. **轻量档骨干**：`hidden/depth/heads` 可调，用于在「模型体积 ≤ 120MB」
         这一硬指标下做体积-精度权衡（见 docs/09）。

    刻意只用 matmul / softmax / layer_norm 等基础算子，保证可导出 ONNX opset 17。
    """

    def __init__(self, hidden: int = 768, depth: int = 6, patch: int = 16, img: int = 224,
                 heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden, patch, patch)
        n = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [_ManualTransformerBlock(hidden, heads=heads, mlp_ratio=mlp_ratio)
             for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, pixel_values: torch.Tensor):
        b = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)   # (B,N,768)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        class _Out:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return _Out(x)


# --------------------------------------------------------------------------
# 空域分支整体
# --------------------------------------------------------------------------
class SpatialBranch(nn.Module):
    """空域分支：F_spa = Concat(phi(x), F_local) ∈ R^896（空间形式 (B,896,14,14)）。"""

    def __init__(self, cfg: dict):
        super().__init__()
        s = cfg["model"]["spatial"]
        backbone = str(s.get("backbone", "clip_vit_b16")).lower()
        self.backbone = CLIPViTBackbone(
            pretrained_name=s.get("clip_pretrained", "openai/clip-vit-base-patch16"),
            local_dir=s.get("clip_local_dir"),
            freeze_first_n=s.get("freeze_first_n_layers", 6),
            unfreeze_last_n=s.get("unfreeze_last_n_layers", 6),
            allow_fallback=True,
            force_fallback=backbone in ("tiny_vit", "fallback", "tiny"),
            variant=str(s.get("backbone_variant", "b16")).lower(),
            weights_path=s.get("backbone_weights"),
        )
        self.local = LocalTextureNet(
            in_ch=3,
            channels=s.get("local_channels", [32, 64, 128]),
            out_hw=14,
        )
        self.global_dim = self.backbone.hidden_dim   # 768
        self.local_dim = s.get("local_dim", 128)     # 128
        self.out_dim = self.global_dim + self.local_dim  # 896

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (F_spa 特征图 (B,896,14,14), 全局语义向量 (B,768))。"""
        cls_vec, patch_map = self.backbone(x)         # (B,768), (B,768,14,14)
        local_map = self.local(x)                     # (B,128,14,14)
        f_spa = torch.cat([patch_map, local_map], dim=1)   # (B,896,14,14)  公式(3)
        return f_spa, cls_vec

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sanity_check(self, img_size: int = 224) -> dict:
        """核对与公式 (1)(2)(3) 的维度一致性。"""
        self.eval()
        x = torch.randn(2, 3, img_size, img_size)
        f_spa, cls_vec = self.forward(x)
        gap_vec = f_spa.mean(dim=(2, 3))
        info = {
            "phi(x) 维度": tuple(cls_vec.shape),
            "F_local 维度": (cls_vec.shape[0], self.local_dim),
            "F_spa 特征图": tuple(f_spa.shape),
            "F_spa(GAP) 向量维度": tuple(gap_vec.shape),
            "期望": "(2,768) + (2,128) -> (2,896,14,14)",
        }
        return info


# --------------------------------------------------------------------------
# checkpoint 兼容性：从权重反推骨干档位
# --------------------------------------------------------------------------
def infer_backbone_variant(state_dict: dict) -> Optional[str]:
    """从 checkpoint 的 state_dict 反推出它用的是哪个骨干档位。

    为什么需要这个函数
    ------------------
    骨干档位在引入 `BACKBONE_PRESETS` 前后不兼容：
      * 旧版兜底骨干 = 768 维 / 6 层 / mlp_ratio 2.0
      * 现在 b16 档   = 768 维 / 12 层 / mlp_ratio 4.0
    用旧 checkpoint 配新 `default.yaml` 直接 `load_state_dict` 会抛出
    `RuntimeError: size mismatch for spatial.backbone.clip.blocks.0.mlp.0.weight`
    —— 只看这个报错很难猜到是"骨干档位变了"。

    本函数按 **MLP 隐层宽度** 与 **Transformer 层数** 反推档位，
    返回档位名（`BACKBONE_PRESETS` 的 key），认不出来则返回 None。
    """
    import re

    probe_keys = (
        "spatial.backbone.clip.blocks.0.mlp.0.weight",
        "spatial.backbone.blocks.0.mlp.0.weight",
    )
    probe = next((k for k in probe_keys if k in state_dict), None)
    if probe is None:
        return None

    shape = tuple(state_dict[probe].shape)
    if len(shape) < 2:
        return None
    inter, hidden = int(shape[0]), int(shape[1])

    idx = set()
    for k in state_dict:
        m = re.match(r"spatial\.backbone\.(?:clip\.)?blocks\.(\d+)\.", k)
        if m:
            idx.add(int(m.group(1)))
    depth = len(idx)
    if depth == 0:
        return None

    ratio = inter / max(hidden, 1)
    for name, (h, d, _heads, r) in BACKBONE_PRESETS.items():
        if h == hidden and d == depth and abs(r - ratio) < 1e-6:
            return name
    return None


def align_cfg_to_checkpoint(cfg: dict, state_dict: dict, verbose: bool = True) -> Optional[str]:
    """按 checkpoint 的骨干档位就地修正 cfg，返回推断出的档位名。

    这样 `evaluate.py` / `train.py --resume` / 桌面工具在加载旧权重时
    不需要用户手工去改 YAML，也不会因为尺寸不匹配而崩溃。
    """
    variant = infer_backbone_variant(state_dict)
    if variant is None:
        return None

    cur = str(cfg["model"]["spatial"].get("backbone_variant", "b16")).lower()
    if variant == cur:
        return variant

    if verbose:
        print(f"[ckpt] 检测到 checkpoint 的骨干档位为 '{variant}'，"
              f"与配置里的 '{cur}' 不一致，已自动对齐。"
              f"（如需保持一致，请把 YAML 里的 backbone_variant 改为 '{variant}'）")
        if variant in LEGACY_VARIANTS:
            print("[ckpt] ⚠ 该档位是「引入档位概念之前」的旧兜底骨干（768/6层/mlp2.0），"
                  "仅用于复现旧结果。它不对应任何真实 ViT 规模、也没有预训练权重，"
                  "其精度指标**不能写进论文**。正式实验请用 b16/s16 档重新训练。")
    cfg["model"]["spatial"]["backbone_variant"] = variant
    # 档位不是 b16 时一定是离线构造的骨干，避免又去联网找 CLIP 权重
    if variant != "b16":
        cfg["model"]["spatial"]["backbone"] = "tiny_vit"
    return variant
