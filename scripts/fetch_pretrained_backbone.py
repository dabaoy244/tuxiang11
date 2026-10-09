#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给空域分支的骨干装上真正的预训练权重。

为什么需要这个脚本
------------------
`configs/*.yaml` 里 `backbone: tiny_vit` 时，骨干是**随机初始化**的
（`src/models/spatial_branch.py::TinyViTFallback`）。用随机初始化的 ViT
在几千~几万张图上从零训练，是训不出来的 —— 实测 5 个 epoch 后分类分支
准确率恒为 0.5（全判"真实"），AUC≈0.51。这不是代码问题，是没有先验知识。

申报书原本指定 CLIP-ViT-B/16，但 CLIP 权重托管在 HuggingFace，
而 `huggingface.co` 在部分网络环境下不可达。本脚本改从
`download.pytorch.org`（PyTorch 官方 CDN，可达性好得多）取
**ImageNet-1K 预训练的 ViT-B/16**，转换成本仓库骨干的参数命名。

为什么 ImageNet 预训练的 ViT-B/16 是合理替代
---------------------------------------------
两者的网络结构完全相同（patch 16 / 768 维 / 12 层 / 12 头 / mlp 4.0 /
patched 224² → 196 token + 1 cls），只是预训练目标不同：
  * CLIP  : 图文对比学习（4 亿对）
  * 本脚本: ImageNet-1K 分类（128 万张）
对本任务（检测生成图像的频域/纹理伪影）而言，ImageNet 预训练同样提供了
关键的底层视觉先验（边缘、纹理、颜色统计），足以让分类头收敛；
论文中应如实写成「以 ImageNet-1K 预训练 ViT-B/16 初始化」并说明原因。

★ 为什么还要一个 s16 档
----------------------
申报书另有「模型体积 ≤ 120MB」的硬指标，ViT-B/16 骨干本身就 87.5M 参数，
没有余量。`BACKBONE_PRESETS` 里的 **s16**（ViT-S/16 规模，384 维 / 12 层，约
22.5M 参数）是折中档，但它的骨干同样是随机初始化 ⇒ 一样训不出来。
本脚本 `--variant s16` 去 timm 取 `vit_small_patch16_224.augreg_in21k_ft_in1k`
（IN21k 预训练 + IN1k 微调，**必须走 hf-mirror.com**，huggingface.co 直连不可达），
键名映射后落成 `pretrained/vit_s16_imagenet.pt`。

用法
----
    python scripts/fetch_pretrained_backbone.py                    # b16：下载 + 转换 + 校验
    python scripts/fetch_pretrained_backbone.py --variant s16       # s16（轻量档，走 hf-mirror）
    python scripts/fetch_pretrained_backbone.py --check-only        # 只校验已有文件（不下载）
    python scripts/fetch_pretrained_backbone.py --verify-only       # 只校验已转换的 .pt
    python scripts/fetch_pretrained_backbone.py --url https://...   # 换源（自定义权重地址）

产出
----
    pretrained/vit_b16_imagenet.pt   ← b16 转换后的 state_dict（本仓库命名）
    pretrained/vit_s16_imagenet.pt   ← s16 转换后的 state_dict（轻量档）
  然后在配置里写：  model.spatial.backbone_weights: pretrained/<对应档位>.pt
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# torchvision / timm 官方权重地址与规模都在下面的 VARIANTS 里（按档位选）
RAW_DIR = os.path.join(ROOT, "pretrained", "_raw")

# --------------------------------------------------------------------------
# 两个档位共用同一套「下载 → 改名 → 装进骨干校验」流程，只是源格式与规模不同：
#   b16 —— torchvision ViT-B/16（download.pytorch.org 直连可达）
#   s16 —— timm ViT-S/16 augreg（HuggingFace 源，**必须走 hf-mirror**，
#           huggingface.co 在部分网络下不可达）。s16 是「模型体积 ≤120MB」
#           这条硬指标的候选档，权重 88.3MB。
# 注意：timm 的 s16 权重里 **没有** 本仓库 `blocks.N.mlp.0/.2` 这种名字，
#       它是 `mlp.fc1/fc2`；且多一个 `head.*` 分类头（要丢掉）。
VARIANTS = {
    "b16": {
        "url": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        "size": 346328529,
        "raw": os.path.join(RAW_DIR, "vit_b_16_imagenet.pth"),
        "out": os.path.join(ROOT, "pretrained", "vit_b16_imagenet.pt"),
        "kind": "torchvision",
        "preset": dict(hidden=768, depth=12, heads=12, mlp_ratio=4.0),
        "note": "torchvision ViT-B/16 (ImageNet-1K, IMAGENET1K_V1)",
    },
    "s16": {
        "url": ("https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
                "/resolve/main/pytorch_model.bin"),
        "size": 88257517,
        "raw": os.path.join(RAW_DIR, "vit_s16_timm.bin"),
        "out": os.path.join(ROOT, "pretrained", "vit_s16_imagenet.pt"),
        "kind": "timm",
        "preset": dict(hidden=384, depth=12, heads=6, mlp_ratio=4.0),
        "note": "timm vit_small_patch16_224.augreg_in21k_ft_in1k (IN21k 预训练 + IN1k 微调)",
    },
}

# 向后兼容：旧调用点（不传 --variant）默认仍是 b16
DEFAULT_URL = VARIANTS["b16"]["url"]
DEFAULT_SIZE = VARIANTS["b16"]["size"]
OUT_PATH = VARIANTS["b16"]["out"]


# --------------------------------------------------------------------------
def download(url: str, dest: str, expect_size: int | None = None) -> bool:
    """带断点续传的下载。返回 True 表示最终文件完整。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expect_size and have == expect_size:
        print(f"  [skip] 已完整（{have / 1e6:.1f} MB）")
        return True
    if expect_size and have > expect_size:
        print(f"  [warn] 本地文件比远端大（{have} > {expect_size}），删除重下")
        os.remove(dest)
        have = 0

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"  [resume] 从 {have / 1e6:.1f} MB 继续")

    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    got = have
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(dest, mode) as f:
        total = expect_size
        if total is None:
            cr = r.headers.get("Content-Range")           # bytes a-b/TOTAL
            total = int(cr.split("/")[-1]) if cr and "/" in cr else None
        last = time.time()
        while True:
            chunk = r.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last >= 2.0:
                last = now
                sp = (got - have) / max(now - t0, 1e-6) / 1e6
                if total:
                    pct = 100 * got / total
                    eta = (total - got) / max((got - have) / max(now - t0, 1e-6), 1) / 60
                    print(f"\r    {pct:5.1f}%  {got / 1e6:7.1f}/{total / 1e6:.1f} MB  "
                          f"{sp:.2f} MB/s  剩余 {eta:.1f} 分钟   ", end="", flush=True)
    print()
    if total and got != total:
        print(f"  [FAIL] 大小不符：得到 {got}，期望 {total}（可重跑本脚本续传）")
        return False
    print(f"  [ok ] 下载完成 {got / 1e6:.1f} MB，用时 {(time.time() - t0) / 60:.1f} 分钟")
    return True


# --------------------------------------------------------------------------
def convert(raw_path: str, out_path: str, kind: str = "torchvision",
            source_note: str = "") -> dict:
    """把上游权重名映射成本仓库骨干（TinyViTFallback）的参数名。

    命名对照（左＝本仓库 TinyViTFallback，右＝torchvision vit_b_16）
        patch_embed.weight           <- conv_proj.weight
        patch_embed.bias             <- conv_proj.bias
        cls_token                    <- class_token
        pos_embed                    <- encoder.pos_embedding
        blocks.N.norm1.{w,b}         <- encoder.layers.encoder_layer_N.ln_1.{w,b}
        blocks.N.qkv.{w,b}           <- ...self_attention.in_proj_{weight,bias}
        blocks.N.proj.{w,b}          <- ...self_attention.out_proj.{w,b}
        blocks.N.norm2.{w,b}         <- encoder.layers.encoder_layer_N.ln_2.{w,b}
        blocks.N.mlp.0.{w,b}         <- encoder.layers.encoder_layer_N.mlp.0.{w,b}
        blocks.N.mlp.2.{w,b}         <- encoder.layers.encoder_layer_N.mlp.3.{w,b}
        norm.{w,b}                   <- encoder.ln.{w,b}

    ★ qkv 不用重排：两边都是 torch 多头注意力的 (3E, E) 布局，行序均为 [q; k; v]。
      本仓库 forward 里 `reshape(b,n,3,h,c//h)` 后取 qkv[0]/[1]/[2] 分别当 q/k/v，
      与这个行序一致。

    kind="timm"（ViT-S/16 等）时**不需要逐层列名**：timm 的键名已与本仓库同构，
    只有 5 个前缀要改，外加丢弃 `head.*`：
        patch_embed.proj.*  -> patch_embed.*
        blocks.N.attn.qkv.* -> blocks.N.qkv.*
        blocks.N.attn.proj.*-> blocks.N.proj.*
        blocks.N.mlp.fc1.*  -> blocks.N.mlp.0.*
        blocks.N.mlp.fc2.*  -> blocks.N.mlp.2.*
        head.{weight,bias}  -> 丢弃（ImageNet 分类头，本任务不用）
    """
    import torch

    ck = torch.load(raw_path, map_location="cpu", weights_only=True)
    src = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck

    if kind == "timm":
        rename = {
            "patch_embed.proj.": "patch_embed.",
            ".attn.qkv.": ".qkv.",
            ".attn.proj.": ".proj.",
            ".mlp.fc1.": ".mlp.0.",
            ".mlp.fc2.": ".mlp.2.",
        }
        dst, dropped = {}, []
        for k, v in src.items():
            if k.startswith("head."):
                dropped.append(k)
                continue
            nk = k
            for a, b in rename.items():
                if a in nk:
                    nk = nk.replace(a, b)
                    break
            if nk in dst:
                raise RuntimeError(f"改名冲突：{k} -> {nk}")
            dst[nk] = v
        n_layer = len({k.split(".")[1] for k in dst if k.startswith("blocks.")})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"model": dst, "source": source_note, "num_layers": n_layer}, out_path)
        return {"张量数": len(dst), "层数": n_layer, "输出": out_path,
                "丢弃(分类头)": len(dropped)}

    dst = {}
    direct = {
        "conv_proj.weight": "patch_embed.weight",
        "conv_proj.bias": "patch_embed.bias",
        "class_token": "cls_token",
        "encoder.pos_embedding": "pos_embed",
        "encoder.ln.weight": "norm.weight",
        "encoder.ln.bias": "norm.bias",
    }
    for s, d in direct.items():
        if s in src:
            dst[d] = src[s]

    # 每层的映射：目标名 -> 候选源后缀（按顺序取第一个存在的）。
    # ★ torchvision 换过命名：早期版本 MLP 是 `mlp.0` / `mlp.3`，
    #   新版是 `mlp.linear_1` / `mlp.linear_2`。两种都列上，避免换版本就失配。
    #   同理 LN 的键名带 `.weight` / `.bias` 后缀，不能只写 `ln_1`
    #   —— 漏了后缀会**一个都匹配不上**，而 strict=False 不会报错，极难察觉。
    per_layer = {
        "norm1.weight": ["ln_1.weight"],
        "norm1.bias": ["ln_1.bias"],
        "norm2.weight": ["ln_2.weight"],
        "norm2.bias": ["ln_2.bias"],
        "qkv.weight": ["self_attention.in_proj_weight"],
        "qkv.bias": ["self_attention.in_proj_bias"],
        "proj.weight": ["self_attention.out_proj.weight"],
        "proj.bias": ["self_attention.out_proj.bias"],
        "mlp.0.weight": ["mlp.linear_1.weight", "mlp.0.weight"],
        "mlp.0.bias": ["mlp.linear_1.bias", "mlp.0.bias"],
        "mlp.2.weight": ["mlp.linear_2.weight", "mlp.3.weight"],
        "mlp.2.bias": ["mlp.linear_2.bias", "mlp.3.bias"],
    }
    n_layer = 0
    unmatched = []
    for i in range(64):                       # 层数未知，扫到没有为止
        pfx = f"encoder.layers.encoder_layer_{i}."
        if not any(k.startswith(pfx) for k in src):
            break
        n_layer += 1
        for target, cands in per_layer.items():
            for c in cands:
                if pfx + c in src:
                    dst[f"blocks.{i}.{target}"] = src[pfx + c]
                    break
            else:
                unmatched.append(f"blocks.{i}.{target}")

    if unmatched:
        print(f"  [warn] {len(unmatched)} 个目标键没找到来源，例如 {unmatched[:4]}")
        print(f"  [warn] 源权重里的层内键名形如：")
        sample = sorted({k.split('.', 3)[-1] for k in src
                         if '.encoder_layer_0.' in k})
        for s in sample[:14]:
            print(f"           {s}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": dst,
                "source": source_note or "torchvision vit_b_16 IMAGENET1K_V1",
                "num_layers": n_layer}, out_path)
    return {"张量数": len(dst), "层数": n_layer, "输出": out_path,
            "未匹配": len(unmatched)}


# --------------------------------------------------------------------------
def verify(path: str, preset: dict | None = None) -> bool:
    """把转换后的权重真的装进骨干，逐项核对，并做一次前向。

    只报"加载没报错"是不够的：`load_state_dict(strict=False)` 会**静默放过**
    没对上的键。所以这里显式列出 missing / unexpected，并要求两者都为空。
    """
    import torch
    from src.models.spatial_branch import TinyViTFallback

    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"]

    preset = preset or VARIANTS["b16"]["preset"]
    model = TinyViTFallback(patch=16, img=224, **preset)
    ret = model.load_state_dict(sd, strict=False)
    missing = list(ret.missing_keys)
    unexpected = list(ret.unexpected_keys)

    print(f"  模型参数张量数 : {len(model.state_dict())}")
    print(f"  权重文件张量数 : {len(sd)}")
    print(f"  missing 键     : {len(missing)}" + (f"  -> {missing[:5]}" if missing else "  ✅"))
    print(f"  unexpected 键  : {len(unexpected)}" + (f"  -> {unexpected[:5]}" if unexpected else "  ✅"))

    ok = not missing and not unexpected

    # 前向 + 退化检测：随机初始化的 ViT 输出几乎均匀，预训练的特征有明显结构
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        y = model(pixel_values=x).last_hidden_state
        # 用一张"有结构"的图对比：有结构时 patch token 的方差应显著更大
        x2 = torch.zeros(2, 3, 224, 224)
        x2[:, :, 40:180, 40:180] = 1.0                    # 一个白方块
        x2[:, 0, 40:180, 40:180] = 0.0                    # 只留 G/B 通道
        y2 = model(pixel_values=x2).last_hidden_state
    pt_std = float(y[:, 1:, :].std())
    pt_std2 = float(y2[:, 1:, :].std())
    # patch 之间的空间差异：预训练骨干对"有结构 vs 无结构"应有可辨差异
    spatial_var = float(y2[:, 1:, :].var(dim=1).mean())
    print(f"  前向输出       : shape={tuple(y.shape)}  patch token std={pt_std:.4f}")
    print(f"  结构敏感度     : 白方块图 patch token std={pt_std2:.4f}，"
          f"token 间空间方差={spatial_var:.4f}")
    if spatial_var < 1e-8:
        print("  ⚠️  token 间无空间差异 —— 骨干可能是退化的（常数输出）")
        ok = False

    # ★ 与"随机初始化骨干"对照：预训练权重若**真的装上了**，在**同一输入**上的
    #   输出应与随机骨干明显不同。这是能真正抓住"load_state_dict 静默没生效 /
    #   权重被零初始化"的一道闸门（strict=False 的 missing/unexpected 有时为空，
    #   但只要源权重是随机/零张量就查不出来）。
    ref = TinyViTFallback(patch=16, img=224, **preset)
    ref.eval()
    with torch.no_grad():
        y0 = ref(pixel_values=x2).last_hidden_state
    cos = float(torch.nn.functional.cosine_similarity(
        y2.flatten(1), y0.flatten(1), dim=1).mean())
    print(f"  与随机骨干余弦相似度 : {cos:.4f}（越接近 1.0 越可疑）")
    if cos > 0.99:
        print("  ⚠️  与随机初始化几乎一致 —— 权重可能根本没生效")
        ok = False
    return ok


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="获取并转换离线 ViT 预训练骨干（b16 / s16）")
    ap.add_argument("--variant", default="b16", choices=sorted(VARIANTS),
                    help="b16 = torchvision ViT-B/16（申报书原设定）；s16 = timm ViT-S/16（轻量档）")
    ap.add_argument("--url", default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--check-only", action="store_true", help="跳过下载，只转换+校验")
    ap.add_argument("--verify-only", action="store_true", help="只校验已转换的文件")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    url = args.url or spec["url"]
    size = args.size if args.size is not None else spec["size"]
    out = args.out or spec["out"]
    raw = args.raw or spec["raw"]

    print("=" * 72)
    print(f" 预训练骨干获取：[{args.variant}] {spec['note']}")
    print("=" * 72)

    if args.verify_only:
        print("\n[3/3] 校验")
        return 0 if verify(out, spec["preset"]) else 1

    if not args.check_only:
        print(f"\n[1/3] 下载\n  源: {url}\n  目标: {raw}")
        if not download(url, raw, size):
            print("\n下载未完成。本脚本支持断点续传，直接重跑即可。")
            return 1
    else:
        print("\n[1/3] 跳过下载（--check-only）")
        if not os.path.exists(raw):
            print(f"  ❌ 原始权重不存在：{raw}")
            return 1

    print("\n[2/3] 转换命名")
    info = convert(raw, out, kind=spec["kind"], source_note=spec["note"])
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n[3/3] 校验（真的装进骨干 + 前向）")
    ok = verify(out, spec["preset"])

    rel = os.path.relpath(out, ROOT).replace("\\", "/")
    print("\n" + "=" * 72)
    if ok:
        print(" ✅ 完成。在配置里启用：")
        print("      model:")
        print("        spatial:")
        print("          backbone: tiny_vit")
        print(f"          backbone_variant: {args.variant}")
        print(f"          backbone_weights: {rel}")
    else:
        print(" ❌ 校验未通过，见上面 missing/unexpected 列表。")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
