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

用法
----
    python scripts/fetch_pretrained_backbone.py                # 下载 + 转换 + 校验
    python scripts/fetch_pretrained_backbone.py --check-only    # 只校验已有文件
    python scripts/fetch_pretrained_backbone.py --mirror https://...  # 换镜像

产出
----
    pretrained/vit_b16_imagenet.pt   ← 转换后的 state_dict（本仓库命名）
  然后在配置里写：  model.spatial.backbone_weights: pretrained/vit_b16_imagenet.pt
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

# torchvision 官方 ViT-B/16 ImageNet-1K 权重（文件名里的 hash 是官方固定的）
DEFAULT_URL = "https://download.pytorch.org/models/vit_b_16-c867db91.pth"
DEFAULT_SIZE = 346328529

RAW_DIR = os.path.join(ROOT, "pretrained", "_raw")
OUT_PATH = os.path.join(ROOT, "pretrained", "vit_b16_imagenet.pt")


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
def convert(raw_path: str, out_path: str) -> dict:
    """把 torchvision ViT-B/16 的权重名映射成本仓库骨干的参数名。

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
    """
    import torch

    ck = torch.load(raw_path, map_location="cpu", weights_only=True)
    src = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck

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
    torch.save({"model": dst, "source": "torchvision vit_b_16 IMAGENET1K_V1",
                "num_layers": n_layer}, out_path)
    return {"张量数": len(dst), "层数": n_layer, "输出": out_path,
            "未匹配": len(unmatched)}


# --------------------------------------------------------------------------
def verify(path: str) -> bool:
    """把转换后的权重真的装进骨干，逐项核对，并做一次前向。

    只报"加载没报错"是不够的：`load_state_dict(strict=False)` 会**静默放过**
    没对上的键。所以这里显式列出 missing / unexpected，并要求两者都为空。
    """
    import torch
    from src.models.spatial_branch import TinyViTFallback

    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"]

    model = TinyViTFallback(hidden=768, depth=12, patch=16, img=224, heads=12,
                            mlp_ratio=4.0)
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
    return ok


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="获取并转换 ViT-B/16 预训练骨干")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--raw", default=os.path.join(RAW_DIR, "vit_b_16_imagenet.pth"))
    ap.add_argument("--check-only", action="store_true", help="跳过下载，只转换+校验")
    ap.add_argument("--verify-only", action="store_true", help="只校验已转换的文件")
    args = ap.parse_args()

    print("=" * 72)
    print(" 预训练骨干获取：torchvision ViT-B/16 (ImageNet-1K)")
    print("=" * 72)

    if args.verify_only:
        print("\n[3/3] 校验")
        return 0 if verify(args.out) else 1

    if not args.check_only:
        print(f"\n[1/3] 下载\n  源: {args.url}\n  目标: {args.raw}")
        if not download(args.url, args.raw, args.size):
            print("\n下载未完成。本脚本支持断点续传，直接重跑即可。")
            return 1
    else:
        print("\n[1/3] 跳过下载（--check-only）")
        if not os.path.exists(args.raw):
            print(f"  ❌ 原始权重不存在：{args.raw}")
            return 1

    print("\n[2/3] 转换命名")
    info = convert(args.raw, args.out)
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n[3/3] 校验（真的装进骨干 + 前向）")
    ok = verify(args.out)

    print("\n" + "=" * 72)
    if ok:
        print(" ✅ 完成。在配置里启用：")
        print("      model:")
        print("        spatial:")
        print("          backbone: tiny_vit")
        print("          backbone_variant: b16")
        print("          backbone_weights: pretrained/vit_b16_imagenet.pt")
    else:
        print(" ❌ 校验未通过，见上面 missing/unexpected 列表。")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
