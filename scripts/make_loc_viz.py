"""生成「篡改定位」可视化对照图（原图 / GT 掩码 / 模型预测掩码）。

目的很直接：**如实展示定位支路当前的输出状态**。
本项目的定位头只在无掩码的 ForenSynths 上随分类支路一起跑过，
从未在 CASIA v2 的像素掩码上训练过，所以预测掩码退化是**预期结果**，
它正是「必须上 GPU 完成 stage2 定位预训练」的直接证据。

用法：
    python scripts/make_loc_viz.py --ckpt checkpoints/abp_best.pt --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2            # noqa: E402
import numpy as np    # noqa: E402
import torch          # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.spatial_branch import align_cfg_to_checkpoint  # noqa: E402
from src.models.vibnet import build_model, load_config  # noqa: E402

OUT = os.path.join(ROOT, "deliverables", "figures")
FONT = "C:/Windows/Fonts/msyh.ttc"


def fnt(sz, bold=False):
    p = "C:/Windows/Fonts/msyhbd.ttc" if bold else FONT
    try:
        return ImageFont.truetype(p if os.path.exists(p) else FONT, sz)
    except Exception:
        return ImageFont.load_default()


def pick_tampered(casia_root: str, k: int) -> list:
    """挑出掩码非空的篡改图（掩码全黑代表真实图）。"""
    imgd = os.path.join(casia_root, "test", "image")
    mskd = os.path.join(casia_root, "test", "mask")
    if not os.path.isdir(imgd):
        return []
    out = []
    for name in sorted(os.listdir(imgd)):
        stem = os.path.splitext(name)[0]
        mp = os.path.join(mskd, stem + ".png")
        if not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None and m.max() > 0:
            out.append((os.path.join(imgd, name), mp))
        if len(out) >= k:
            break
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/default.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints/vibnet_best.pt"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    align_cfg_to_checkpoint(cfg, state.get("model", state))
    model = build_model(cfg)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    size = cfg["data"]["image_size"]
    thr = cfg["eval"].get("threshold", 0.5)

    pairs = pick_tampered(os.path.join(ROOT, "data/Datasets/CASIAv2"), args.k)
    if not pairs:
        print("[loc-viz] 未找到带掩码的篡改图，跳过")
        return 1

    cells = []
    for ip, mp in pairs:
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        out = model(torch.from_numpy(x).float().to(device), sample_vib=False, beta=0.0)
        prob = out["mask_prob"][0, 0].float().cpu().numpy()
        logit = out["cls_logits"].float().cpu().numpy()[0]
        p_fake = float(np.exp(logit[1]) / np.exp(logit).sum())
        cells.append((ip, gt, prob, p_fake))

    T = 170
    W = T * 3 + 40
    H = T * len(cells) + 116
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((16, 12), "图 16 · 篡改定位可视化（CASIA v2 test，含掩码的篡改图）",
           font=fnt(19, True), fill=(28, 32, 38))
    d.text((16, 40), "结论先行：定位支路尚未在像素掩码上训练，预测掩码退化为「全背景」，"
                     "mIoU = 0 —— 这是上 GPU 后必须补的第一步。", font=fnt(12), fill=(180, 40, 40))
    heads = ["原图", "GT 掩码", "模型预测（概率图）"]
    for i, h in enumerate(heads):
        d.text((16 + i * T + (T - 8) / 2, 74), h, font=fnt(13, True), fill=(28, 32, 38), anchor="ma")
    for r, (ip, gt, prob, p_fake) in enumerate(cells):
        y0 = 96 + r * T
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (T - 8, T - 8)), cv2.COLOR_BGR2RGB)),
                 (16, y0))
        g = cv2.resize(gt, (T - 8, T - 8), interpolation=cv2.INTER_NEAREST)
        im.paste(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)), (16 + T, y0))
        heat = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.resize(heat, (T - 8, T - 8)), cv2.COLORMAP_JET)
        im.paste(Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)), (16 + 2 * T, y0))
        d.text((16 + 2 * T, y0 + T - 6), f"P(伪造)={p_fake:.3f}  max={prob.max():.4f}",
               font=fnt(11), fill=(90, 96, 106))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "fig16_localization_viz.png")
    im.save(p)
    print(f"[loc-viz] {os.path.relpath(p, ROOT)}")
    print(f"[loc-viz] 预测概率最大值范围："
          f"{min(c[2].max() for c in cells):.4f} ~ {max(c[2].max() for c in cells):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
