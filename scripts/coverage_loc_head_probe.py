"""COVERAGE 定位头的『机制』探针（云端 CPU 跑；与 eval_external_domain 同口径）。

要回答的问题：`full` 的 miou_tampered_only = 0.1106 **低于**平凡基线"整图都判为篡改"(0.1168)
和"居中框"(0.1340)。那它到底输出了什么？三种可能：
  (a) 输出近饱和的大块（pred 面积 >> GT 面积）⇒ IoU 被面积稀释；
  (b) 输出小而无意义的斑点 ⇒ IoU 趋近 0；
  (c) 输出位置系统性偏离/集中在中心 ⇒ 靠几何运气。
本脚本统计 pred 的面积占比、质心偏移、以及 mp 的分布，用来区分这三者。
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluation.evaluate import align_cfg_to_checkpoint, build_dataloaders  # noqa: E402
from src.models.vibnet import build_model  # noqa: E402

CKPT = os.environ.get(
    "PROBE_CKPT",
    "/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/full/ckpt/ablation_full_best.pt",
)
EXT_ROOT = "/root/autodl-tmp/vibnet-forgery-detector/data/external"
THR = 0.5

spec = importlib.util.spec_from_file_location(
    "eed", os.path.join(ROOT, "scripts", "eval_external_domain.py"))
eed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eed)

cfg = eed.build_cfg("configs/default_crossval.yaml", "COVERAGE", EXT_ROOT)
torch.set_num_threads(8)

state = torch.load(CKPT, map_location="cpu", weights_only=False)
align_cfg_to_checkpoint(cfg, state.get("model", state))
model = build_model(cfg)
model.load_state_dict(state.get("model", state), strict=False)
model.eval()
print(f"[probe] 已加载 {CKPT}")

loader = build_dataloaders(cfg, use_demo=False)["test"]

pred_frac, gt_frac, ious, cent_off, mp_mean, mp_p90 = [], [], [], [], [], []
with torch.no_grad():
    for batch in loader:
        out = model(batch["image"], sample_vib=False, beta=0.0)
        mp = out["mask_prob"].float().numpy()
        gt = batch["mask"].numpy()
        for i in range(mp.shape[0]):
            g = gt[i].squeeze() > 0.5
            if g.sum() == 0:
                continue  # 只统计 GT 真有前景的（与 miou_tampered_only 同口径）
            p = mp[i].squeeze() > THR
            h, w = g.shape
            inter = float(np.logical_and(p, g).sum())
            union = float(np.logical_or(p, g).sum())
            ious.append(inter / union if union else 0.0)
            pred_frac.append(float(p.sum()) / (h * w))
            gt_frac.append(float(g.sum()) / (h * w))
            ys, xs = np.nonzero(p)
            if len(ys):
                cy, cx = ys.mean() / (h - 1), xs.mean() / (w - 1)
                cent_off.append(float(np.hypot(cy - 0.5, cx - 0.5)))
            else:
                cent_off.append(float("nan"))
            mp_mean.append(float(mp[i].mean()))
            mp_p90.append(float(np.percentile(mp[i], 90)))

a = lambda x: np.array(x)  # noqa: E731
print(f"[probe] 统计样本 n = {len(ious)}（GT 真有前景）")
print(f"  mIoU（本探针复算）        = {a(ious).mean():.4f}   （对齐 full 报告的 0.1106）")
print(f"  GT   面积占比            = {a(gt_frac).mean():.4f}")
print(f"  预测 面积占比(p>{THR})    = {a(pred_frac).mean():.4f}  中位={np.median(a(pred_frac)):.4f}")
print(f"  面积比 pred/GT           = {a(pred_frac).mean() / a(gt_frac).mean():.2f}×")
print(f"  预测质心偏移(归一化 0~0.71)= {np.nanmean(a(cent_off)):.4f}  中位={np.nanmedian(a(cent_off)):.4f}")
print(f"  mask_prob 均值           = {a(mp_mean).mean():.4f}")
print(f"  mask_prob p90            = {a(mp_p90).mean():.4f}")
print()
print("判读：面积比 >>1 且质心偏移小 ⇒ (a)+(c) 近饱和中心大块，IoU 被面积稀释；")
print("      面积比 ≈1 但 IoU 低 ⇒ 面积正确、位置错，(b)；")
print("      面积比 <<1 ⇒ 输出稀疏散点，(b)。")
