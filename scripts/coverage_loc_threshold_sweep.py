"""COVERAGE 定位指标的『阈值敏感性』探针（云端 CPU）。

背景：`full` 的 miou_tampered_only = 0.1106，**低于**平凡基线
"整图都判篡改"(= GT 面积占比 0.1168) 和 "居中框"(0.1340)。
上一个探针发现：预测的**中位**面积只占 2.3%（GT 11.7%，欠预测 ~5×），
mask_prob 的 p90 只有 0.29 ⇒ 概率场整体没抬起来。那么 0.1106 有多少是
"定位头没信号"、有多少只是 0.5 这个固定阈值在域外**失配**？

本脚本扫阈值，报每个阈值下 91 张篡改图的平均 IoU 与平均预测面积，
用来判断：**有没有任何一个阈值能让 IoU 超过平凡基线**。
（若存在，则"零样本无定位能力"的结论不成立，只是操作点没校准；
  若不存在，则定位头确实无迁移信号。）
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
THRS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
# 平凡参照（上一个探针在同 91 张上实测）
TRIVIAL_ALL_ONES = 0.1168
TRIVIAL_CENTER_BOX = 0.1340
TRIVIAL_RANDOM_BOX = 0.0631

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
loader = build_dataloaders(cfg, use_demo=False)["test"]

probs, gts = [], []
with torch.no_grad():
    for batch in loader:
        mp = model(batch["image"], sample_vib=False, beta=0.0)["mask_prob"].float().numpy()
        gt = batch["mask"].numpy()
        for i in range(mp.shape[0]):
            g = gt[i].squeeze() > 0.5
            if g.sum() == 0:
                continue
            probs.append(mp[i].squeeze())
            gts.append(g)

print(f"[sweep] 篡改图 n = {len(probs)}")
print(f"{'thr':>5} {'meanIoU':>9} {'medIoU':>8} {'pred面积':>9} {'超平凡?':>10}")
best = (0.0, None)
for t in THRS:
    ious, areas = [], []
    for p, g in zip(probs, gts):
        pm = p > t
        inter = float(np.logical_and(pm, g).sum())
        union = float(np.logical_or(pm, g).sum())
        ious.append(inter / union if union else 0.0)
        areas.append(float(pm.sum()) / g.size)
    mi, md, ma = float(np.mean(ious)), float(np.median(ious)), float(np.mean(areas))
    tag = "yes" if mi > TRIVIAL_CENTER_BOX else ("vs all-ones" if mi > TRIVIAL_ALL_ONES else "no")
    if mi > best[0]:
        best = (mi, t)
    print(f"{t:>5.2f} {mi:>9.4f} {md:>8.4f} {ma:>9.4f} {tag:>10}")
print()
print(f"平凡基线：整图全判={TRIVIAL_ALL_ONES:.4f}  居中框={TRIVIAL_CENTER_BOX:.4f}  随机框={TRIVIAL_RANDOM_BOX:.4f}")
print(f"★ 最优阈值 = {best[1]}，对应 meanIoU = {best[0]:.4f}")
print(f"   报告口径用的是 thr=0.50 -> 0.1106（即默认操作点）")
