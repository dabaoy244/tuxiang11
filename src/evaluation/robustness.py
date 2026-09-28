"""鲁棒性测试（申报书 6.3 "鲁棒性测试"）

对测试集图像施加常见后处理，评估检测精度与定位精度的**下降率**：
    JPEG 压缩  : 质量因子 10/20/30/50/70/90
    缩放       : 0.5/0.75/1.25/1.5/2.0
    裁剪       : 随机裁剪 10%/20%/30%/40%/50%
    旋转       : 0/90/180/270 度
    加噪       : 高斯(std 0.01/0.02/0.05)、椒盐(density 0.01/0.02/0.05)

输出：outputs/robustness.json + 可直接粘进论文的对比表（Markdown）

用法：
    python -m src.evaluation.robustness --ckpt checkpoints/vibnet_best.pt --demo
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..data.datasets import build_dataloaders
from ..engine.metrics import ClassificationMetrics, LocalizationMetrics
from ..models.spatial_branch import align_cfg_to_checkpoint
from ..models.vibnet import build_model, load_config


# ==========================================================================
# 图像后处理算子（作用于归一化前的 uint8 BGR 图像）
# ==========================================================================
def op_identity(img: np.ndarray) -> np.ndarray:
    return img


def make_jpeg(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return f


def make_scale(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        nh, nw = max(8, int(h * factor)), max(8, int(w * factor))
        interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
        small = cv2.resize(img, (nw, nh), interpolation=interp)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)   # 还原到原尺寸再评测
    return f


def make_crop(ratio: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        ch, cw = int(h * (1 - ratio)), int(w * (1 - ratio))
        y0 = int(np.random.randint(0, max(1, h - ch)))
        x0 = int(np.random.randint(0, max(1, w - cw)))
        c = img[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
    return f


def make_rotate(angle: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        if angle == 0:
            return img
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return f


def make_gauss(std: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, std * 255.0, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return f


def make_saltpepper(density: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        n = int(density * out.shape[0] * out.shape[1])
        if n <= 0:
            return out
        ys = np.random.randint(0, out.shape[0], n)
        xs = np.random.randint(0, out.shape[1], n)
        # ★ 噪声值必须与单点像素的形状一致：彩色图是 (n,3)，灰度图是 (n,)。
        # 原实现写死成 (n,) 赋值给 out[ys, xs]，在任何彩色图上都会
        # ValueError: shape mismatch —— 与 op_identity 同属「本脚本从未跑通过」的证据。
        vals = np.random.choice([0, 255], (n,) + out.shape[2:]).astype(out.dtype)
        out[ys, xs] = vals
        return out
    return f


# ==========================================================================
def build_ops(cfg: dict) -> List[tuple]:
    """按配置构造后处理算子列表。

    ★ 抽成独立函数是为了**可被自检覆盖**：本函数原先内联在 run_robustness 里，
    并且第一项曾误写成 `op_identity()`（把函数当值调用），导致整个鲁棒性脚本
    在任何输入下都直接 TypeError —— 而它不在任何自检路径上，静默坏掉了很久。
    回归测试见 scripts/smoke_test.py::test_robustness_ops。
    """
    r = cfg["eval"]["robustness"]
    ops = [("原图", op_identity)]
    ops += [(f"JPEG q={q}", make_jpeg(q)) for q in r["jpeg_quality"]]
    ops += [(f"缩放 x{s}", make_scale(s)) for s in r["scale"]]
    ops += [(f"裁剪 {int(x*100)}%", make_crop(x)) for x in r["crop_ratio"]]
    ops += [(f"旋转 {a}°", make_rotate(a)) for a in r["rotate"]]
    ops += [(f"高斯噪声 σ={s}", make_gauss(s)) for s in r["gauss_std"]]
    ops += [(f"椒盐噪声 d={d}", make_saltpepper(d)) for d in r["saltpepper_density"]]
    return ops


@torch.no_grad()
def run_robustness(
    cfg: dict,
    ckpt: Optional[str] = None,
    use_demo: bool = False,
    device: Optional[str] = None,
    max_samples: int = 400,
    save_dir: str = "outputs",
) -> Dict[str, object]:
    from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = None
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # 按 checkpoint 反推骨干档位并对齐配置（详见 spatial_branch.infer_backbone_variant）
        align_cfg_to_checkpoint(cfg, state.get("model", state))
    model = build_model(cfg)
    if state is not None:
        model.load_state_dict(state.get("model", state), strict=False)
    model.to(dev).eval()

    loader = build_dataloaders(cfg, use_demo=use_demo)["test"]
    thr = cfg["eval"].get("threshold", 0.5)
    size = cfg["data"]["image_size"]

    # 先缓存一批原始图像（避免反复读盘）
    cache: List[dict] = []
    for batch in loader:
        for i in range(batch["image"].shape[0]):
            cache.append({
                "x": batch["image"][i:i + 1],
                "label": int(batch["label"][i]),
                "mask": batch["mask"][i:i + 1] if batch["mask"] is not None else None,
                "valid": bool(batch["mask_valid"][i]) if "mask_valid" in batch else False,
            })
            if len(cache) >= max_samples:
                break
        if len(cache) >= max_samples:
            break

    ops = build_ops(cfg)

    def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
        """BGR uint8 (H,W,3) -> 归一化 tensor。"""
        import cv2

        h, w = img_uint8.shape[:2]
        if h != w:
            s = size / min(h, w)
            img_uint8 = cv2.resize(img_uint8, (int(w * s), int(h * s)))
            h, w = img_uint8.shape[:2]
            top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
            img_uint8 = img_uint8[top:top + size, left:left + size]
        else:
            img_uint8 = cv2.resize(img_uint8, (size, size))
        rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()

    from ..data.transforms import denormalize

    results: List[dict] = []
    baseline = None
    for name, op in ops:
        cls_m = ClassificationMetrics()
        loc_m = LocalizationMetrics(thr)
        n_loc = 0
        for item in cache:
            # 反归一化 -> uint8 BGR -> 施加后处理 -> 重新归一化
            vis = (denormalize(item["x"])[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            import cv2

            bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            bgr = op(bgr)
            xt = to_tensor(bgr).to(dev)
            out = model(xt, sample_vib=False, beta=0.0)
            cls_m.update(out["cls_logits"].float().cpu().numpy(),
                         np.array([item["label"]]))
            if item["valid"] and item["mask"] is not None:
                loc_m.update(out["mask_prob"].float().cpu().numpy(),
                             item["mask"].numpy())
                n_loc += 1

        c = cls_m.compute(thr)
        l = loc_m.compute() if n_loc else {}
        entry = {
            "op": name,
            "acc": c.get("acc"), "f1": c.get("f1"), "auc": c.get("auc"),
            "miou": l.get("miou"), "dice": l.get("dice"),
            "n_loc": n_loc,
        }
        if baseline is None:
            baseline = entry
            entry["acc_drop"] = 0.0
            entry["miou_drop"] = 0.0
        else:
            entry["acc_drop"] = round((baseline["acc"] or 0) - (c.get("acc") or 0), 4)
            entry["miou_drop"] = (round((baseline["miou"] or 0) - (l.get("miou") or 0), 4)
                                  if l else None)
        results.append(entry)
        print(f"[robust] {name:18s} ACC={entry['acc']:.4f} "
              f"F1={entry['f1']:.4f} "
              f"mIoU={(entry['miou'] if entry['miou'] is not None else float('nan')):.4f} "
              f"(ΔACC={entry['acc_drop']:+.4f})")

    out = {"n_samples": len(cache), "threshold": thr, "results": results}
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    with open(os.path.join(save_dir, "robustness.md"), "w", encoding="utf-8") as f:
        f.write("| 后处理操作 | ACC | F1 | mIoU | ACC 下降 | mIoU 下降 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['op']} | {r['acc']:.4f} | {r['f1']:.4f} | "
                    f"{'—' if r['miou'] is None else format(r['miou'], '.4f')} | "
                    f"{r['acc_drop']:+.4f} | "
                    f"{'—' if r['miou_drop'] is None else format(r['miou_drop'], '+.4f')} |\n")
    print(f"[robust] 结果已写入 {save_dir}/robustness.json 和 robustness.md")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/vibnet_best.pt")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_robustness(cfg, args.ckpt, args.demo, args.device, args.max_samples)


if __name__ == "__main__":
    main()
