"""离线演示数据集生成器

作用：在**没有 GPU、也没有下载公开数据集**的情况下，让整条流水线
（数据加载 -> 前向 -> 损失 -> 反向 -> 三阶段训练 -> 评测 -> ONNX 导出 -> 桌面工具）
端到端跑通，用于：

  1. 代码正确性自检（shape / 梯度 / 保存加载 / 导出）
  2. 团队成员在个人电脑上"先把工程跑起来"，再去服务器上换真数据集训练
  3. 桌面工具开发时的联调数据

⚠ 重要：演示数据是**程序化合成**的，不具备真实取证价值。
   报告里的指标只能用于验证"流程通不通"，**绝对不能**写进论文或结题材料。

合成策略（三类样本）：
  real            —— 程序化生成的"自然图像"（平滑背景 + 随机形状 + 传感器颗粒噪声）
  copymove/splice —— 复制移动 / 拼接：局部区域带轻微亮度-噪声-模糊失配，
                     掩码为被篡改的局部区域（用于训练定位支路）
  gensynth_like   —— 模拟生成模型的周期性上采样伪影（最近邻上采样 + 频域梳状峰），
                     掩码为整图（对应"整图伪造"），用于训练整图分类支路
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
def _smooth_noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    """低分辨率随机噪声上采样得到平滑纹理，模拟自然背景。"""
    import cv2

    small = rng.random((max(2, size // scale), max(2, size // scale))).astype(np.float32)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def make_natural_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """程序化"自然图像"：多尺度平滑背景 + 随机几何形状 + 颗粒噪声。"""
    import cv2

    base = np.zeros((size, size, 3), dtype=np.float32)
    for s in (64, 32, 16, 8):
        w = 1.0 / s
        for c in range(3):
            base[:, :, c] += w * _smooth_noise(rng, size, s)
    base = base / (base.max() + 1e-6)

    # 色调映射
    tint = rng.uniform(0.6, 1.0, size=3).astype(np.float32)
    img = np.clip(base * tint[None, None, :] * 255.0, 0, 255).astype(np.uint8)

    # 随机几何形状
    n_shapes = int(rng.integers(3, 9))
    for _ in range(n_shapes):
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        if rng.random() < 0.5:
            x1, y1 = rng.integers(0, size - 20, size=2)
            x2, y2 = x1 + rng.integers(15, size // 2), y1 + rng.integers(15, size // 2)
            cv2.rectangle(img, (int(x1), int(y1)), (int(min(x2, size - 1)), int(min(y2, size - 1))),
                          color, thickness=-1)
        else:
            cx, cy = rng.integers(20, size - 20, size=2)
            ax, ay = rng.integers(8, size // 4, size=2)
            cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)),
                        float(rng.uniform(0, 180)), 0, 360, color, thickness=-1)

    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    # 传感器颗粒噪声（真实图像都有，是"自然性"的一部分）
    noise = rng.normal(0, rng.uniform(1.5, 4.0), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_copymove(rng: np.random.Generator, size: int = 256):
    """复制移动 / 拼接：把一块区域搬运到别处，并制造轻微的噪声/亮度/模糊失配。"""
    import cv2

    img = make_natural_image(rng, size)
    w = int(rng.integers(size // 8, size // 3))
    h = int(rng.integers(size // 8, size // 3))
    x1 = int(rng.integers(0, size - w))
    y1 = int(rng.integers(0, size - h))
    patch = img[y1:y1 + h, x1:x1 + w].copy()

    x2 = int(rng.integers(0, size - w))
    y2 = int(rng.integers(0, size - h))
    while abs(x2 - x1) < w and abs(y2 - y1) < h:      # 避免完全重叠（否则掩码退化）
        x2 = int(rng.integers(0, size - w))
        y2 = int(rng.integers(0, size - h))

    # 三种失配之一或组合：亮度偏移 / 噪声水平不同 / 轻微模糊
    p = patch.astype(np.float32)
    if rng.random() < 0.8:
        p = p * rng.uniform(0.88, 1.12) + rng.uniform(-8, 8)
    if rng.random() < 0.6:
        p = p + rng.normal(0, rng.uniform(2.5, 7.0), p.shape)     # 噪声失配（关键线索）
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        p = cv2.GaussianBlur(p.astype(np.uint8), (k, k), 0).astype(np.float32)
    patch2 = np.clip(p, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:          # 轻微旋转后粘贴
        ang = float(rng.uniform(-25, 25))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        patch2 = cv2.warpAffine(patch2, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        mask_patch = cv2.warpAffine(np.ones((h, w), np.uint8), m, (w, h),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    else:
        mask_patch = np.ones((h, w), np.uint8)

    out = img.copy()
    out[y2:y2 + h, x2:x2 + w] = patch2
    mask = np.zeros((size, size), np.uint8)
    mask[y2:y2 + h, x2:x2 + w] = mask_patch
    return out, (mask * 255)


def make_gensynth_like(rng: np.random.Generator, size: int = 256):
    """模拟生成模型的周期性上采样伪影：下采样 + 最近邻上采样 + 频域梳状峰。"""
    import cv2

    img = make_natural_image(rng, size)
    factor = int(rng.choice([2, 4]))
    small = cv2.resize(img, (size // factor, size // factor), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)   # 棋盘伪影
    img = cv2.addWeighted(img, 0.45, up, 0.55, 0)

    # 频域梳状峰（GAN 上采样卷积核留下的周期性指纹）
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32).mean(axis=2)))
    mag, phase = np.abs(f), np.angle(f)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    period = size // factor
    comb = (np.cos(2 * np.pi * (yy - cy) / period) + np.cos(2 * np.pi * (xx - cx) / period))
    mag = mag * (1.0 + 0.05 * comb)
    rec = np.real(np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * phase))))
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)
    mask = np.full((size, size), 255, np.uint8)       # 整图伪造
    return out, mask


# --------------------------------------------------------------------------
def generate_demo_dataset(root: str, train: int = 240, val: int = 60, test: int = 60,
                          size: int = 256, seed: int = 3407) -> dict:
    """生成演示数据集，返回各 split 的样本统计。"""
    import cv2

    summary = {}
    for split, n in (("train", train), ("val", val), ("test", test)):
        rng = _rng(seed + hash(split) % 1000)
        img_dir = os.path.join(root, split, "image")
        mask_dir = os.path.join(root, split, "mask")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        n_real = n // 3
        n_cm = n // 3
        n_gen = n - n_real - n_cm
        count = {"real": 0, "copymove": 0, "gensynth_like": 0}

        for i in range(n_real):
            img = make_natural_image(rng, size)
            name = f"{split}_real_{i:05d}"
            cv2.imwrite(os.path.join(img_dir, name + ".png"), img)
            cv2.imwrite(os.path.join(mask_dir, name + ".png"), np.zeros((size, size), np.uint8))
            count["real"] += 1

        for i in range(n_cm):
            img, mask = make_copymove(rng, size)
            name = f"{split}_cm_{i:05d}"
            cv2.imwrite(os.path.join(img_dir, name + ".png"), img)
            cv2.imwrite(os.path.join(mask_dir, name + ".png"), mask)
            count["copymove"] += 1

        for i in range(n_gen):
            img, mask = make_gensynth_like(rng, size)
            name = f"{split}_gen_{i:05d}"
            cv2.imwrite(os.path.join(img_dir, name + ".png"), img)
            cv2.imwrite(os.path.join(mask_dir, name + ".png"), mask)
            count["gensynth_like"] += 1

        summary[split] = count | {"total": n}
    return summary


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="生成离线演示数据集")
    ap.add_argument("--root", default="data/demo")
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    info = generate_demo_dataset(args.root, args.train, args.val, args.test, args.size)
    print(f"[demo] 数据集已生成于 {os.path.abspath(args.root)}")
    for split, c in info.items():
        print(f"  {split:5s} total={c['total']:4d}  real={c['real']:3d}  "
              f"copymove={c['copymove']:3d}  gensynth_like={c['gensynth_like']:3d}")


if __name__ == "__main__":
    main()
