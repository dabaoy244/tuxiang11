"""图像预处理与数据增强。

训练增强刻意保持"温和"：JPEG 压缩、缩放、加噪等属于**鲁棒性测试**手段，
若在训练中过度使用会把频域伪影一并破坏，反而损害检测性能。
申报书 6.3 的鲁棒性测试是"评测手段"，不是"训练增强"，二者不能混淆。

此处训练增强只做：随机水平翻转 + 轻微色彩抖动 + 极小角度旋转（<=5°）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # CLIP 官方均值
IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)   # CLIP 官方标准差


class Transform:
    """轻量图像变换（只用 numpy + cv2，避免引入 albumentations 依赖）。"""

    def __init__(self, size: int = 224, train: bool = False):
        self.size = size
        self.train = train
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def _resize(self, img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        if h == w:
            return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA
                              if h > self.size else cv2.INTER_LINEAR)
        # 保持长宽比缩放 + 中心裁剪，避免拉伸改变频域伪影的各向异性
        scale = self.size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        top = max(0, (nh - self.size) // 2)
        left = max(0, (nw - self.size) // 2)
        return img[top:top + self.size, left:left + self.size]

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        """img: HxWx3 uint8 RGB；mask: HxW uint8 {0,1} 或 None。

        返回 (tensor (3,S,S) float32, mask_tensor (1,S,S) float32 或 None)
        """
        import cv2

        img = self._resize(img)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.float32), (self.size, self.size),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.float32)

        if self.train:
            if np.random.rand() < 0.5:                        # 水平翻转
                img = img[:, ::-1].copy()
                if mask is not None:
                    mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.3:                        # 轻微旋转 +-5 度
                ang = np.random.uniform(-5, 5)
                m = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, 1.0)
                img = cv2.warpAffine(img, m, (self.size, self.size), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
                if mask is not None:
                    mask = cv2.warpAffine(mask, m, (self.size, self.size), flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_REFLECT_101)
            if np.random.rand() < 0.3:                        # 轻微色彩抖动（不破坏高频）
                gain = np.random.uniform(0.95, 1.05, size=3).astype(np.float32)
                img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        x = img.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()

        m = None
        if mask is not None:
            m = torch.from_numpy(mask).unsqueeze(0).contiguous()
        return x, m


def build_transform(size: int = 224, train: bool = False) -> Transform:
    return Transform(size=size, train=train)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """反归一化回 [0,1]，用于可视化。输入 (3,H,W) 或 (B,3,H,W)。"""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1) if x.dim() == 4 \
        else torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1) if x.dim() == 4 \
        else torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)
