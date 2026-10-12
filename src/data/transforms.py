"""图像预处理与数据增强。

训练增强默认保持"温和"：JPEG 压缩、缩放、加噪等属于**鲁棒性测试**手段，
若在训练中过度使用会把频域伪影一并破坏，反而损害检测性能。
申报书 6.3 的鲁棒性测试是"评测手段"，不是"训练增强"，二者不能混淆。

默认训练增强只有：随机水平翻转 + 轻微色彩抖动 + 极小角度旋转（<=5°）。

★ 2026-10-02 新增「强增强」开关（`aug` 参数，**默认全关**）
----------------------------------------------------------------
背景：ForenSynths 训练集只用到了 4 个类的约 4 万张（占比约 5.6%），
而跨生成器泛化差距是项目两大短板之一。Wang et al. 2020（CNNDetection）
指出 **JPEG 压缩 + 高斯模糊**是对付"生成器指纹被后处理破坏"的关键手段，
其收益往往高于单纯加大数据量。

但本项目与那篇论文有一处**根本不同**：本模型带一路**频域分支**
（`src/models/freq_branch.py` + `dft.py`）。JPEG/模糊恰恰会抹掉高频伪影，
也就是说 —— **这两种增强对本模型可能是"双刃"：帮分类支路泛化，
却可能饿死频域支路**。所以这里刻意做成**可开关且默认关闭**，
必须用「同数据、同配置、只切这一个开关」的对照实验来判定，
**不得**在未验证的情况下默认打开并声称它有效。

配置写法（`configs/*.yaml` 的 `data:` 段）：

    data:
      augment:
        jpeg_prob: 0.5          # 每张图以 0.5 概率做一次 JPEG 重压缩；0=关
        jpeg_quality: [30, 90]  # 质量区间（整数）
        blur_prob: 0.3          # 每张图以 0.3 概率做一次高斯模糊；0=关
        blur_sigma: [0.0, 1.5]  # sigma 区间；取到 0 相当于不模糊
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # CLIP 官方均值
IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)   # CLIP 官方标准差


class Transform:
    """轻量图像变换（只用 numpy + cv2，避免引入 albumentations 依赖）。"""

    def __init__(self, size: int = 224, train: bool = False,
                 aug: Optional[dict] = None):
        self.size = size
        self.train = train
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

        # ---- 强增强开关（默认全关；见模块 docstring 的「双刃」说明）----
        a = aug or {}
        self.jpeg_prob = self._prob(a.get("jpeg_prob"))
        self.blur_prob = self._prob(a.get("blur_prob"))
        self.jpeg_quality = self._pair(a.get("jpeg_quality"), (30, 90))
        self.blur_sigma = self._pair(a.get("blur_sigma"), (0.0, 1.5))
        # 评测（train=False）时一律不增强 —— 否则同一张图两次评测分数不同，
        # 指标不可复现。这条比"省一个 if"重要得多，所以显式关掉。
        if not self.train:
            self.jpeg_prob = 0.0
            self.blur_prob = 0.0

    @staticmethod
    def _prob(v) -> float:
        """把配置里的概率归一成 [0,1] 的 float。

        ⚠ 不能写成 `min(max(v or 0.0, 0.0), 1.0)` —— v 是字符串（如 `"x"`）时
        `max("x", 0.0)` 会抛 `TypeError: '>' not supported between ...`，
        把"配置写错"变成"训练在数据加载期崩溃"。统一走 float() 再兜底。
        """
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if f != f:                       # NaN
            return 0.0
        return float(min(max(f, 0.0), 1.0))

    @staticmethod
    def _pair(v: Optional[Sequence], default: tuple) -> tuple:
        """把配置里的 [lo, hi] 归一成 (float, float)；非法值退回默认并保序。"""
        try:
            lo, hi = float(v[0]), float(v[1])
        except Exception:                       # noqa: BLE001
            return (float(default[0]), float(default[1]))
        return (lo, hi) if lo <= hi else (hi, lo)

    def _jpeg(self, img: np.ndarray) -> np.ndarray:
        """解到内存的 JPEG 重压缩（不落盘）。

        ⚠ 不做 BGR↔RGB 互换：`imencode` 按输入数组的既有顺序编码，
        `imdecode` 原样还原同一顺序 —— 这里进的是 RGB，出去仍是 RGB。
        若中途多套一次 `cvtColor`，反而会把红蓝通道**静默调换**。
        """
        import cv2

        q = int(round(np.random.uniform(self.jpeg_quality[0], self.jpeg_quality[1])))
        q = int(min(max(q, 1), 100))
        ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(img),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return img if dec is None else dec

    def _blur(self, img: np.ndarray) -> np.ndarray:
        import cv2

        sigma = float(np.random.uniform(self.blur_sigma[0], self.blur_sigma[1]))
        if sigma <= 1e-6:
            return img
        # 核宽取奇数且不小于 3（sigma=0 已被上面挡掉）
        k = max(3, int(2 * round(3.0 * sigma) + 1))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), sigma)

    def _resize_pair(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        """把 img（和可选的 mask）用**同一套几何变换**落到 size×size。

        ★★★ 2026-10-10 修复一个**一直在的严重 bug**：原实现里
              img  = self._resize(img)                    # 等比缩放到短边=size + 中心裁剪
              mask = cv2.resize(mask, (size, size))       # 整幅直接拉伸
            这是**两个不同的几何变换**。对非方形图，图像只保留原图**中心的
            min(h,w)×min(h,w) 方形**，掩码却覆盖 100% 原图 ⇒ 掩码与图像错位。

        实测（用"亮块+对应掩码"探针直接量图/掩码重合度）：
            224x224   → IoU 1.0000   （方形走同一分支，无问题）
            1024x1024 → IoU 0.9998
            256x384   → IoU 0.6636   ← 错位
            384x256   → IoU 0.6636
            600x800   → IoU 0.7576
        而本仓 CASIAv2 训练集只有 **0.2%** 是方形（可见面积占比 67.3%）
        ⇒ 定位支路**几乎全程在错位监督下训练**，且所有定位 mIoU 都是在这条
        管线下测的。修完必须重训定位支路，旧的定位数字与新数字**不可比**。

        为什么必须让 mask 跟着走同一条路：定位是像素级对应任务，
        图与掩码的空间对应关系是**唯一**的监督信号，任何不一致都会直接把
        监督信号变成噪声。
        """
        import cv2

        h, w = img.shape[:2]
        if h == w:
            interp = cv2.INTER_AREA if h > self.size else cv2.INTER_LINEAR
            img2 = cv2.resize(img, (self.size, self.size), interpolation=interp)
            m2 = (cv2.resize(mask.astype(np.float32), (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)
                  if mask is not None else None)
        else:
            # 保持长宽比缩放 + 中心裁剪，避免拉伸改变频域伪影的各向异性
            scale = self.size / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img2 = cv2.resize(img, (nw, nh), interpolation=interp)
            m2 = (cv2.resize(mask.astype(np.float32), (nw, nh),
                             interpolation=cv2.INTER_NEAREST)
                  if mask is not None else None)
            top = max(0, (nh - self.size) // 2)
            left = max(0, (nw - self.size) // 2)
            img2 = img2[top:top + self.size, left:left + self.size]
            if m2 is not None:
                m2 = m2[top:top + self.size, left:left + self.size]

        # 兜底：round 极端情况下裁剪后可能差 1 像素，保证输出严格 size×size
        if img2.shape[0] != self.size or img2.shape[1] != self.size:
            img2 = cv2.resize(img2, (self.size, self.size),
                              interpolation=cv2.INTER_LINEAR)
            if m2 is not None:
                m2 = cv2.resize(m2, (self.size, self.size),
                                interpolation=cv2.INTER_NEAREST)

        if m2 is not None:
            m2 = (m2 > 0.5).astype(np.float32)
        return img2, m2

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        """img: HxWx3 uint8 RGB；mask: HxW uint8 {0,1} 或 None。

        返回 (tensor (3,S,S) float32, mask_tensor (1,S,S) float32 或 None)
        """
        import cv2

        img, mask = self._resize_pair(img, mask)

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

            # ---- 强增强（默认关；见模块 docstring）----
            # 放在几何增强之后：JPEG/模糊是光度操作，与翻转/旋转可交换；
            # 放最后也让「几何增强 → 后处理降质」的顺序更贴近真实取证链路。
            # mask 不受影响（这两种操作都不移动像素）。
            if self.jpeg_prob > 0 and np.random.rand() < self.jpeg_prob:
                img = self._jpeg(img)
            if self.blur_prob > 0 and np.random.rand() < self.blur_prob:
                img = self._blur(img)

        x = img.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()

        m = None
        if mask is not None:
            m = torch.from_numpy(mask).unsqueeze(0).contiguous()
        return x, m


def build_transform(size: int = 224, train: bool = False,
                    aug: Optional[dict] = None) -> Transform:
    return Transform(size=size, train=train, aug=aug)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """反归一化回 [0,1]，用于可视化。输入 (3,H,W) 或 (B,3,H,W)。"""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1) if x.dim() == 4 \
        else torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1) if x.dim() == 4 \
        else torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)
