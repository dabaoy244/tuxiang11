"""数据集：ForenSynths（跨生成模型真伪）/ CASIA v2（拼接·复制移动·修图）/ COVERAGE（复制移动）

目录规范（由 scripts/prepare_datasets.py 统一转换生成，见 docs/03）：

    data/Datasets/
    ├── ForenSynths/                     # 整图真伪，无像素掩码
    │   ├── train/<class>/{0_real,1_fake}/*.png|jpg      # 20 类 LSUN 真实 + ProGAN 生成
    │   ├── val/<class>/{0_real,1_fake}/*                # 官方 held-out ProGAN
    │   └── test/<generator>/<class>/{0_real,1_fake}/*   # 13 种生成器
    ├── CASIAv2/                          # 传统篡改，有像素掩码
    │   ├── train/image/*.jpg   train/mask/*.png
    │   ├── val/...
    │   └── test/...
    └── COVERAGE/
        ├── train/{image,mask}/...
        ├── val/...
        └── test/...

兼容性：ForenSynths 的官方层级并不统一，本类会**递归探测任意深度**的
`0_real` / `1_fake` 配对目录，因此以下三种都能识别：

    <split>/<class>/{0_real,1_fake}              （本项目规范化后的布局）
    <split>/<generator>/{0_real,1_fake}          （官方 test/ 的生成器层）
    <split>/<generator>/<class>/{0_real,1_fake}  （官方 test/progan/ 的多类别层）

⚠️ 只有 `<base>/<split>/` 存在时才会用 `<base>` 作回退（兼容旧的扁平布局）。
不要把 `<base>` 无条件当回退，否则 `split="train"` 在 `train/` 缺失时会扫到
`val/` 与 `test/`，把验证集和测试集静默混进训练集。
"""

from __future__ import annotations

import os
import random
from glob import glob
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transform

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")


def _list_images(folder: str) -> List[str]:
    out: List[str] = []
    for e in IMG_EXT:
        out.extend(glob(os.path.join(folder, e)))
        out.extend(glob(os.path.join(folder, e.upper())))
    return sorted(set(out))


def _read_image(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:                     # 中文路径兜底
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: str) -> np.ndarray:
    import cv2

    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.float32)


def _mask_keys(stem: str) -> List[str]:
    """由文件名主干生成候选匹配键。

    CASIA / COVERAGE 的掩码常带 `_gt` 后缀（如 `xxx_00138_gt.png` → 键 `xxx_00138`），
    而图像名不带。所以两个方向都试：原名 + 去掉 `_gt` 后的名字。
    """
    s = stem.lower()
    out = [s]
    if s.endswith("_gt"):
        out.append(s[:-3])
    return out


#: CASIA 的篡改图命名前缀：Tp_（拼接/复制移动）、T_N_（修图，v2 里已并入）
_TAMPER_PREFIXES = ("tp_", "t_n_", "t_")


def _looks_tampered(filename: str) -> bool:
    """按命名规范判断文件名是否自称"篡改图"。

    · CASIA：`Tp_` / `T_N_` / `t_` 前缀；
    · COVERAGE 官方：`<i>t.tif`（如 `37t.tif`）—— 主干为「纯数字 + 结尾 t」。

    ⚠ 第二类曾经缺席：COVERAGE 的官方命名不匹配任何 CASIA 前缀，于是
    `TamperDataset` 里"篡改图缺掩码 ⇒ 报错跳过、绝不静默当真实图"的安全网
    **对 COVERAGE 完全不生效** —— 一旦掩码缺一个，那张篡改图会被当成真实图
    （正样本喂成负样本）且没有任何提示。纯字符串判断即可，不必引入 re。
    """
    low = filename.lower()
    stem = os.path.splitext(low)[0]
    if stem.endswith("t") and stem[:-1].isdigit():
        return True
    return any(low.startswith(p) for p in _TAMPER_PREFIXES)


# ==========================================================================
class BaseDataset(Dataset):
    kind = "base"

    def __init__(self, size: int = 224, train: bool = False):
        self.tf = build_transform(size, train)
        self.samples: List[dict] = []
        self.layout: Optional[str] = None      # 供排查用的布局标记

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = _read_image(s["image"])
        mask = _read_mask(s["mask"]) if s.get("mask") else None
        x, m = self.tf(img, mask)
        return {
            "image": x,
            "label": int(s["label"]),
            "mask": m,
            "has_mask": bool(s.get("mask")),
            "name": os.path.basename(s["image"]),
            "kind": self.kind,
        }


# ==========================================================================
def _interleave_labels(samples: list, seed: int = 3407) -> list:
    """把「正负样本各成一块」的列表改成交错排列（正负交替，块内各自打乱）。

    为什么需要
    ----------
    `GenSynthsDataset` 先扫 `0_real` 再扫 `1_fake`，`TamperDataset` 先扫 `Au` 再扫
    `Tp` —— 两者建出来的样本列表都是**前段全真、后段全伪造**。于是用
    `--max-batches N` 做小规模评测时，取到的永远是前面的真图。
    实测：CASIAv2 取前 128 张时 `n_tampered=0`，三个 mIoU 口径全 0，
    看起来像"模型完全不会定位"，其实只是**没抽到任何篡改图**。

    交错后任意前缀都近似 1:1，部分评测才有代表性。
    完整评测不受影响 —— 指标本身与样本顺序无关。
    """
    rng = random.Random(seed)
    pos = [s for s in samples if s.get("label") == 1]
    neg = [s for s in samples if s.get("label") != 1]
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos):
            out.append(pos[i])
        if i < len(neg):
            out.append(neg[i])
    return out


# ==========================================================================
class GenSynthsDataset(BaseDataset):
    """ForenSynths：GAN / 扩散 / 自回归 多生成模型整图真伪检测（1=伪造, 0=真实）。"""

    kind = "gensynth"

    def __init__(self, root: str, name: str = "ForenSynths", split: str = "train",
                 size: int = 224, max_samples: Optional[int] = None, train: bool = False,
                 include_generators: Optional[Sequence[str]] = None,
                 exclude_generators: Optional[Sequence[str]] = None,
                 max_per_generator: Optional[int] = None):
        """参数 `include_generators` / `exclude_generators` / `max_per_generator`
        用于**构造跨生成器验证集（val_cross）**：

            val_sets:
              - name: ForenSynths
                kind: gensynth
                split: test                      # ← 生成器信息只存在于 test/ 这一层
                include_generators: [biggan, cyclegan, stargan]
                max_per_generator: 300           # 每生成器真/假各取 300

        「生成器名」= 相对 `<split>/` 的**第一层目录名**（ForenSynths 官方把生成器
        放在这一层，其下才是 `0_real` / `1_fake` 或类别目录）。若布局是扁平的
        `<split>/{0_real,1_fake}`，第一层就成了 `0_real`/`1_fake`，此时筛选键不是
        生成器名 —— 所以下面做了硬失败保护，避免筛出空集还照跑。
        """
        super().__init__(size, train)
        base = os.path.join(root, name)

        # 候选根目录：优先 <base>/<split>，其次兼容旧的扁平布局 <base>/{0_real,1_fake}。
        # ⚠ 注意不要把 <base> 无条件当作回退——否则 split="train" 在 train/ 缺失时
        #   会递归扫到 val/ 和 test/，把验证集和测试集静默混进训练集。
        candidates: List[str] = []
        split_dir = os.path.join(base, split)
        if os.path.isdir(split_dir):
            candidates.append(split_dir)
        if (os.path.isdir(os.path.join(base, "0_real"))
                or os.path.isdir(os.path.join(base, "1_fake"))):
            candidates.append(base)

        real_list: List[str] = []
        fake_list: List[str] = []
        chosen: Optional[str] = None
        for c in candidates:
            real_list = _list_images(os.path.join(c, "0_real"))
            fake_list = _list_images(os.path.join(c, "1_fake"))
            if not real_list and not fake_list:
                # 官方层级不固定，递归任意深度：
                #   <class>/{0_real,1_fake}
                #   <generator>/<class>/{0_real,1_fake}
                #   <generator>/<split>/{0_real,1_fake}
                for real_dir in sorted(glob(os.path.join(c, "**", "0_real"),
                                            recursive=True)):
                    if not os.path.isdir(real_dir):
                        continue
                    real_list += _list_images(real_dir)
                    fake_list += _list_images(
                        os.path.join(os.path.dirname(real_dir), "1_fake"))
            if real_list or fake_list:
                chosen = c
                break

        # ---- 生成器级筛选 / 每生成器限额（构造 val_cross 用）------------------
        if (include_generators or exclude_generators or max_per_generator) and chosen:
            real_list, fake_list = self._filter_and_cap(
                real_list, fake_list, chosen,
                include_generators, exclude_generators, max_per_generator)

        for p in real_list:
            self.samples.append({"image": p, "label": 0, "mask": None})
        for p in fake_list:
            self.samples.append({"image": p, "label": 1, "mask": None})

        self._maybe_subsample(max_samples)

    @staticmethod
    def _generator_of(path: str, root: str) -> str:
        """样本相对 `<split>/` 的第一层目录名（小写），即生成器名。"""
        rel = os.path.relpath(path, root).replace("\\", "/")
        return rel.split("/")[0].lower()

    def _filter_and_cap(self, real_list, fake_list, root,
                        include, exclude, max_per_generator):
        """按生成器筛选并限额，附带**反静默失效**校验。

        返回 (real_list, fake_list)。筛没了会直接抛错 —— 而不是让下游
        "训练/评测正常跑完、只是集合变小、指标还更好看"。
        """
        inc = {g.lower() for g in include} if include else None
        exc = {g.lower() for g in exclude} if exclude else set()
        found = sorted({self._generator_of(p, root)
                        for p in (real_list + fake_list)})
        # ⚠ 括号不能省：集合的 `-` 比 `|` 结合更紧，写成
        #   `(inc or set()) | exc - set(found)` 会把 inc 里的**全部**合法名字
        #   都误判成"不存在"，直接抛错。
        unknown = sorted(((inc or set()) | exc) - set(found))
        if unknown:
            raise ValueError(
                f"[GenSynthsDataset] 生成器筛选里有盘上不存在的名字：{unknown}；"
                f"实际存在：{found}（根目录 {root}）。"
                f"若这里列出的是 0_real/1_fake 或类别名，说明该 split 是扁平布局，"
                f"不含生成器层级，不能用 include/exclude_generators 筛选。")

        def keep(p: str) -> bool:
            g = self._generator_of(p, root)
            if inc is not None and g not in inc:
                return False
            return g not in exc

        n_before = len(real_list) + len(fake_list)
        real_list = [p for p in real_list if keep(p)]
        fake_list = [p for p in fake_list if keep(p)]
        if n_before and not (real_list or fake_list):
            raise ValueError(
                f"[GenSynthsDataset] 生成器筛选把 {n_before} 张全部剔除了，"
                f"include={include} exclude={exclude}，盘上有 {found}。")

        if max_per_generator:
            rng = random.Random(3407)

            def cap(paths: List[str]) -> List[str]:
                by_gen: Dict[str, List[str]] = {}
                for p in paths:
                    by_gen.setdefault(self._generator_of(p, root), []).append(p)
                out: List[str] = []
                for g, ps in sorted(by_gen.items()):
                    rng.shuffle(ps)
                    out.extend(ps[:max_per_generator])
                return out

            real_list, fake_list = cap(real_list), cap(fake_list)

        # ★ 把「要了什么 / 实际拿到什么」打出来。构造 val_cross 时最容易犯的错
        #   就是把 include 写成别名（如 style_gan），结果只筛出 1 个生成器，
        #   而验证集看上去"还是能算 AUC"，问题直到中期检查才暴露。
        kept_gens = sorted({self._generator_of(p, root)
                            for p in (real_list + fake_list)})
        rel_root = os.sep.join(os.path.normpath(root).split(os.sep)[-2:])
        print(f"[GenSynthsDataset] {rel_root}: "
              f"生成器 {len(kept_gens)}/{len(found)} 个 {kept_gens}"
              f"，样本 真 {len(real_list)} / 假 {len(fake_list)}")
        return real_list, fake_list

    def _maybe_subsample(self, max_samples: Optional[int]) -> None:
        samples = self.samples
        if max_samples and len(samples) > max_samples:
            rng = random.Random(3407)
            pos = [s for s in samples if s["label"] == 1]
            neg = [s for s in samples if s["label"] == 0]
            half = max_samples // 2
            rng.shuffle(pos); rng.shuffle(neg)
            samples = pos[:half] + neg[:half]
        # 无论是否抽样都交错：否则前段全真、后段全伪造，
        # 用 --max-batches 做部分评测时抽不到正类样本（见 _interleave_labels）。
        self.samples = _interleave_labels(samples)


class TamperDataset(BaseDataset):
    """传统篡改数据集（CASIA v2 / COVERAGE）：图 + 像素级 GT 掩码。"""

    kind = "tamper"

    def __init__(self, root: str, name: str, split: str = "train", size: int = 224,
                 max_samples: Optional[int] = None, train: bool = False):
        super().__init__(size, train)
        base = os.path.join(root, name)
        cand = [os.path.join(base, split), base]

        # 支持三种落盘布局（官方 CASIA / COVERAGE 与整理后的版本各不相同）：
        #   A 已整理： <c>/image/ + <c>/mask/      （prepare_datasets.py 的产物）
        #   B 三分目录：<c>/Tp/ + <c>/Au/ + <c>/{Gt,Groundtruth}/   （ModelScope 版 CASIAv2）
        #   C 原始平铺：<c>/ 下直接是 Au_*.jpg / Tp_*.jpg，掩码在 <c>/Groundtruth/
        # 早期实现只认 `c/image,mask` 和"c 下直接有 Tp/"，于是布局 B/C 一个都吃不到，
        # 而 `img_dir` 又被当成"含 Tp 的那个目录本身"，在布局 B 下会列出 0 张图。
        image_dirs: List[tuple] = []          # [(目录, 先验标签 or None)]
        mask_dir: Optional[str] = None

        for c in cand:
            if not os.path.isdir(c):
                continue
            if os.path.isdir(os.path.join(c, "image")) and \
                    os.path.isdir(os.path.join(c, "mask")):
                image_dirs = [(os.path.join(c, "image"), None)]
                mask_dir = os.path.join(c, "mask")
                break
            if os.path.isdir(os.path.join(c, "Tp")):
                for sub, lab in (("Tp", 1), ("T_N", 1), ("Au", 0)):
                    d = os.path.join(c, sub)
                    if os.path.isdir(d):
                        image_dirs.append((d, lab))
                mask_dir = next((os.path.join(c, m) for m in
                                 ("Groundtruth", "Gt", "gt", "mask")
                                 if os.path.isdir(os.path.join(c, m))), None)
                break
            if _list_images(c):
                image_dirs = [(c, None)]
                mask_dir = next((os.path.join(c, m) for m in
                                 ("Groundtruth", "Gt", "gt")
                                 if os.path.isdir(os.path.join(c, m))), None)
                break

        if not image_dirs:
            return
        self.layout = ("prepared" if mask_dir and "image" in (image_dirs[0][0] or "")
                       else "casia_raw")

        # ---- 掩码索引 ----------------------------------------------------
        # ⚠ 这里曾经用过 `stem.split("_")[0]` 作键，是个**静默失效**的严重 bug：
        #   CASIA 的文件名形如
        #       Tp_D_CND_M_N_ani00018_sec00096_00138.tif        （篡改图）
        #       Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png     （对应掩码）
        #   取第一个下划线之前永远是 "Tp"/"Au"，于是：
        #     * 5123 个掩码**全部塌缩成一个键 "tp"** → 每张篡改图都匹配到
        #       `mask_index` 里最后写入的那一张掩码（全部一样且错误）；
        #     * 真实图 Au_* 的键是 "au"，掩码集里没有 → 7491 张真实图**被整个跳过**。
        #   结果是 dataset 里只剩"全部共用同一张掩码的篡改图"，定位支路训的是垃圾，
        #   mIoU 无论怎么调都不可能对。
        #   改为用「完整文件名（去扩展名，再去掉 `_gt` 后缀）」精确匹配。
        mask_index: Dict[str, str] = {}
        if mask_dir:
            for mp in _list_images(mask_dir):
                stem = os.path.splitext(os.path.basename(mp))[0]
                for k in _mask_keys(stem):
                    mask_index.setdefault(k, mp)

        missing_mask = 0
        for d, prior in image_dirs:
            for ip in _list_images(d):
                fname = os.path.basename(ip)
                stem = os.path.splitext(fname)[0]
                mp = next((mask_index[k] for k in _mask_keys(stem)
                           if k in mask_index), None)

                if mp is None:
                    # 没有对应掩码。真实图属正常（定位支路用 mask_valid=False 过滤）；
                    # 但若它应是篡改图（目录先验或文件名前缀），说明掩码集不全，
                    # **不能静默当成真图**（那等于把正样本喂成负样本），计数后跳过。
                    if prior == 1 or (prior is None and _looks_tampered(fname)):
                        missing_mask += 1
                        continue
                    self.samples.append({"image": ip, "label": 0, "mask": None})
                    continue

                m = _read_mask(mp)
                self.samples.append({"image": ip, "label": 1 if m.sum() > 0 else 0,
                                     "mask": mp})

        self.missing_mask = missing_mask
        if missing_mask:
            print(f"[TamperDataset] ⚠️ {name}/{split}: {missing_mask} 张应为篡改的图"
                  f"找不到掩码，已跳过（请检查掩码目录是否完整）")
        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        n_neg = len(self.samples) - n_pos
        if self.samples:
            print(f"[TamperDataset] {name}/{split}: 载入 {len(self.samples)} 条"
                  f"（篡改 {n_pos} / 真实 {n_neg}，掩码 {len(mask_index)} 个）")

        samples = self.samples
        if max_samples and len(samples) > max_samples:
            rng = random.Random(3407)
            pos = [s for s in samples if s["label"] == 1]
            neg = [s for s in samples if s["label"] == 0]
            rng.shuffle(pos); rng.shuffle(neg)
            half = max_samples // 2
            samples = pos[:half] + neg[:half]
        # 无论是否抽样都交错：否则列表是「前面全 Au 真图、后面全 Tp 篡改图」，
        # 用 --max-batches 做部分评测时 `n_tampered=0`，mIoU 全 0（见 _interleave_labels）。
        self.samples = _interleave_labels(samples)


class DemoDataset(BaseDataset):
    """离线演示数据集（由 src/data/synth.py 生成），用于无网络/无 GPU 时跑通全流程。"""

    kind = "demo"

    def __init__(self, root: str, split: str = "train", size: int = 224,
                 max_samples: Optional[int] = None, train: bool = False):
        super().__init__(size, train)
        img_dir = os.path.join(root, split, "image")
        mask_dir = os.path.join(root, split, "mask")
        for ip in _list_images(img_dir):
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = None
            for e in (".png", ".jpg"):
                cand = os.path.join(mask_dir, stem + e)
                if os.path.exists(cand):
                    mp = cand
                    break
            m = _read_mask(mp) if mp else None
            label = 1 if (m is not None and m.sum() > 0) else 0
            self.samples.append({"image": ip, "label": label, "mask": mp})
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]


# ==========================================================================
class MultiTaskDataset(Dataset):
    """把多个子数据集拼成一个，按 `kind` 区分是否带掩码。"""

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        self.index: List[tuple] = []
        for di, ds in enumerate(self.datasets):
            for i in range(len(ds)):
                self.index.append((di, i))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        di, i = self.index[idx]
        return self.datasets[di][i]

    def stats(self) -> dict:
        out = {}
        for ds in self.datasets:
            name = getattr(ds, "name", ds.__class__.__name__)
            pos = sum(1 for s in getattr(ds, "samples", []) if s["label"] == 1)
            out[f"{name}({ds.kind})"] = {"total": len(ds), "fake": pos,
                                         "real": len(ds) - pos}
        return out


def collate_multitask(batch: List[dict]) -> dict:
    """混合批次：有掩码的样本正常堆叠，无掩码的补零并用 `mask_valid` 标记。"""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    has_mask = torch.tensor([b["has_mask"] for b in batch], dtype=torch.bool)

    masks = None
    if has_mask.any():
        c, h, w = batch[0]["image"].shape
        masks = torch.zeros(len(batch), 1, h, w)
        for i, b in enumerate(batch):
            if b["mask"] is not None:
                masks[i] = b["mask"]
    return {
        "image": images,
        "label": labels,
        "mask": masks,
        "mask_valid": has_mask,
        "name": [b["name"] for b in batch],
        "kind": [b["kind"] for b in batch],
    }


# ==========================================================================
def _has_real_fake(d: str) -> bool:
    """`d/0_real` 或 `d/1_fake` 是否为目录（**只判一层**，不递归）。"""
    return (os.path.isdir(os.path.join(d, "0_real"))
            or os.path.isdir(os.path.join(d, "1_fake")))


def layout_hint(root: str, wanted: list) -> str:
    """空 split 报错时附上「我找了哪个目录、而盘上实际有什么」。

    为什么值得写：train 被解到 `ForenSynths/<类>/`（而不是 `ForenSynths/train/<类>/`）
    时，数据**明明在盘上**却一条都扫不到。只报「未找到任何样本」会把人引向
    「是不是没下完」，而真正的问题是**层级放错了一层**。2026-09-24 上云实测遇到。
    """
    out: List[str] = []
    for name, sp in [(n, s) for n, s, _, _ in wanted]:
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            out.append(f"  · {name}/{sp}：{base} 不存在（数据还没下？）")
            continue
        out.append(f"  · {name}/{sp}：找过 {base}/{sp}"
                   f"，其次扁平布局 {base}/{{0_real,1_fake}}")
        subs = sorted(x for x in os.listdir(base)
                      if os.path.isdir(os.path.join(base, x)))
        out.append(f"    实际存在：{subs if subs else '（空）'}")
        # 只有当 <base>/<split> **缺失或为空**时，才把根下的 {0_real,1_fake} 子目录
        # 当作"被放错层的本 split 数据"。否则（例如 val 已就位时）这条提示会变成
        # **错误建议** —— 按它做就是把 train 的类搬进 val/，那是灾难。
        sp_dir = os.path.join(base, sp)
        sp_ready = os.path.isdir(sp_dir) and bool(os.listdir(sp_dir))
        wrong = [] if sp_ready else [x for x in subs
                                     if x != sp and _has_real_fake(os.path.join(base, x))]
        if wrong:
            out.append(f"    ⚠ {base}/{{{','.join(wrong)}}}/ 下有 {{0_real,1_fake}}，"
                       f"但 split=\"{sp}\" 只认 {base}/{sp}/ —— 若这确实是该 split 的数据，"
                       f"它被放错了一层。归位命令：")
            out.append(f"      cd {base} && mkdir -p {sp} && mv {' '.join(wrong)} {sp}/")
    return "\n".join(out)


# ==========================================================================
def build_dataloaders(cfg: dict, use_demo: bool = False) -> Dict[str, DataLoader]:
    """构建 train / val / test 三个 DataLoader。

    use_demo=True 时使用离线演示数据集（无 GPU、无公开数据集也能跑通）。
    """
    d = cfg["data"]
    size = d.get("image_size", 224)
    bs = cfg["train"].get("batch_size", 16)
    nw = d.get("num_workers", 4)

    def make(sets, split, train: bool, batch_size: int, shuffle: bool):
        ds_list = []
        # ⚠️ 逐条记录"配置里要了什么 / 实际加载到多少"，最后打出来。
        #    历史上这里会把空数据集**静默丢掉**（见下面那行过滤），
        #    后果是：你以为在 COVERAGE 上训练，其实压根没加载，训练却"正常"跑完。
        #    这类静默失效只能靠把数字摆到眼前来防。
        wanted = []
        for s in sets:
            # 配置项自带的 split 优先于 DataLoader 的角色名（train/val/test）。
            # 这样才表达得了「拿 val 当训练集、拿 test 当验证集」这类实验协议，
            # 例如 configs/lite_realval.yaml。不写 split 时行为与原来完全一致。
            sp = s.get("split") or split
            if use_demo or s.get("kind") == "demo":
                ds = DemoDataset(d["demo"]["root"], sp, size,
                                 s.get("max_samples"), train)
            elif s["kind"] == "gensynth":
                ds = GenSynthsDataset(d["root"], s["name"], sp, size,
                                      s.get("max_samples"), train,
                                      include_generators=s.get("include_generators"),
                                      exclude_generators=s.get("exclude_generators"),
                                      max_per_generator=s.get("max_per_generator"))
            else:
                ds = TamperDataset(d["root"], s["name"], sp, size,
                                   s.get("max_samples"), train)
            wanted.append((s["name"], sp, len(ds), s.get("kind", "?")))
            ds_list.append(ds)

        loaded = [x for x in ds_list if len(x) > 0]
        empty = [(n, sp, k) for (n, sp, n_i, k) in wanted if n_i == 0]
        if empty:
            print(f"[data] ⚠ split={split}：配置了 {len(empty)} 个数据集但**一个样本都没有**"
                  f"，已跳过 —— {[f'{n}({sp})' for n, sp, _ in empty]}")
            print("[data]   这不会报错，但意味着这些数据**没有参与**这次训练/评测。"
                  "请核对 data/Datasets/ 下的目录名与布局。")
            # ★ 关键：把"该往哪儿放"直接算出来印给用户，而不是让他自己猜。
            #   2026-09-24 云上实测：train 被解到 ForenSynths/<类>/ 时，只打一行 ⚠，
            #   训练会"正常"跑完但少掉 14.4 万张训练图 —— 必须顺手给出归位命令。
            print(layout_hint(d["root"], [(n, sp, 0, k) for n, sp, k in empty]))
        if not loaded:
            raise RuntimeError(
                f"split={split} 未找到任何样本（配置了 {len(wanted)} 个数据集，全部为空）。"
                f"请先运行 scripts/prepare_datasets.py，或使用 --demo 走离线演示数据集。"
                f"已配置：{[(n, sp, k) for n, sp, _, k in wanted]}\n"
                + layout_hint(d["root"], wanted)
            )
        ds = loaded[0] if len(loaded) == 1 else MultiTaskDataset(loaded)
        total = sum(len(x) for x in loaded)
        detail = "、".join(f"{n}({sp}) {c} 条" for n, sp, c, _ in wanted if c > 0)
        print(f"[data] split={split:<5s} 共 {total} 条 ← {detail}")
        if hasattr(ds, "name") is False:
            try:
                ds.name = split
            except Exception:
                pass
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=nw,
            pin_memory=d.get("pin_memory", True) and torch.cuda.is_available(),
            collate_fn=collate_multitask, drop_last=train and len(ds) > batch_size,
            persistent_workers=nw > 0,
        )

    if use_demo:
        demo_set = [{"name": "demo", "kind": "demo", "max_samples": None}]
        train_sets, val_sets, test_sets = demo_set, demo_set, demo_set
    else:
        train_sets = d["train_sets"]
        val_sets = d.get("val_sets") or [dict(s, split="val") for s in d["train_sets"]]
        test_sets = d["test_sets"]

    return {
        "train": make(train_sets, "train", True, bs, True),
        "val": make(val_sets, "val", False, cfg["eval"].get("batch_size", 32), False),
        "test": make(test_sets, "test", False, cfg["eval"].get("batch_size", 32), False),
    }
