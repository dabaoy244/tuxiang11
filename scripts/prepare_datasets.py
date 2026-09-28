"""数据集获取与规范化组织（申报书 3.3 / 6.3 测试数据集）

三个公开标准数据集：
  ① ForenSynths    —— 跨生成模型整图真伪；20 类约 72 万张训练图
                      （4 类子集约 14.4 万张），验证 8000，测试约 9 万张（13 种生成器）
  ② CASIA v2       —— 传统篡改（拼接 / 复制移动 / 修图），7491 真 + 5123 篡改（合计 12614）
  ③ COVERAGE       —— 复制移动细粒度定位，100 真 + 100 篡改

本脚本做三件事：
  1) 检查各数据集的落盘情况，打印缺失项与官方获取渠道；
  2) 把原始布局**规范化**为项目统一布局（见 src/data/datasets.py 顶部注释）；
  3) 划分 train/val/test 并输出 manifest.json（记录每条的来源、标签、掩码路径），
     便于论文里写"数据划分可复现"。

用法：
    # 下载（另有独立脚本，支持断点续传与镜像）
    python scripts/fetch_datasets.py --split val progan_test
    # 检查 / 规范化 / 生成清单
    python scripts/prepare_datasets.py --root data/Datasets --check
    python scripts/prepare_datasets.py --root data/Datasets --organize --mode link
    python scripts/prepare_datasets.py --root data/Datasets --manifest

注意：
  * 各数据集版权归原作者所有，仅可用于学术研究；请遵守各自 License。
  * ForenSynths **无需向作者申请**，官方已托管 HuggingFace 公开下载：
        https://huggingface.co/datasets/sywang/CNNDetection
    本项目用 scripts/fetch_datasets.py 拉取（断点续传 + hf-mirror 镜像 + 选择性解压）。
    （2026.09.18 更正：早期注释里"需要向作者申请""约 10 万张"均为错误信息。）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from glob import glob
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")

SOURCES = {
    "ForenSynths": {
        "desc": "跨生成模型整图真伪；20 类约 72 万张训练（4 类子集约 14.4 万），"
                "验证 8000，测试约 9 万张",
        "source": "https://huggingface.co/datasets/sywang/CNNDetection "
                  "（官方公开，无需申请；用 scripts/fetch_datasets.py 下载）",
        # 训练/验证：<split>/<类别>/{0_real,1_fake}/ ；测试：test/<生成器>/[<类别>/]{0_real,1_fake}/
        # ⚠ train 必须带 <split> 这一层：GenSynthsDataset(split="train") 只认
        #   <base>/train（见 src/data/datasets.py 的注释），少这一层会**一条样本都扫不到**。
        "layout": "{root}/ForenSynths/{train,val}/<class>/{0_real,1_fake}/  "
                  "+ {root}/ForenSynths/test/<generator>/[<class>/]{0_real,1_fake}/",
        "download": "python scripts/fetch_datasets.py --split val progan_test test train",
    },
    "CASIAv2": {
        "desc": "传统篡改：拼接 / 复制移动 / 修图（7491 真 + 5123 篡改，共 12614；"
                "篡改 = 3274 复制移动 + 1849 拼接）",
        "source": "GitHub: namtpham/casia2groundtruth（推荐，标注已修正）；"
                  "或 Kaggle 镜像（注意部分版本不含 GT 掩码，做不了定位）；"
                  "或 CASIA v2.0 官方站点",
        "layout": "{root}/CASIAv2/<split>/{image,mask}/",
    },
    "COVERAGE": {
        "desc": "复制移动细粒度定位（100 真 + 100 篡改）",
        "source": "https://github.com/wjctl/COVERAGE（掩码后缀随版本为 "
                  "*_forged.tif 或 *_gt.png，落盘后先确认）",
        "layout": "{root}/COVERAGE/<split>/{image,mask}/",
    },
}

SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}


def _list_images(folder: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(folder):
        return out
    for e in IMG_EXT:
        out += glob(os.path.join(folder, e)) + glob(os.path.join(folder, e.upper()))
    return sorted(set(out))


def _relink(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


# ==========================================================================
def _scan_real_fake(base_dir: str, root: str, counts: Dict[str, int]) -> None:
    """自动探测任意层级下的 0_real / 1_fake 配对目录（官方布局层级不固定）。"""
    if not os.path.isdir(base_dir):
        return
    for real in glob(os.path.join(base_dir, "**", "0_real"), recursive=True):
        if not os.path.isdir(real):
            continue
        rel = os.path.relpath(os.path.dirname(real), root).replace("\\", "/")
        n_real = len(_list_images(real))
        if n_real:
            counts[f"{rel}/0_real"] = n_real
        fake = os.path.join(os.path.dirname(real), "1_fake")
        n_fake = len(_list_images(fake))
        if n_fake:
            counts[f"{rel}/1_fake"] = n_fake


def check(root: str) -> Dict[str, dict]:
    print(f"\n数据集根目录：{os.path.abspath(root)}\n" + "-" * 70)
    report = {}
    for name, meta in SOURCES.items():
        base = os.path.join(root, name)
        exists = os.path.isdir(base)
        counts: Dict[str, int] = {}
        if name == "ForenSynths":
            # 官方布局层级不固定（训练/验证按类别，测试按生成器），全量递归探测
            for d in (os.path.join(root, "ForenSynths"),
                      os.path.join(root, "ForenSynths_test")):
                _scan_real_fake(d, root, counts)
        elif exists:
            for split in ("train", "val", "test"):
                n = len(_list_images(os.path.join(base, split, "image")))
                if n:
                    counts[f"{split}/image"] = n
        total = sum(counts.values())
        status = "已就绪" if total > 0 else ("目录存在但为空" if exists else "缺失")
        report[name] = {"status": status, "counts": counts, "total": total}
        print(f"【{name}】{status}  {meta['desc']}")
        print(f"        获取：{meta['source']}")
        if "download" in meta:
            print(f"        下载：{meta['download']}")
        # 布局串里含 {0_real,1_fake} 这类字面花括号，不能用 str.format，改用 replace
        print(f"        期望布局：{meta['layout'].replace('{root}', root)}")
        if counts:
            print(f"        现有 {len(counts)} 个子集，合计 {total} 张：")
            for k in sorted(counts)[:12]:
                print(f"          {k}/  {counts[k]}")
            if len(counts) > 12:
                print(f"          ...（另有 {len(counts) - 12} 个子集未列出）")
        print()
    return report


# ==========================================================================
def split_tamper_dataset(base: str, mode: str = "link", seed: int = 3407) -> None:
    """把 CASIAv2 / COVERAGE 的 image+mask 平铺目录划分成 train/val/test。"""
    src_img = os.path.join(base, "image")
    src_mask = os.path.join(base, "mask")
    if not os.path.isdir(src_img):
        return
    imgs = _list_images(src_img)
    if not imgs:
        return
    rng = random.Random(seed)
    # 按"是否篡改"分层抽样，保证各 split 的正负比例一致
    pos, neg = [], []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        mp = next((os.path.join(src_mask, stem + e) for e in (".png", ".jpg", ".bmp")
                   if os.path.exists(os.path.join(src_mask, stem + e))), None)
        m_found = False
        if mp:
            try:
                import cv2

                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                m_found = m is not None and (m > 127).any()
            except Exception:
                m_found = True
        (pos if m_found else neg).append((p, mp))

    rng.shuffle(pos)
    rng.shuffle(neg)
    for tag, items in (("pos", pos), ("neg", neg)):
        n = len(items)
        n_tr = int(n * SPLIT_RATIO["train"])
        n_va = int(n * SPLIT_RATIO["val"])
        buckets = {"train": items[:n_tr],
                   "val": items[n_tr:n_tr + n_va],
                   "test": items[n_tr + n_va:]}
        for split, sub in buckets.items():
            for p, mp in sub:
                stem = os.path.splitext(os.path.basename(p))[0]
                _relink(p, os.path.join(base, split, "image", os.path.basename(p)), mode)
                if mp:
                    _relink(mp, os.path.join(base, split, "mask",
                                             os.path.basename(mp)), mode)
                else:
                    # 真实图像没有掩码文件 -> 生成全黑掩码，保持"每条都有 mask"
                    import numpy as np
                    import cv2

                    out = os.path.join(base, split, "mask", stem + ".png")
                    if not os.path.exists(out):
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        h, w = img.shape if img is not None else (256, 256)
                        cv2.imwrite(out, np.zeros((h, w), np.uint8))


def organize(root: str, mode: str = "link") -> None:
    print(f"\n规范化组织数据集（mode={mode}）")
    for name in ("CASIAv2", "COVERAGE"):
        base = os.path.join(root, name)
        if os.path.isdir(base):
            split_tamper_dataset(base, mode)
            print(f"  [ok] {name} 已划分 train/val/test")

    foren = os.path.join(root, "ForenSynths")
    if os.path.isdir(foren):
        # 官方布局 <generator>/<split>/{0_real,1_fake} 已经能被 GenSynthsDataset 直接读取，
        # 这里只做一致性检查与合并（可选）。
        n = 0
        for p in glob(os.path.join(foren, "*", "*", "0_real")):
            n += len(_list_images(p))
        print(f"  [ok] ForenSynths 检测到 {n} 张真实图（官方层级可直接使用）")


# ==========================================================================
def manifest(root: str, out: str = "") -> str:
    """生成 manifest.json：每条样本的路径 / 标签 / 掩码 / 来源 / split。"""
    recs = []
    for name in ("ForenSynths", "CASIAv2", "COVERAGE"):
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            continue
        for split in ("train", "val", "test"):
            img_dir = os.path.join(base, split, "image")
            if os.path.isdir(img_dir):
                mask_dir = os.path.join(base, split, "mask")
                for p in _list_images(img_dir):
                    stem = os.path.splitext(os.path.basename(p))[0]
                    mp = next((os.path.join(mask_dir, stem + e)
                               for e in (".png", ".jpg", ".bmp")
                               if os.path.exists(os.path.join(mask_dir, stem + e))), None)
                    label = 0
                    if mp:
                        import cv2

                        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                        label = int(m is not None and (m > 127).any())
                    recs.append({"path": os.path.relpath(p, root), "label": label,
                                 "mask": os.path.relpath(mp, root) if mp else None,
                                 "source": name, "split": split, "task": "tamper"})
                continue
            for cls, lab in (("0_real", 0), ("1_fake", 1)):
                for p in _list_images(os.path.join(base, split, cls)):
                    recs.append({"path": os.path.relpath(p, root), "label": lab,
                                 "mask": None, "source": name, "split": split,
                                 "task": "gensynth"})

    out = out or os.path.join(root, "manifest.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"root": os.path.abspath(root), "n": len(recs), "records": recs},
                  f, ensure_ascii=False, indent=2)
    stats: Dict[str, Dict[str, int]] = {}
    for r in recs:
        k = f"{r['source']}/{r['split']}"
        stats.setdefault(k, {"total": 0, "fake": 0})
        stats[k]["total"] += 1
        stats[k]["fake"] += r["label"]
    print(f"\nmanifest 已写入 {out}（共 {len(recs)} 条）")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} total={v['total']:6d}  fake={v['fake']:6d}  "
              f"real={v['total']-v['fake']:6d}")
    return out


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "Datasets"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--organize", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--mode", default="link", choices=["link", "symlink", "copy"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    if not any([args.check, args.organize, args.manifest]):
        args.check = args.manifest = True

    if args.check:
        check(args.root)
    if args.organize:
        organize(args.root, args.mode)
    if args.manifest:
        manifest(args.root, args.out)


if __name__ == "__main__":
    main()
