#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用**图像内容**实证「ImageNet 索引 ↔ synset」的锚点，并拼接成便于肉眼核对的对照图。

为什么要做这一步
----------------
两侧类别对齐完全依赖一张「索引 → synset」映射表。若表里有错，
算出来的「交集 14 类」就是假的，由它自建的「内容可控测试域」也就不成立。
而这张表**很难从外部可靠地取回**：本机只有模型中介的取回通道，
实测取回结果里出现了 `n021258`、`n021260` 这种**结构非法**的 synset
（合法 synset 是 `n` + 8 位数字），说明取回过程本身会截断/错位。

所以改用**自证**的方式：数据自己就带着答案 ——
  伪造侧文件名前缀 = 索引，真实侧文件名前缀 = synset。
若是同一个 index 与同一个 synset 指向同一类物体，
那么「伪造侧第 k 个索引的图」与「真实侧对应 synset 的图」**应该看起来是同一类东西**。

本脚本为交集类别的每一对拼一行 `[伪造 | 真实]` 的对照图，供人工肉眼核对。
**只读，不改动任何数据。**

用法
----
    python scripts/verify_intersection_classes.py \
        --fake-dir "D:/QQ临时文件/smallai.zip/smallai/smallai" \
        --real-dir "D:/QQ临时文件/smallnature/smallnature"
"""
from __future__ import annotations

import argparse
import os
import re
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    print("需要 Pillow", file=sys.stderr)
    raise

IMG_EXT = (".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".webp")
CELL = 220
PAD = 6
LABEL_H = 22


def main() -> int:
    ap = argparse.ArgumentParser(description="用图像内容实证索引↔synset 的锚点")
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--index-map", default="data/Datasets/_meta/imagenet_index_synset.txt")
    ap.add_argument("--out-dir", default="outputs/class_verify")
    ap.add_argument("--per-montage", type=int, default=6, help="每张对照图放几行（默认 6）")
    args = ap.parse_args()

    idx2syn = {}
    for line in open(args.index_map, encoding="utf-8"):
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():
            idx2syn[int(p[0])] = p[1]

    fake_files, real_files = {}, {}
    for f in os.listdir(args.fake_dir):
        m = re.match(r"^(\d+)_", f)
        if m and f.lower().endswith(IMG_EXT):
            fake_files.setdefault(int(m.group(1)), []).append(f)
    for f in os.listdir(args.real_dir):
        m = re.match(r"^(n\d+)_", f)
        if m and f.lower().endswith(IMG_EXT):
            real_files.setdefault(m.group(1), []).append(f)

    # 交集：伪造侧索引经映射表翻成 synset 后，与真实侧 synset 对比
    inter = sorted({idx2syn[k] for k in fake_files if k in idx2syn}
                   & set(real_files))
    print(f"映射表 {len(idx2syn)} 条；伪造侧 {len(fake_files)} 类 / 真实侧 "
          f"{len(real_files)} 类；交集 {len(inter)} 类")

    syn2idx = {v: k for k, v in idx2syn.items()}
    rows = []
    for syn in inter:
        idx = syn2idx.get(syn)
        if idx is None or idx not in fake_files:
            print(f"  ⚠ {syn} 无法反查索引，跳过")
            continue
        # 两侧都固定取排序后的第一张，保证可复现
        ff = sorted(fake_files[idx])[0]
        rf = sorted(real_files[syn])[0]
        rows.append((idx, syn, ff, rf))
    if not rows:
        print("❌ 没有可核对的类别")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    made = []
    for page in range(0, len(rows), args.per_montage):
        chunk = rows[page:page + args.per_montage]
        W = PAD * 3 + CELL * 2
        H = PAD + len(chunk) * (LABEL_H + CELL + PAD)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for i, (idx, syn, ff, rf) in enumerate(chunk):
            y = PAD + i * (LABEL_H + CELL + PAD)
            dr.text((PAD, y + 4), f"index {idx}  <->  {syn}     左: 伪造 {ff}   右: 真实 {rf}",
                    fill=(0, 0, 0))
            for j, (path, fmt) in enumerate(((os.path.join(args.fake_dir, ff), "png"),
                                            (os.path.join(args.real_dir, rf), "jpeg"))):
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        s = min(w, h)
                        im = im.crop(((w - s) // 2, (h - s) // 2,
                                      (w - s) // 2 + s, (h - s) // 2 + s))
                        im = im.resize((CELL, CELL), Image.BICUBIC)
                except Exception as exc:
                    print(f"  ❌ 打不开 {path}: {str(exc)[:60]}")
                    continue
                canvas.paste(im, (PAD + j * (CELL + PAD), y + LABEL_H))
        out = os.path.join(args.out_dir, f"class_check_{page // args.per_montage + 1}.png")
        canvas.save(out)
        made.append(out)
        print(f"  已写出 {out}（{len(chunk)} 行）")

    print(f"\n共 {len(rows)} 对，{len(made)} 张对照图 -> {args.out_dir}")
    print("请逐行肉眼核对：左右两张是否属于**同一个类别**。")
    print("若某行明显不是同一类 → 该条映射有误，交集不成立，必须修正映射表后重跑。")
    print("核对完成后，把已验证的锚点写回映射表头部的注释里。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
