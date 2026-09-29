#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 COVERAGE 从「数据集目录」搬成「外部测试域」。

为什么需要这一步
----------------
`data/` 整个被 .gitignore 忽略（数据集不入库），所以「外部域在哪儿、怎么来的」
必须靠一个**可复现的脚本**表达。`scripts/fetch_coverage.py` 只负责把官方数据
弄进 `data/Datasets/COVERAGE/`；本脚本负责把它变成**独立外部域**。

为什么必须搬到别处（不是洁癖，是防静默污染）
--------------------------------------------
`build_dataloaders` 会按配置里的 `data.root` 去找**每一个**出现在
train/val/test 里的数据集名。只要 `data/Datasets/COVERAGE` 存在，任何把 COVERAGE
写回配置的动作都会让**正在排队的消融臂**多喂一个数据集 ⇒ 各臂 val/test 口径不再
一致 ⇒ 整张消融表作废。放到 `data/external/` 下就物理上不可能被扫到。

用法
----
    python scripts/stage_coverage_external.py            # 从 data/Datasets/COVERAGE 搬
    python scripts/stage_coverage_external.py --check    # 只校验，不写
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 官方 COVERAGE 100 对；剔除 9 对「掩码与图像尺寸不符」后 = 100 真 + 91 篡改
EXPECT_IMAGE = 191
EXPECT_MASK = 191


def main() -> int:
    ap = argparse.ArgumentParser(description="把 COVERAGE 搬成独立外部测试域")
    ap.add_argument("--src", default="data/Datasets/COVERAGE",
                    help="源：fetch_coverage.py 整理出来的数据集目录")
    ap.add_argument("--dst", default="data/external/COVERAGE",
                    help="目标：外部域目录（**绝不能在 data/Datasets/ 下**）")
    ap.add_argument("--check", action="store_true", help="只校验，不复制")
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)

    # ---- 守卫 1：目标不能在训练域根下（这是本脚本存在的唯一理由） ----------------
    rel_dst = os.path.relpath(dst, ROOT).replace("\\", "/")
    if rel_dst.split("/")[:2] == ["data", "Datasets"]:
        print(f"❌ 目标 {rel_dst} 落在 data/Datasets/ 下 —— 会被训练/消融链路按 root 扫到，"
              f"污染各臂 val/test 口径。请改到 data/external/。")
        return 2

    if not os.path.isdir(src):
        print(f"❌ 源目录不存在：{src}\n   先跑 scripts/fetch_coverage.py 把官方数据弄进来。")
        return 1

    out_img = os.path.join(dst, "test", "image")
    out_msk = os.path.join(dst, "test", "mask")

    if args.check:
        for d, n in ((out_img, EXPECT_IMAGE), (out_msk, EXPECT_MASK)):
            got = len(os.listdir(d)) if os.path.isdir(d) else 0
            flag = "✓" if got == n else "❌"
            print(f"{flag} {os.path.relpath(d, ROOT)}: {got}（期望 {n}）")
        return 0

    # ---- 搬运：源目录根下的 image/ 与 mask/ 就是「全部 191 张」 -------------------
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)
    copied = {"image": 0, "mask": 0}
    for sub, out in (("image", out_img), ("mask", out_msk)):
        s = os.path.join(src, sub)
        if not os.path.isdir(s):
            print(f"❌ 源缺 {os.path.relpath(s, ROOT)}")
            return 1
        for f in sorted(os.listdir(s)):
            sp = os.path.join(s, f)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(out, f))
                copied[sub] += 1

    n_img = len(os.listdir(out_img))
    n_msk = len(os.listdir(out_msk))
    print(f"[搬运] {os.path.relpath(src, ROOT)} → {rel_dst}")
    print(f"  image {copied['image']} 个文件 → 目标共 {n_img}")
    print(f"  mask  {copied['mask']} 个文件 → 目标共 {n_msk}")

    bad = []
    if n_img != EXPECT_IMAGE:
        bad.append(f"image {n_img} ≠ {EXPECT_IMAGE}")
    if n_msk != EXPECT_MASK:
        bad.append(f"mask {n_msk} ≠ {EXPECT_MASK}")
    if bad:
        print("❌ 数量不符： " + "；".join(bad))
        print("   说明源数据不完整；不要将就着用，先核对 data/Datasets/COVERAGE。")
        return 1

    print("✅ 外部域就绪。评测：")
    print("   python scripts/eval_external_domain.py --domain COVERAGE --ckpt <ckpt> --expect 191")
    return 0


if __name__ == "__main__":
    sys.exit(main())
