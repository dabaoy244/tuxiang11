#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把「扁平的真/假文件夹对」改造成 ForenSynths 式布局，**并在改造的同时消除平凡可分线索**。

背景（为什么必须做这一步）
--------------------------
`scripts/audit_flat_realfake.py` 在同伴给的 smallai/smallnature 上测出：

    是否 PNG 格式         单特征 AUC = 1.0000   ← 一个 bit 就能全部分对
    每像素字节数(压缩率)  单特征 AUC = 0.9954
    原图分辨率            单特征 AUC = 0.9050

也就是说：**不学任何伪造线索**、只看「这文件是不是 PNG」就能拿到接近满分。
直接把它接进评测流程，得到的数字是假的。必须先统一格式 / 压缩 / 分辨率，
再重跑一遍审计确认平凡线索 AUC 落到 0.6 以下。

本脚本做两件事
--------------
1. `--mode normalized`（默认）：把两侧统一到
      同一边长（默认 512×512，真图按短边缩放后中心裁剪）
      同一编码器与参数（JPEG q=90, 4:2:0, 非渐进, optimize=False）
      同一色彩模式（RGB）、不带任何 EXIF/ICC 元数据
   输出成 `<out-root>/<split>/<gen-name>/{0_real,1_fake}/*.jpg`。
2. `--mode hardlink`：**不重编码**，把同一批源文件用硬链接挂成同样的布局。
   用途：作为「统一处理前」的基线，要求与 normalized 用**同一份配对计划**，
   这样 before/after 才是同一批样本、可逐一对比（见 --plan 参数）。

配对策略（默认 intersection + 逐类 1:1）
----------------------------------------
两侧的类别集合往往只重合一小部分（本批 14/86）。把「只在伪造侧出现的类别」
喂进去，模型完全可以学成「认得这是虎鲨 → 判伪造」而毫无伪造检测能力。
所以默认只取交集类别，且**逐类取 min(伪造数, 真实数)** 各取同样多张 ——
这样每个类别内部真假数量相等，**类别身份与真假标签彻底解耦**。

用法
----
    # 1) 生成「统一处理后」的评测域
    python scripts/build_gensynth_from_flat.py \
        --fake-dir "D:/QQ临时文件/smallai.zip/smallai/smallai" \
        --real-dir "D:/QQ临时文件/smallnature/smallnature" \
        --out-root data/Datasets/GenImageSDv14 --gen-name sdv14 \
        --mode normalized \
        --plan-out data/Datasets/GenImageSDv14/_meta/plan_intersection.json

    # 2) 用同一份配对计划挂出「统一处理前」的基线（硬链接，不占空间）
    python scripts/build_gensynth_from_flat.py \
        --fake-dir ... --real-dir ... \
        --out-root data/Datasets/GenImageSDv14_raw --gen-name sdv14raw \
        --mode hardlink \
        --plan-in data/Datasets/GenImageSDv14/_meta/plan_intersection.json

    # 3) 两侧都跑审计与评测
    python scripts/audit_flat_realfake.py --fake-dir <...>/1_fake --real-dir <...>/0_real ...
    python scripts/eval_cross_generator.py --data-root <...> --per-class 5000 ...

⚠ 硬链接要求源与目标在**同一个卷**上（本机都在 D:）。跨卷会自动退化为复制，
  脚本会打印实际使用的模式。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("需要 Pillow：pip install pillow", file=sys.stderr)
    raise

IMG_EXT = (".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".webp")
BAR = "=" * 74

# 统一的编码参数：两侧必须**逐字一致**，否则又制造出新的可分线索
JPEG_KW = dict(quality=90, subsampling=2, optimize=False, progressive=False)


def sec(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


# ---------------------------------------------------------------- 命名与扫描
def list_images(d: str) -> List[str]:
    if not os.path.isdir(d):
        raise SystemExit(f"目录不存在：{d}")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def parse_class(name: str, pattern: Optional[str]) -> Optional[str]:
    if pattern:
        m = re.match(pattern, name)
        if not m:
            return None
        gd = m.groupdict() or {}
        return gd.get("cls") or (m.group(1) if m.groups() else None)
    return name.split("_")[0]


def load_index_map(path: Optional[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                out[int(parts[0])] = parts[1]
    return out


# ---------------------------------------------------------------- 配对计划
def build_plan(fake_dir: str, real_dir: str, fake_re: Optional[str], real_re: Optional[str],
               idx2syn: Dict[int, str], classes: str, cap: Optional[int],
               min_px: int, seed: int) -> Tuple[List[dict], dict]:
    """返回 (pairs, stats)。

    pairs 每项：{"cls": synset, "fake": 源文件名, "real": 源文件名}
    只取交集类别时逐类 1:1；classes="all" 时单侧类别也保留（会产生类别-标签绑定，见文件头）。
    """
    fake_by: Dict[str, List[str]] = {}
    real_by: Dict[str, List[str]] = {}
    skipped_small = 0

    for name in list_images(fake_dir):
        k = parse_class(name, fake_re)
        if k is None:
            continue
        key = idx2syn.get(int(k), k) if k.isdigit() else k
        fake_by.setdefault(key, []).append(name)

    for name in list_images(real_dir):
        k = parse_class(name, real_re)
        if k is None:
            continue
        if min_px > 0:
            try:
                with Image.open(os.path.join(real_dir, name)) as im:
                    w, h = im.size
            except Exception:
                skipped_small += 1
                continue
            if min(w, h) * min(w, h) < min_px and w * h < min_px:
                skipped_small += 1
                continue
        real_by.setdefault(k, []).append(name)

    rng = random.Random(seed)
    for v in fake_by.values():
        rng.shuffle(v)
    for v in real_by.values():
        rng.shuffle(v)

    if classes == "intersection":
        keys = sorted(set(fake_by) & set(real_by))
    else:
        keys = sorted(set(fake_by) | set(real_by))

    pairs: List[dict] = []
    rows: List[dict] = []
    for k in keys:
        f, r = fake_by.get(k, []), real_by.get(k, [])
        if classes == "intersection":
            n = min(len(f), len(r))
        else:
            n = max(len(f), len(r))
        if cap is not None:
            n = min(n, cap)
        pf, pr = f[:n], r[:n]
        if classes == "all":
            # 单侧类别：另一侧为空，长度按所在一侧算
            pf = f[:n]
            pr = r[:n]
        rows.append({"cls": k, "avail_fake": len(f), "avail_real": len(r),
                     "used_fake": len(pf), "used_real": len(pr)})
        for i in range(max(len(pf), len(pr))):
            pairs.append({"cls": k,
                          "fake": pf[i] if i < len(pf) else None,
                          "real": pr[i] if i < len(pr) else None})

    stats = {
        "fake_dir": fake_dir, "real_dir": real_dir,
        "classes_mode": classes, "seed": seed,
        "n_classes_fake": len(fake_by), "n_classes_real": len(real_by),
        "n_classes_used": len(keys),
        "skipped_real_too_small": skipped_small,
        "per_class": rows,
        "n_fake": sum(1 for p in pairs if p["fake"]),
        "n_real": sum(1 for p in pairs if p["real"]),
    }
    return pairs, stats


# ---------------------------------------------------------------- 图像变换
def normalize_image(path: str, size: int, degrade: str) -> Image.Image:
    """读入 → RGB → 短边缩放到 size → 中心裁剪 size×size →（可选）降采样再升采样。

    对两侧施加**完全相同**的算子。真图本来分辨率各异、伪造图固定 512，
    统一到同一尺寸后「分辨率」「长宽比」两个线索同时消失。
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
        im = im.resize((nw, nh), Image.BICUBIC)
        left = (nw - size) // 2
        top = (nh - size) // 2
        im = im.crop((left, top, left + size, top + size))
        if degrade == "updown":
            # 共同低通：把 8×8 / 16×16 那一类高频格式指纹在两图上一起抹掉。
            # 代价是图像变糊，所以只在实测仍有强线索时才用。
            half = max(64, size // 2)
            im = im.resize((half, half), Image.BICUBIC).resize((size, size), Image.BICUBIC)
        return im.copy()


def out_name(src_name: str, label: int) -> str:
    stem = os.path.splitext(src_name)[0]
    return f"{stem}.jpg"


# ---------------------------------------------------------------- 落盘
def link_or_copy(src: str, dst: str) -> str:
    """硬链接优先；跨卷/不支持时退化为复制。返回实际模式。"""
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        import shutil
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把扁平真/假文件夹对改造成 ForenSynths 式布局，并消除平凡可分线索")
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-class-regex", default=r"^(?P<cls>\d+)_")
    ap.add_argument("--real-class-regex", default=r"^(?P<cls>n\d+)_")
    ap.add_argument("--index-map", default="data/Datasets/_meta/imagenet_index_synset.txt",
                    help="`索引 synset` 映射文件（每行一条），用于把数字前缀翻成 synset")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gen-name", required=True)
    ap.add_argument("--mode", choices=["normalized", "hardlink"], default="normalized")
    ap.add_argument("--classes", choices=["intersection", "all"], default="intersection")
    ap.add_argument("--cap-per-class", type=int, default=None,
                    help="每类每侧最多取多少张（默认不设，取 min(伪造,真实)）")
    ap.add_argument("--min-px", type=int, default=0,
                    help="真实图最小像素数下限，低于此值的直接跳过（避免极端放大）")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--degrade", choices=["none", "updown"], default="none")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--plan-out", default=None, help="把配对计划写到这个 json")
    ap.add_argument("--plan-in", default=None,
                    help="复用已存在的配对计划（保证 before/after 是同一批样本）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root_abs = os.path.abspath(args.out_root)
    gen_dir = os.path.join(root_abs, args.split, args.gen_name)
    d_real = os.path.join(gen_dir, "0_real")
    d_fake = os.path.join(gen_dir, "1_fake")
    meta_dir = os.path.join(root_abs, "_meta")

    sec("1. 配对计划")
    if args.plan_in and os.path.exists(args.plan_in):
        blob = json.load(open(args.plan_in, encoding="utf-8"))
        pairs, stats = blob["pairs"], blob["stats"]
        print(f"  复用计划 {args.plan_in}")
    else:
        pairs, stats = build_plan(args.fake_dir, args.real_dir,
                                  args.fake_class_regex, args.real_class_regex,
                                  load_index_map(args.index_map), args.classes,
                                  args.cap_per_class, args.min_px, args.seed)
    print(f"  伪造侧类别 {stats['n_classes_fake']}，真实侧 {stats['n_classes_real']}，"
          f"实际使用 {stats['n_classes_used']} 个（mode={stats['classes_mode']}）")
    if stats["skipped_real_too_small"]:
        print(f"  真实侧因过小而跳过 {stats['skipped_real_too_small']} 张")
    n_f = sum(1 for p in pairs if p["fake"])
    n_r = sum(1 for p in pairs if p["real"])
    print(f"  计划：伪造 {n_f} 张 / 真实 {n_r} 张，合计 {n_f + n_r} 张")
    if stats["classes_mode"] == "intersection":
        print("  ✅ 逐类 1:1 —— 每个类别内部真假数量相等，类别身份与真假标签已解耦")
    else:
        print("  ⚠ classes=all：只出现在一侧的类别会让「类别身份」与真假标签绑定，")
        print("     指标可能被捷径抬高；论文里必须显式声明这一限制。")
    if args.dry_run:
        print("\n  --dry-run：不写任何文件，退出。")
        return 0

    sec("2. 写出")
    os.makedirs(d_real, exist_ok=True)
    os.makedirs(d_fake, exist_ok=True)
    for f in os.listdir(d_real):
        os.remove(os.path.join(d_real, f))
    for f in os.listdir(d_fake):
        os.remove(os.path.join(d_fake, f))
    print(f"  目标：{gen_dir}\n  mode={args.mode}  size={args.size}  degrade={args.degrade}")

    manifest, modes = [], set()
    n_skip = 0
    for i, p in enumerate(pairs):
        for label, key, src_dir, dst_dir in ((0, "real", args.real_dir, d_real),
                                            (1, "fake", args.fake_dir, d_fake)):
            name = p[key]
            if not name:
                continue
            src = os.path.join(src_dir, name)
            dst = os.path.join(dst_dir, out_name(name, label))
            rec = {"src": src, "dst": dst, "label": label, "cls": p["cls"]}
            try:
                if args.mode == "hardlink":
                    rec["link"] = link_or_copy(src, dst)
                    modes.add(rec["link"])
                else:
                    img = normalize_image(src, args.size, args.degrade)
                    img.save(dst, format="JPEG", **JPEG_KW)
                    rec["out_w"], rec["out_h"] = img.size
                    rec["out_bytes"] = os.path.getsize(dst)
                    rec["src_bytes"] = os.path.getsize(src)
                    with Image.open(src) as im:
                        rec["src_fmt"] = (im.format or "?").upper()
                        rec["src_w"], rec["src_h"] = im.size
                manifest.append(rec)
            except Exception as exc:              # 坏图必须报出来，不能静默
                n_skip += 1
                print(f"  ❌ 失败 {name}：{str(exc)[:80]}")
        if (i + 1) % 500 == 0:
            print(f"    进度 {i + 1}/{len(pairs)}")

    print(f"  写出 {len(manifest)} 个文件（失败 {n_skip}）"
          + (f"  链接模式：{sorted(modes)}" if modes else ""))

    sec("3. 元数据落盘")
    os.makedirs(meta_dir, exist_ok=True)
    tag = f"{args.gen_name}_{args.mode}"
    mf_path = os.path.join(meta_dir, f"manifest_{tag}.json")
    blob = {"args": vars(args), "stats": stats,
            "jpeg_kw": JPEG_KW if args.mode == "normalized" else None,
            "manifest": manifest}
    with open(mf_path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=2)
    print(f"  {mf_path}")
    if args.plan_out:
        with open(args.plan_out, "w", encoding="utf-8") as fh:
            json.dump({"pairs": pairs, "stats": stats}, fh,
                      ensure_ascii=False, indent=2)
        print(f"  {args.plan_out}")

    sec("4. 下一步（必须做，顺序不能反）")
    print(f"  1) 审计新目录，确认平凡线索 AUC 落到 0.6 以下：")
    print(f"     python scripts/audit_flat_realfake.py \\")
    print(f"         --fake-dir \"{os.path.relpath(d_fake)}\" \\")
    print(f"         --real-dir \"{os.path.relpath(d_real)}\" \\")
    print(f"         --fake-class-regex '^(?P<cls>\\\\d+)_' \\")
    print(f"         --real-class-regex '^(?P<cls>n\\\\d+)_' --sample 600")
    print(f"  2) 接进评测（脚本按 <root>/<生成器>/**/{{0_real,1_fake}} 递归找，"
          f"无需改 configs）：")
    print(f"     python scripts/eval_cross_generator.py \\")
    print(f"         --config configs/ablation_b16_pretrained.yaml \\")
    print(f"         --ckpt checkpoints/abp_best.pt \\")
    print(f"         --data-root {os.path.relpath(os.path.join(root_abs, args.split))} \\")
    print(f"         --per-class 5000 --no-dedup-real --tag gensynth_{args.gen_name}")
    print(f"  3) ⚠ 本批出处为 GenImage 的 train split —— **只做零样本测试域，"
          f"永远不要用它训练**，否则就是数据泄漏。")
    print()
    return 0 if n_skip == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
