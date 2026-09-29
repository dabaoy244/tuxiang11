#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""摄取 COVERAGE（复制移动细粒度定位）到项目统一布局。

为什么需要这个脚本
------------------
`configs/*.yaml` 把 COVERAGE 同时声明进了 **train(120) / val(100) / test(100)**，
`scripts/prepare_datasets.py` 里也早就写好了「把 image+mask 平铺目录划分成
train/val/test」的逻辑 —— 但**没有任何脚本负责把它变成 `COVERAGE/{image,mask}/`**：
`scripts/fetch_tamper_datasets.py` 只做 CASIAv2。于是它长期处于
「配置里声明了、磁盘上却没有」的状态，而加载器对此**只打一行警告、不报错**
（典型的静默缺口：你以为评测覆盖了 3 个数据集，其实只有 2 个）。

官方源（都不支持命令行直取，必须人工下载）
------------------------------------------
原始仓库（**只有 README，图片在外链**）：https://github.com/wenbihan/coverage
  · OneDrive 文件夹（海外）：https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH
  · 百度网盘（国内，推荐）：https://pan.baidu.com/s/11i_swrFveLc9uZr1eR006Q  提取码 zduj
规模：100 张原始图 + 100 张复制移动篡改图（含重复区域掩码）。

实测结论（2026-09-29，本机与 AutoDL 实例双向验证，避免有人再白试一遍）：
  · GitHub 搜索 / HuggingFace 镜像 / ModelScope / Gitee **均无 COVERAGE 镜像**；
  · `onedrive.live.com` 在本机被代理拦（CONNECT tunnel 502）、在云端直连超时；
  · 百度网盘分享页需登录 + JS 渲染，curl 直取只会拿到 302；
  · 旧注释里写的 `github.com/wjctl/COVERAGE` **已 404**（死链）。
  ⇒ 所以本脚本的前提是：**由人下载，由脚本规范化**。

它做的三件事
------------
  1. 扫描你解压后的目录，**先报告**里面到底有什么（`--dry-run` 不写任何文件）；
  2. 容错地把 image 与 mask 配成对（去掉 `_gt` / `_forged` / `mask_` 等前后缀），
     并把掩码统一改名为「与图像主干同名 + .png」，消除后缀歧义；
  3. 产出扁平布局 `COVERAGE/{image,mask}/`，再调用 prepare_datasets 的成熟逻辑
     划分 train/val/test（默认 8:1:1）。

为什么必须"报告 + 断言"，不能直接猜
----------------------------------
COVERAGE 不同打包版本里掩码后缀是 `_gt.png` 或 `_forged.tif` 不等。若脚本默默按
「图像主干 + .png」去找掩码，就会**一张都配不上**，于是全部被判成"真实图 + 全黑掩码"，
训练照样跑完、mIoU 却恒为 0 —— 这与本项目踩过的 `split("_")[0]` 掩码键塌缩是同一类坑。
因此：配对数低于阈值时**直接失败退出**，并打印实际观测到的掩码文件名供人工核对。

用法
----
    # 0) 看现状（不写文件）
    python scripts/fetch_coverage.py --check

    # 1) 看清楚下载来的目录里有什么（不写文件）
    python scripts/fetch_coverage.py --from-dir D:/Downloads/COVERAGE --dry-run

    # 2) 正式摄取 + 规范化 + 划分（云上同理，把 --root 指到数据盘）
    python scripts/fetch_coverage.py --from-dir D:/Downloads/COVERAGE
    python scripts/fetch_coverage.py --archive /root/autodl-tmp/COVERAGE.zip \
        --workdir /root/autodl-tmp/_cov_tmp
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")

#: 目录名命中这些词 -> 该目录下的文件优先当作掩码
MASK_DIR_HINTS = ("mask", "masks", "gt", "groundtruth", "ground_truth", "ground-truth")

#: 掩码文件名可能带的**后缀**（不同打包版本的差异）
MASK_SUFFIXES = ("_gt_mask", "_mask_gt", "_gtmask", "_groundtruth",
                 "_gt", "_mask", "_forged", "-gt", "-mask", "-forged")

#: 掩码文件名可能带的**前缀**
MASK_PREFIXES = ("mask_", "gt_", "mask-", "gt-")

#: 篡改图常见前缀（仅用于**第二轮**模糊配对；遇到歧义必须报错而不是猜）
TAMPER_PREFIXES = ("tp_", "t_n_", "t_", "forged_", "fake_", "tamper_")
#: 真实图常见前缀
GENUINE_PREFIXES = ("ori_", "au_", "orig_", "original_", "real_", "genuine_")


# ==========================================================================
def list_images(folder: str) -> List[str]:
    out: List[str] = []
    for e in IMG_EXT:
        out += glob(os.path.join(folder, e)) + glob(os.path.join(folder, e.upper()))
    return sorted(set(out))


def walk_images(src: str) -> List[str]:
    """递归收集 src 下所有图片（按扩展名过滤，大小写不敏感）。"""
    exts = tuple(e[1:].lower() for e in IMG_EXT)
    out = []
    for dirpath, _dirnames, filenames in os.walk(src):
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _in_mask_dir(path: str, src: str) -> bool:
    """relpath 的任一目录段命中 MASK_DIR_HINTS -> 认为它放在掩码目录里。"""
    rel = os.path.relpath(path, src).replace("\\", "/").lower()
    parts = rel.split("/")[:-1]
    return any(p in MASK_DIR_HINTS for p in parts)


def _strip_mask_affixes(stem: str) -> List[str]:
    """由掩码文件名主干推导出「配对的图像主干」候选。"""
    s = stem.lower()
    out = [s]
    for suf in MASK_SUFFIXES:
        if s.endswith(suf):
            out.append(s[: -len(suf)])
    for pre in MASK_PREFIXES:
        if s.startswith(pre):
            out.append(s[len(pre):])
    for suf in MASK_SUFFIXES:                      # 前后缀可能同时存在
        for base in list(out):
            if base.endswith(suf):
                out.append(base[: -len(suf)])
    # 去重并保序，丢掉空串
    seen, uniq = set(), []
    for k in out:
        if k and k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def _strip_known_prefix(stem: str) -> Optional[str]:
    """图像主干去掉已知的真/假前缀（模糊配对用）。"""
    s = stem.lower()
    for pre in TAMPER_PREFIXES + GENUINE_PREFIXES:
        if s.startswith(pre):
            return s[len(pre):]
    return None


def is_mask_like(path: str, src: str) -> bool:
    """判定一个文件是否更像掩码而不是图像。

    判据（任一成立即算掩码）：放在 mask/gt 目录下；文件名带掩码后缀。
    注意：`.tif` 掩码也存在（prepare_datasets 注释里提到 `*_forged.tif`），
    所以**不能**用"位深/通道数"来判断 —— 那样会在不同版本间反复失效。
    """
    if _in_mask_dir(path, src):
        return True
    s = os.path.splitext(os.path.basename(path))[0].lower()
    return any(s.endswith(suf) for suf in MASK_SUFFIXES)


# ==========================================================================
def discover(src: str) -> Dict:
    """扫描 src，给出图像/掩码清单与配对结果（**只读，不写文件**）。"""
    files = walk_images(src)
    imgs, masks = [], []
    for p in files:
        (masks if is_mask_like(p, src) else imgs).append(p)

    # ---- 掩码索引：主干（去前后缀后）-> 掩码路径 --------------------------
    mask_by_key: Dict[str, str] = {}
    mask_dup: Dict[str, int] = {}
    for mp in masks:
        stem = os.path.splitext(os.path.basename(mp))[0]
        for k in _strip_mask_affixes(stem):
            if k in mask_by_key and mask_by_key[k] != mp:
                mask_dup[k] = mask_dup.get(k, 1) + 1
                continue                      # 撞键不覆盖，留给歧义报告
            mask_by_key.setdefault(k, mp)

    img_by_key: Dict[str, List[str]] = {}
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0].lower()
        img_by_key.setdefault(stem, []).append(ip)

    # ---- 第一轮：精确键匹配 ---------------------------------------------
    pair: Dict[str, str] = {}                  # 图像路径 -> 掩码路径
    used_mask = set()
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0].lower()
        if stem in mask_by_key:
            pair[ip] = mask_by_key[stem]
            used_mask.add(mask_by_key[stem])

    # ---- 第二轮：去前缀模糊匹配（歧义即放弃并记录）----------------------
    fuzzy = 0
    ambiguous: List[str] = []
    for mp in masks:
        if mp in used_mask:
            continue
        stem = os.path.splitext(os.path.basename(mp))[0]
        cand: List[str] = []
        for k in _strip_mask_affixes(stem):
            for ip in img_by_key.get(k, []):
                if ip not in pair:
                    cand.append(ip)
        # 也试：图像去掉 tp_/ori_ 前缀后与掩码键相同
        for key, ips in img_by_key.items():
            stripped = _strip_known_prefix(key)
            if stripped and stripped in _strip_mask_affixes(stem):
                cand += [ip for ip in ips if ip not in pair]
        cand = sorted(set(cand))
        if len(cand) == 1:
            pair[cand[0]] = mp
            used_mask.add(mp)
            fuzzy += 1
        elif len(cand) > 1:
            ambiguous.append(
                f"{os.path.basename(mp)} -> 同时匹配 {[os.path.basename(c) for c in cand]}")

    return {"src": src, "images": imgs, "masks": masks, "pair": pair,
            "unpaired_images": [p for p in imgs if p not in pair],
            "unpaired_masks": [p for p in masks if p not in used_mask],
            "mask_dup": mask_dup, "fuzzy": fuzzy, "ambiguous": ambiguous}


def _read_mask_binary(path: str, size: Optional[Tuple[int, int]] = None):
    """读掩码为 2D 二值图（0/255）。读不到就抛错（不静默跳过）。"""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"掩码读取失败：{path}")
    if size is not None and (m.shape[0], m.shape[1]) != size:
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def report(d: Dict, expected_real: int, expected_tampered: int) -> Tuple[int, int]:
    """打印发现报告，返回 (篡改数, 真实数)。篡改数为 0 时**不退出**，由调用方决定。"""
    print("=" * 78)
    print(f"扫描目录: {d['src']}")
    print(f"  图像候选（非掩码）: {len(d['images'])}")
    print(f"  掩码候选          : {len(d['masks'])}")
    print(f"  精确配对成功      : {len(d['pair']) - d['fuzzy']}")
    print(f"  去前缀模糊配对    : {d['fuzzy']}")

    tampered, genuine, unreadable = [], [], []
    for ip, mp in sorted(d["pair"].items()):
        try:
            m = _read_mask_binary(mp)
        except IOError as e:
            unreadable.append((ip, str(e)))
            continue
        (tampered if m.any() else genuine).append(ip)
    genuine = [p for p in d["unpaired_images"]] + genuine

    print(f"  -> 篡改图（掩码有正像素）: {len(tampered)}")
    print(f"  -> 真实图（无掩码或全黑）: {len(genuine)}")
    if unreadable:
        print(f"  [!] 掩码读取失败 {len(unreadable)} 个，例如：")
        for ip, msg in unreadable[:5]:
            print(f"      {os.path.basename(ip)}: {msg[:70]}")

    if d["mask_dup"]:
        print(f"  [!] 掩码键撞车 {len(d['mask_dup'])} 处（同一键对应多张掩码），已保留首个：")
        for k, n in list(d["mask_dup"].items())[:5]:
            print(f"      key={k!r} 命中 {n} 张")
    if d["ambiguous"]:
        print(f"  [!] 配对歧义 {len(d['ambiguous'])} 处（**已放弃配对**，未猜）：")
        for s in d["ambiguous"][:5]:
            print(f"      {s[:120]}")
    if d["unpaired_masks"]:
        print(f"  [!] 未被用到的掩码 {len(d['unpaired_masks'])} 个，例如：")
        for mp in d["unpaired_masks"][:5]:
            print(f"      {os.path.relpath(mp, d['src'])}")

    # ---- 关键守卫：一张都没配上 -> 直接失败，不产出一份"全是真实图"的垃圾 ----
    print("-" * 78)
    if len(d["masks"]) and not tampered:
        print("[X] 观测到掩码文件，却没有任何一张图与掩码配对成功（或掩码全黑）。")
        print("    这几乎必然是**命名/后缀不匹配**，而不是'数据里没有篡改图'。")
        print("    请把上面文件名贴出来核对；必要时给 _strip_mask_affixes 补后缀，")
        print("    **不要**直接继续 —— 否则会得到'篡改图被当成真实图 + 全黑掩码'的静默垃圾。")
        return 0, len(genuine)

    if expected_tampered and abs(len(tampered) - expected_tampered) > max(5, expected_tampered * 0.1):
        print(f"[!] 篡改图 {len(tampered)} 张，与官方规模 {expected_tampered} 张偏差较大，请核对。")
    if expected_real and abs(len(genuine) - expected_real) > max(5, expected_real * 0.1):
        print(f"[!] 真实图 {len(genuine)} 张，与官方规模 {expected_real} 张偏差较大，请核对。")
    print("=" * 78)
    return len(tampered), len(genuine)


# ==========================================================================
def ingest(d: Dict, base: str, mode: str = "copy") -> Dict:
    """把配好的对写成扁平布局 base/image + base/mask（掩码统一为同名 .png）。"""
    img_dir = os.path.join(base, "image")
    mask_dir = os.path.join(base, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    pair = d["pair"]
    stats = {"image": 0, "mask_from_src": 0, "mask_generated": 0, "skipped": 0}

    for ip in sorted(d["images"]):
        name = os.path.basename(ip)
        stem = os.path.splitext(name)[0]
        dst_img = os.path.join(img_dir, name)
        if not os.path.exists(dst_img):
            try:
                if mode == "link":
                    os.link(ip, dst_img)
                elif mode == "symlink":
                    os.symlink(os.path.abspath(ip), dst_img)
                else:
                    shutil.copy2(ip, dst_img)
            except OSError:
                shutil.copy2(ip, dst_img)
        stats["image"] += 1

        gray = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            stats["skipped"] += 1
            print(f"  [!] 图像读取失败，跳过：{ip}")
            continue
        h, w = gray.shape[:2]
        dst_mask = os.path.join(mask_dir, stem + ".png")
        mp = pair.get(ip)
        if mp:
            m = _read_mask_binary(mp, (h, w))
            cv2.imwrite(dst_mask, m)
            stats["mask_from_src"] += 1
        else:
            # 真实图没有掩码 -> 补全黑掩码，保证"每条样本都有 mask"
            cv2.imwrite(dst_mask, np.zeros((h, w), np.uint8))
            stats["mask_generated"] += 1
    return stats


def extract(archive: str, workdir: str) -> str:
    """解压 zip/tar 到 workdir，返回解压出的目录（自动向下钻一层）。"""
    os.makedirs(workdir, exist_ok=True)
    low = archive.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(workdir)
    elif low.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        import tarfile
        with tarfile.open(archive) as t:
            t.extractall(workdir)
    elif low.endswith(".rar"):
        raise SystemExit("[X] .rar 需要 unrar/7z，本机没有；请先手动解压后用 --from-dir")
    else:
        raise SystemExit(f"[X] 不认识的压缩格式：{archive}")
    # 自动向下钻：若只有唯一子目录且其中没有图片，就进去找
    cur = workdir
    for _ in range(4):
        entries = [e for e in os.listdir(cur) if not e.startswith(".")]
        subs = [e for e in entries if os.path.isdir(os.path.join(cur, e))]
        if len(entries) == 1 and len(subs) == 1:
            cur = os.path.join(cur, subs[0])
        else:
            break
    return cur


def ingest_archive(archive: str, workdir: str) -> str:
    return extract(archive, workdir)


def cmd_check(root: str) -> int:
    base = os.path.join(root, "COVERAGE")
    print(f"检查 {base}")
    if not os.path.isdir(base):
        print("  [缺] 目录不存在 —— 需要先人工下载官方包（见脚本头部说明）")
        return 1
    flat_ok = os.path.isdir(os.path.join(base, "image")) and \
        os.path.isdir(os.path.join(base, "mask"))
    print(f"  扁平布局 {{image,mask}}: {'有' if flat_ok else '无'}")
    total = 0
    for split in ("train", "val", "test"):
        idir = os.path.join(base, split, "image")
        mdir = os.path.join(base, split, "mask")
        if os.path.isdir(idir):
            n = len(list_images(idir))
            nm = len(list_images(mdir)) if os.path.isdir(mdir) else 0
            total += n
            print(f"  {split:<5}: image={n:<5} mask={nm}")
        else:
            print(f"  {split:<5}: 不存在")
    if flat_ok:
        print(f"  平铺 image={len(list_images(os.path.join(base, 'image')))} "
              f"mask={len(list_images(os.path.join(base, 'mask')))}（尚未划分）")
    print(f"  合计已就绪样本: {total}")
    return 0 if total else 1


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="摄取 COVERAGE 到项目统一布局（人工下载 + 本脚本规范化）")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "Datasets"),
                    help="数据集根目录（默认 data/Datasets）")
    ap.add_argument("--from-dir", default="", help="已解压的 COVERAGE 目录")
    ap.add_argument("--archive", default="", help="官方压缩包（zip/tar*），会解压到 --workdir")
    ap.add_argument("--workdir", default="", help="解压临时目录（默认 <root>/../_cov_tmp）")
    ap.add_argument("--mode", default="copy", choices=["copy", "link", "symlink"],
                    help="图像的搬运方式（掩码一律写新文件）")
    ap.add_argument("--dry-run", action="store_true", help="只扫描报告，不写任何文件")
    ap.add_argument("--no-split", action="store_true",
                    help="只做扁平化，不调用 prepare_datasets 划分 train/val/test")
    ap.add_argument("--check", action="store_true", help="只看现状")
    ap.add_argument("--expected-real", type=int, default=100)
    ap.add_argument("--expected-tampered", type=int, default=100)
    a = ap.parse_args()

    if a.check:
        return cmd_check(a.root)

    src = a.from_dir
    if a.archive:
        workdir = a.workdir or os.path.join(os.path.dirname(os.path.abspath(a.root)),
                                            "_cov_tmp")
        print(f"解压 {a.archive} -> {workdir} ...")
        src = ingest_archive(a.archive, workdir)
        print(f"  解压后目录: {src}")
    if not src:
        ap.error("需要 --from-dir 或 --archive（或 --check）")
    if not os.path.isdir(src):
        raise SystemExit(f"[X] 目录不存在：{src}")

    d = discover(src)
    n_t, n_r = report(d, a.expected_real, a.expected_tampered)
    if not n_t:
        return 2
    if a.dry_run:
        print("（--dry-run：未写入任何文件）")
        return 0

    base = os.path.join(a.root, "COVERAGE")
    if os.path.isdir(os.path.join(base, "image")):
        raise SystemExit(
            f"[X] {base}/image 已存在。为避免覆盖既有数据，请先人工确认后删除，"
            f"或改用 --root 指向别处。")
    print(f"\n写入扁平布局 -> {base}")
    st = ingest(d, base, a.mode)
    print(f"  image={st['image']}  掩码取自源={st['mask_from_src']}  "
          f"生成全黑掩码={st['mask_generated']}  跳过={st['skipped']}")

    if not a.no_split:
        print("\n划分 train/val/test（复用 prepare_datasets.split_tamper_dataset）")
        try:
            from scripts.prepare_datasets import split_tamper_dataset   # type: ignore
        except ImportError:
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            from prepare_datasets import split_tamper_dataset           # type: ignore
        split_tamper_dataset(base, mode=("link" if a.mode == "copy" else a.mode))
        for split in ("train", "val", "test"):
            idir = os.path.join(base, split, "image")
            if os.path.isdir(idir):
                print(f"  {split:<5}: image={len(list_images(idir))} "
                      f"mask={len(list_images(os.path.join(base, split, 'mask')))}")

    print("\n完成。建议接着跑：")
    print(f"  python scripts/prepare_datasets.py --root {a.root} --check")
    print(f"  python scripts/prepare_datasets.py --root {a.root} --manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
