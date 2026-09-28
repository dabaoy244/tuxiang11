#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""获取篡改定位数据集（CASIA v2.0），并整理成本仓库需要的目录布局。

为什么需要这个脚本
------------------
`mIoU ≥ 56%` 这个指标依赖**像素级标注**，只有 CASIA v2 / COVERAGE 这类数据集才有。
但仓库原先缺两样东西，导致定位支路从未跑通过真实数据：

  1. **没有下载器** —— `scripts/fetch_datasets.py` 只覆盖 ForenSynths；
     CASIA 的获取方式在文档里只写了「GitHub 镜像」，而 `github.com` 在本网络下
     DNS 直接解析失败。
  2. **没有布局转换** —— 官方/CASIAv2 的目录是 `Au/`（真图）+ `Tp/`（篡改）
     + `Gt/`（掩码），而 `TamperDataset` 的主路径吃的是
     `<split>/{image,mask}/`，中间缺一步转换。

数据源
------
ModelScope 上的勘误版（`huggingface.co` 在本网络不可达，ModelScope 可达且国内直连）：
    https://modelscope.cn/datasets/Sunnyhaze/CASIAv2-Manipulated-image

实测内容与规模（与多篇文献交叉核对一致）：
    Au/  7491 张真实图像      345.7 MB
    Tp/  5123 张篡改图像 .tif  3.00 GB
    Gt/  5123 张掩码 .png       6.7 MB
    合计 12614 张 / 3.28 GB —— 即 CASIA v2.0 的 7491 真实 + 5123 篡改

产出布局
--------
    data/Datasets/CASIAv2/
        train/{image,mask}/   80%
        val/{image,mask}/     10%
        test/{image,mask}/    10%
    掩码统一重命名为「与图像同名的 .png」（去掉官方的 `_gt` 后缀），
    真实图像补一张全黑掩码 —— 这样定位支路对"无篡改"也有负样本监督。

用法
----
    python scripts/fetch_tamper_datasets.py                 # 下载 + 整理
    python scripts/fetch_tamper_datasets.py --download-only
    python scripts/fetch_tamper_datasets.py --organize-only  # 只整理（已下过）
    python scripts/fetch_tamper_datasets.py --check          # 只看现状
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import ssl
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MS_DATASET = "Sunnyhaze/CASIAv2-Manipulated-image"
MS_API = "https://www.modelscope.cn/api/v1/datasets"
UA = {"User-Agent": "Mozilla/5.0 (compatible; vibnet-dataset-fetcher/1.0)"}
CTX = ssl.create_default_context()

#: 期望规模（用于校验下载完整性）
EXPECT = {"Au": 7491, "Tp": 5123, "Gt": 5123}


# --------------------------------------------------------------------------
def ms_tree(recursive: bool = True, page_size: int = 3000) -> list:
    """拉取 ModelScope 仓库文件树（服务端分页上限 3000，需翻页）。"""
    out, seen = [], set()
    for page in range(1, 40):
        url = (f"{MS_API}/{MS_DATASET}/repo/tree?Revision=master"
               f"&Recursive={'true' if recursive else 'false'}"
               f"&PageSize={page_size}&PageNumber={page}")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
            files = json.loads(r.read())["Data"]["Files"]
        if not files:
            break
        new = [f for f in files if f["Path"] not in seen]
        for f in new:
            seen.add(f["Path"])
        out += new
        if len(files) < page_size:
            break
    return out


def ms_download(relpath: str, dest: str) -> tuple:
    """下载单个文件（带大小校验）。返回 (relpath, ok, 说明)。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return relpath, True, "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    q = urllib.parse.quote(relpath, safe="/")
    url = f"{MS_API}/{MS_DATASET}/repo?Revision=master&FilePath={q}"
    tmp = dest + ".part"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1024 * 512)
            if os.path.getsize(tmp) == 0:
                raise IOError("空文件")
            os.replace(tmp, dest)
            return relpath, True, "ok"
        except Exception as e:                       # noqa: BLE001
            if attempt == 3:
                return relpath, False, f"{type(e).__name__}: {str(e)[:60]}"
            time.sleep(2 * (attempt + 1))
    return relpath, False, "unreachable"


def download_all(raw_dir: str, workers: int) -> dict:
    files = [f for f in ms_tree() if f["Type"] == "blob"]
    # 只取三大目录，跳过 .gitattributes / 脚本
    todo = []
    for f in files:
        top = f["Path"].split("/")[0]
        if top in ("Au", "Tp", "Gt"):
            todo.append(f["Path"])
    total_bytes = sum(f.get("Size", 0) for f in files
                      if f["Path"].split("/")[0] in ("Au", "Tp", "Gt"))
    print(f"  远端文件 {len(todo)} 个，共 {total_bytes / 1024 ** 3:.2f} GB")

    done = failed = skipped = 0
    t0 = time.time()
    got_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ms_download, p, os.path.join(raw_dir, p)): p
                for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rel, ok, msg = fut.result()
            if ok:
                done += 1
                if msg == "skip":
                    skipped += 1
                else:
                    fp = os.path.join(raw_dir, rel)
                    if os.path.exists(fp):
                        got_bytes += os.path.getsize(fp)
            else:
                failed += 1
                if failed <= 5:
                    print(f"\n  [FAIL] {rel}  {msg}")
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                rate = got_bytes / max(el, 1e-6) / 1024 ** 2
                print(f"\r    已完成 {i}/{len(todo)}  "
                      f"（新下 {done - skipped} / 跳过 {skipped} / 失败 {failed}）"
                      f"  {el / 60:.1f} 分钟  {rate:.2f} MB/s   ",
                      end="", flush=True)
    print()
    return {"total": len(todo), "ok": done, "skipped": skipped, "failed": failed,
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------
def _link_or_copy(src: str, dst: str) -> None:
    """优先硬链接（不占额外空间），失败则复制。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _black_png(path: str, size: tuple) -> None:
    """写一张全黑掩码（真实图像用：表示"无篡改区域"）。"""
    import numpy as np
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8)).save(path)


def organize(raw_dir: str, ds_dir: str, ratios=(0.8, 0.1, 0.1), seed: int = 3407,
             with_black_mask: bool = True) -> dict:
    """把 Au/ + Tp/ + Gt/ 整理成 {train,val,test}/{image,mask}/。"""
    from PIL import Image

    au_dir = os.path.join(raw_dir, "Au")
    tp_dir = os.path.join(raw_dir, "Tp")
    gt_dir = os.path.join(raw_dir, "Gt")
    for d, lab in ((au_dir, "Au"), (tp_dir, "Tp"), (gt_dir, "Gt")):
        if not os.path.isdir(d):
            print(f"  ❌ 缺少目录 {d}（先运行下载）")
            return {}

    # 掩码索引：去掉 `_gt` 后缀作为键（Gt 里是 `<图名>_gt.png`）
    gt_index = {}
    for fn in os.listdir(gt_dir):
        stem, ext = os.path.splitext(fn)
        if stem.lower().endswith("_gt"):
            stem = stem[:-3]
        gt_index[stem.lower()] = os.path.join(gt_dir, fn)

    items = []          # (image_path, mask_path or None, label)
    for fn in sorted(os.listdir(tp_dir)):
        stem = os.path.splitext(fn)[0]
        mp = gt_index.get(stem.lower())
        items.append((os.path.join(tp_dir, fn), mp, 1))
    for fn in sorted(os.listdir(au_dir)):
        items.append((os.path.join(au_dir, fn), None, 0))

    n_tp = sum(1 for _, m, l in items if l == 1)
    n_au = sum(1 for _, m, l in items if l == 0)
    print(f"  原始：篡改 {n_tp}（有掩码 "
          f"{sum(1 for _, m, l in items if l == 1 and m)}） / 真实 {n_au}")

    # 按类别分别打乱后切分，保证三个 split 的真/假比例一致
    rng = random.Random(seed)
    splits = ("train", "val", "test")
    # ⚠ 这里曾经用 `id(it)` 当字典键，是个**必然崩**的 bug：
    #   `id()` 取的是对象的内存地址，不是内容。
    #   上面 `assign[id(it)] = s` 用 `it`（items 里真实存在的元素）时侥幸能用，
    #   但下面遍历写成 `for (ip, mp, lab) in items:` —— 那是**拆包**，
    #   `id((ip, mp, lab))` 取的是**临时构造元组**的地址，语句一结束就被回收，
    #   地址可能被复用 → `KeyError`（或更糟：命中一个错误的键而不报错）。
    #   元组 (str, str|None, int) 本身可哈希，直接拿它当键即可。
    assign = {}
    for label, group in (("fake", [i for i in items if i[2] == 1]),
                         ("real", [i for i in items if i[2] == 0])):
        g = list(group)
        rng.shuffle(g)
        n = len(g)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        for idx, it in enumerate(g):
            s = splits[0] if idx < n_tr else (
                splits[1] if idx < n_tr + n_va else splits[2])
            assign[it] = s
    if len(assign) != len(items):
        raise RuntimeError(
            f"划分表条目数 {len(assign)} != 样本数 {len(items)}，"
            f"说明 items 里存在重复的 (图, 掩码, 标签) 组合，请检查下载目录。")

    counts = {s: {"real": 0, "fake": 0, "black_mask": 0} for s in splits}
    for it in items:
        ip, mp, lab = it
        s = assign[it]
        stem, ext = os.path.splitext(os.path.basename(ip))
        out_img = os.path.join(ds_dir, s, "image", stem + ext)
        out_msk = os.path.join(ds_dir, s, "mask", stem + ".png")
        _link_or_copy(ip, out_img)
        if mp:
            _link_or_copy(mp, out_msk)
        elif with_black_mask:
            try:
                with Image.open(ip) as im:
                    _black_png(out_msk, im.size)
                counts[s]["black_mask"] += 1
            except Exception as e:                 # noqa: BLE001
                print(f"  [warn] 生成黑掩码失败 {stem}: {e}")
        counts[s]["fake" if lab == 1 else "real"] += 1

    return {"totals": {"real": n_au, "fake": n_tp}, "splits": counts}


# --------------------------------------------------------------------------
def check(ds_dir: str, raw_dir: str) -> int:
    print("=" * 68)
    print(" 篡改数据集现状")
    print("=" * 68)
    ok = True
    print("\n[原始下载]")
    for k, exp in EXPECT.items():
        d = os.path.join(raw_dir, k)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        mark = "✅" if n >= exp else "⚠️"
        if n < exp:
            ok = False
        print(f"  {mark} {k:4s} {n:6d} / {exp}")
    print("\n[整理后的划分]")
    for s in ("train", "val", "test"):
        img = os.path.join(ds_dir, s, "image")
        msk = os.path.join(ds_dir, s, "mask")
        ni = len(os.listdir(img)) if os.path.isdir(img) else 0
        nm = len(os.listdir(msk)) if os.path.isdir(msk) else 0
        print(f"  {s:6s} image={ni:6d}  mask={nm:6d}"
              + ("  ✅" if ni and ni == nm else "  ⚠️ 数量不一致/未生成"))
        if not ni or ni != nm:
            ok = False
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="获取并整理 CASIA v2.0（篡改定位）")
    ap.add_argument("--root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--organize-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-black-mask", action="store_true",
                    help="不为真实图像生成全黑掩码")
    args = ap.parse_args()

    raw_dir = os.path.join(args.root, "_downloads", "modelscope", "CASIAv2")
    ds_dir = os.path.join(args.root, "Datasets", "CASIAv2")

    if args.check:
        return check(ds_dir, raw_dir)

    print("=" * 68)
    print(" CASIA v2.0（ModelScope 勘误版）")
    print("=" * 68)
    print(f" 数据源   : {MS_API}/{MS_DATASET}")
    print(f" 原始目录 : {raw_dir}")
    print(f" 目标目录 : {ds_dir}")
    try:
        free = shutil.disk_usage(args.root).free / 1024 ** 3
        print(f" 磁盘可用 : {free:.1f} GB")
    except Exception:                              # noqa: BLE001
        pass

    if not args.organize_only:
        print("\n[1/2] 下载（多线程）")
        stat = download_all(raw_dir, args.workers)
        print(f"  完成：{stat['ok']}/{stat['total']}（其中跳过已存在 {stat['skipped']}）"
              f"，失败 {stat['failed']}，用时 {stat['seconds'] / 60:.1f} 分钟")
        if stat["failed"]:
            print("  ⚠️ 有失败项，重跑本脚本会续传（已存在的文件会跳过）")

    if args.download_only:
        return 0

    print("\n[2/2] 整理布局")
    res = organize(raw_dir, ds_dir,
                   with_black_mask=not args.no_black_mask)
    if res:
        for s, c in res["splits"].items():
            print(f"  {s:6s} 真实 {c['real']:5d} + 篡改 {c['fake']:5d}"
                  f" = {c['real'] + c['fake']:5d}"
                  f"（其中全黑掩码 {c['black_mask']}）")

    print("\n自检：")
    return check(ds_dir, raw_dir)


if __name__ == "__main__":
    sys.exit(main())
