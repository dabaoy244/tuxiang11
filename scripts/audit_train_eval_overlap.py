#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练集 / 评测集**内容级**重叠审计（数据泄漏检查）。

为什么需要它
------------
`docs/05_评测与实验设计.md` 的检查清单里有一条：
「测试集与训练集**无重叠**（按图像内容/来源检查，**不只是文件名**）」。
而 `configs/*.yaml` 的注释里直接写着「验证集：ForenSynths/test 1300 张
（**与训练集图像无交集**）」。

**这句话此前从未被验证过，而实测是错的。** 本脚本就是那个验证，
而且是按内容哈希验证的：

* 只比文件名 → 会漏掉「同一张图在两个目录里叫不同名字」；
* 只比文件名 → 也会误报「同名但其实是两张不同的图」（实测 car 有 22 个这种）。

实测结论（2026-09-19，ForenSynths）
----------------------------------
* **池级**：val（训练池）的真图有 **400 张 / 4000（10%）**与 test 池内的图像
  **逐字节相同**，且高度集中：`cat` 200/200、`horse` 200/200 是**整类命中**。
* **实际子集**：config 的 train 1600 ∩ eval 1300 只命中 **3 张**（0.23%）；
  train 1600 ∩ 跨生成器评测计划 2600 也只命中 **3 张**。
* 也就是说：**"无交集"是错的，但当前报告的指标受影响极小**；
  风险在于**扩大评测规模**——池级 400 张会按比例进来。

做法
----
1. 用**项目自己的数据集类**（`GenSynthsDataset`）+ **项目自己的生成器扫描与
   分层抽样**（`eval_cross_generator.scan_generators` / `build_plan`）重建
   实验真正用到的那几份样本清单 —— 不是"大概是什么样"，而是 `max_samples`
   与随机种子 3407 决定的那一份，可逐项复现。
2. 对清单里的每张图算 MD5（只看字节，不看内容语义）。
3. 求交集，并区分：
   * `real-real`：训练真图 = 评测真图（最严重：真实类分数被记忆抬高）
   * `fake-fake`：训练伪造 = 评测伪造（同生成器时等于同域泄漏）
   * `real-fake`：训练真图出现在评测伪造侧（标签冲突，通常是整理错误）

用法
----
    # 复现对照实验用的两份子集（train 1600 / eval 1300）+ 跨生成器评测计划
    python scripts/audit_train_eval_overlap.py

    # 池级（不看子采样，看池子本身有多脏）—— 首次约 5 分钟，之后走缓存
    python scripts/audit_train_eval_overlap.py --pool

    # 生成"去泄漏"排除清单：评测侧凡与训练池内容相同的一律列出来
    python scripts/audit_train_eval_overlap.py --pool --write-exclude \
        data/Datasets/ForenSynths/_meta/leak_exclude_from_val.txt

    # 只自检本脚本的逻辑（不碰数据）
    python scripts/audit_train_eval_overlap.py --self-test

退出码：0 = 无重叠；1 = 发现重叠（可直接接进 CI / smoke_test）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "outputs", "_cache")
KIND_NAME = {0: "real", 1: "fake"}


# ---------------------------------------------------------------- 基础工具
def md5_of(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:                                   # noqa: BLE001
        return None


def cls_of(path: str) -> str:
    """取类别名：<...>/<class>/{0_real,1_fake}/x.png"""
    parts = path.replace(os.sep, "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in ("0_real", "1_fake"):
            return parts[i - 1] if i >= 1 else "?"
    return parts[-2] if len(parts) >= 2 else "?"


def hash_set(items) -> dict:
    """items: [(path, label)] → {md5: [(path, label), ...]}"""
    out: dict = defaultdict(list)
    for p, y in items:
        h = md5_of(p)
        if h:
            out[h].append((p, y))
    return out


def rel(p: str) -> str:
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:                                  # 跨盘符
        return p


def overlap_detail(A: dict, B: dict) -> dict:
    """两个 {md5: [(path,label)]} 的交集明细。纯函数，便于自检。"""
    kinds: Counter = Counter()
    per_cls: dict = defaultdict(Counter)
    examples = []
    n_pairs = 0
    for h in set(A) & set(B):
        for pa, ya in A[h]:
            for pb, yb in B[h]:
                kind = f"{KIND_NAME[ya]}-{KIND_NAME[yb]}"
                kinds[kind] += 1
                per_cls[kind][cls_of(pa)] += 1
                n_pairs += 1
                if len(examples) < 5:
                    examples.append((kind, cls_of(pa), pa, pb))
    return {"hashes": len(set(A) & set(B)), "pairs": n_pairs,
            "kinds": kinds, "per_cls": per_cls, "examples": examples}


def report(tag_a: str, tag_b: str, A: dict, B: dict) -> int:
    total_a = sum(len(v) for v in A.values())
    total_b = sum(len(v) for v in B.values())
    d = overlap_detail(A, B)
    print(f"\n--- {tag_a}（{total_a} 张，去重后 {len(A)} 个哈希）")
    print(f"    ∩ {tag_b}（{total_b} 张，去重后 {len(B)} 个哈希）")
    if not d["pairs"]:
        print("    ✅ 无内容级重叠")
        return 0
    print(f"    ❌ 重叠哈希 {d['hashes']} 个 → 命中 {d['pairs']} 对")
    for k, c in sorted(d["kinds"].items(), key=lambda kv: -kv[1]):
        detail = ", ".join(f"{c2}×{n2}" for c2, n2 in d["per_cls"][k].most_common(6))
        print(f"       {k:10s} {c:5d} 对   主要类别：{detail}")
    for kind, cls, pa, pb in d["examples"][:3]:
        print(f"       例（{kind}，{cls}）：\n         A={rel(pa)}\n         B={rel(pb)}")
    return d["pairs"]


# ---------------------------------------------------------------- 清单构建
def build_config_subsets(root: str, name: str, train_max: int, eval_max: int):
    """用项目自己的数据集类复现 config 里的 train/eval 子集（含 3407 抽样）。"""
    from src.data.datasets import GenSynthsDataset

    tr = GenSynthsDataset(root=root, name=name, split="val",
                          max_samples=train_max or None, train=True)
    ev = GenSynthsDataset(root=root, name=name, split="test",
                          max_samples=eval_max or None, train=False)
    return ([(s["image"], s["label"]) for s in tr.samples],
            [(s["image"], s["label"]) for s in ev.samples])


def build_crossgen_subset(data_root: str, name: str, per_class: int, seed: int):
    """复现 eval_cross_generator.py 的分层评测计划（默认 dedup_real）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ecg", os.path.join(ROOT, "scripts", "eval_cross_generator.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    gens = m.scan_generators(os.path.join(data_root, name, "test"))
    plan, per_gen = m.build_plan(gens, per_class, True, seed)
    return [(p, y) for _gen, p, y in plan], per_gen


def walk_split(split_dir: str):
    """遍历一个 split 目录下所有图，返回 [(path, label)]。"""
    items = []
    for cur, _dirs, files in os.walk(split_dir):
        base = os.path.basename(cur)
        if base not in ("0_real", "1_fake"):
            continue
        y = 0 if base == "0_real" else 1
        for f in files:
            items.append((os.path.join(cur, f), y))
    return items


def hash_pool(split_dir: str, cache_key: str, use_cache: bool = True) -> dict:
    """对一个 split 目录整体做哈希，结果缓存到 outputs/_cache/。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"md5_{cache_key}.json")
    items = walk_split(split_dir)
    mtimes = {p: int(os.path.getmtime(p)) for p, _ in items}
    if use_cache and os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("mtimes") == mtimes:
            print(f"  [cache] {cache_key}: 命中缓存（{len(saved['hashes'])} 张）")
            out: dict = defaultdict(list)
            for h, lst in saved["hashes"].items():
                out[h] = [(p, y) for p, y in lst]
            return out
    t0 = time.time()
    out = hash_set(items)
    print(f"  [hash ] {cache_key}: {len(items)} 张，用时 {time.time() - t0:.1f}s")
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"mtimes": mtimes,
                   "hashes": {h: v for h, v in out.items()}}, f)
    return out


# ---------------------------------------------------------------- 自检
def self_test() -> int:
    """纯逻辑自检：正例必须被抓到，同名不同图必须**不**被抓到。"""
    print("[自检] 重叠判定逻辑")
    bad = 0

    def chk(name, got, want):
        nonlocal bad
        ok = got == want
        if not ok:
            bad += 1
        print(f"   {'✅' if ok else '❌'} {name}: {got}（期望 {want}）")

    with tempfile.TemporaryDirectory() as d:
        def mk(relp, content):
            p = os.path.join(d, relp)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(content)
            return p

        tr = [
            (mk("tr/air/0_real/a.png", b"AAA"), 0),
            (mk("tr/air/1_fake/b.png", b"BBB"), 1),
            (mk("tr/cat/0_real/n.png", b"REAL1"), 0),   # 与评测同名但内容不同
        ]
        ev = [
            (mk("ev/gen/air/0_real/a.png", b"AAA"), 0),      # real-real 命中
            (mk("ev/gen/air/1_fake/b.png", b"BBB"), 1),      # fake-fake 命中
            (mk("ev/gen/air/0_real/c.png", b"CCC"), 0),      # 无关
            (mk("ev/gen/cat/0_real/n.png", b"REAL2"), 0),    # 同名不同图 → 不算
        ]
        d_overlap = overlap_detail(hash_set(tr), hash_set(ev))
        chk("命中对数（应为 2）", d_overlap["pairs"], 2)
        chk("real-real 计数", d_overlap["kinds"].get("real-real", 0), 1)
        chk("fake-fake 计数", d_overlap["kinds"].get("fake-fake", 0), 1)
        chk("同名不同图不计入", d_overlap["kinds"].get("real-real", 0), 1)
        chk("类别解析", cls_of(ev[0][0]), "air")

        tr2 = [(mk("tr2/air/0_real/z.png", b"ZZZ"), 0)]
        chk("完全不相交 → 0", overlap_detail(hash_set(tr2), hash_set(ev))["pairs"], 0)

        # 同一张图重复出现在多个生成器下：只按内容计一次对（避免重复计数）
        ev_dup = ev + [(mk("ev/gen2/air/0_real/a.png", b"AAA"), 0)]
        chk("同内容多副本 → 计数随副本增加",
            overlap_detail(hash_set(tr), hash_set(ev_dup))["pairs"], 3)

    print(f"[自检] {'全部通过' if not bad else f'{bad} 项失败'}")
    return 1 if bad else 0


# ---------------------------------------------------------------- 主流程
def write_exclude(train_pool: dict, eval_items, out_path: str) -> int:
    """把评测侧与训练池内容相同的图写成排除清单（每行一个路径）。"""
    train_md5 = set(train_pool)
    hits = []
    for p, _y in eval_items:
        h = md5_of(p)
        if h and h in train_md5:
            hits.append(p)
    hits.sort()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 去泄漏排除清单：评测侧这些图与训练池**内容相同**（MD5 相同）\n")
        f.write(f"# 来源：scripts/audit_train_eval_overlap.py --write-exclude\n")
        f.write(f"# 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  条目数：{len(hits)}\n")
        for p in hits:
            f.write(os.path.relpath(p, ROOT).replace(os.sep, "/") + "\n")
    print(f"\n[exclude] 已写出 {len(hits)} 条到 {rel(out_path)}")
    print("          用法：eval_cross_generator.py --exclude-list <该文件>")
    return len(hits)


def main() -> int:
    ap = argparse.ArgumentParser(description="训练/评测集内容级重叠（泄漏）审计")
    ap.add_argument("--root", default=os.path.join(ROOT, "data/Datasets"))
    ap.add_argument("--name", default="ForenSynths")
    ap.add_argument("--train-split", default="val")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--train-max", type=int, default=1600,
                    help="config 里训练子集大小（0=全集）")
    ap.add_argument("--eval-max", type=int, default=1300,
                    help="config 里评测子集大小（0=全集）")
    ap.add_argument("--per-class", type=int, default=100,
                    help="跨生成器分层评测的每生成器每侧张数（0=跳过）")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--pool", action="store_true",
                    help="额外做池级比对（整个 split，不看子采样；首次较慢）")
    ap.add_argument("--no-cache", action="store_true", help="池级哈希不使用缓存")
    ap.add_argument("--write-exclude", default="",
                    help="把评测侧与训练池内容相同的图写成排除清单")
    ap.add_argument("--self-test", action="store_true", help="只跑本脚本的逻辑自检")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    print("=" * 74)
    print("训练集 / 评测集 内容级重叠审计（按 MD5，不看文件名）")
    print("=" * 74)

    split_dir = lambda s: os.path.join(args.root, args.name, s)     # noqa: E731

    tr_items, ev_items = build_config_subsets(
        args.root, args.name, args.train_max, args.eval_max)
    print(f"\n[1] 复现 config 子集：train={len(tr_items)} 张"
          f"（{args.train_split}），eval={len(ev_items)} 张（{args.eval_split}）")
    A = hash_set(tr_items)
    B = hash_set(ev_items)
    bad = report("train 子集", "eval 子集", A, B)

    if args.per_class:
        pl_items, per_gen = build_crossgen_subset(
            args.root, args.name, args.per_class, args.seed)
        print(f"\n[2] 复现跨生成器分层评测：{len(pl_items)} 张"
              f"（per_class={args.per_class}，{len(per_gen)} 个生成器）")
        bad += report("train 子集", "跨生成器评测计划", A, hash_set(pl_items))

    if args.pool or args.write_exclude:
        print(f"\n[3] 池级比对（整个 split，不受子采样影响）")
        V = hash_pool(split_dir(args.train_split),
                      f"{args.name}_{args.train_split}", not args.no_cache)
        T = hash_pool(split_dir(args.eval_split),
                      f"{args.name}_{args.eval_split}", not args.no_cache)
        bad += report(f"{args.train_split} 池", f"{args.eval_split} 池", V, T)
        if args.write_exclude:
            write_exclude(V, ev_items=[(p, y) for p, y in walk_split(
                split_dir(args.eval_split))], out_path=args.write_exclude)

    print("\n" + "=" * 74)
    if bad:
        print(f"❌ 结论：存在 {bad} 对内容级重叠。")
        print("   ⚠ configs 里「与训练集图像无交集」的说法**不成立**。")
        print("   处理：① 论文口径如实写明实测重叠数；② 用 --write-exclude 生成排除清单，")
        print("          再用 eval_cross_generator.py --exclude-list 重算干净指标。")
        return 1
    print("✅ 结论：本次实际用到的样本无内容级重叠。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
