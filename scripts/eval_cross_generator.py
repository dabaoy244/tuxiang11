#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨生成器泛化评测 —— 产出论文的核心表格。

为什么必须单独写这个脚本
------------------------
`src/data/datasets.py::_maybe_subsample` 只按**标签**做均衡（各取一半真假），
**不按生成器分层**。ForenSynths 测试集里 stylegan2 有 15976 张、crn/imle 各 12764 张，
而 san 只有 419 张、seeingdark 只有 360 张、cyclegan 2642 张。
用 `max_samples: 1600` 去抽，抽到的几乎全是大集合的图，小生成器基本不出现 ——
**结果就是没法报"逐生成器精度"，而跨生成器泛化恰恰是这个项目最核心的结论。**
（实测过：早先按 max_samples 抽样，8 个样本里出现同一张图。）

本脚本改为**按生成器分层采样**：每个生成器各取 N 张真 + N 张假，
逐生成器报 ACC / AUC / AP，再给出宏平均。

一个容易踩的坑：真图是复用的
----------------------------
ForenSynths 各生成器目录下的 `0_real` 来自同一批真实图（LSUN/CelebA-HQ 等），
文件名相同。若各生成器都各取 N 张真图，聚合统计里**同一张真图会被重复计入 13 次**，
把"总准确率"抬得虚高。所以默认开启 `--dedup-real`：按文件名去重，
同一张真图只算一次（在它第一次出现的生成器里）。

用法
----
    python scripts/eval_cross_generator.py --config configs/lite_realval.yaml \
        --ckpt checkpoints/realval_best.pt --per-class 150 --out outputs/cross_gen

    # 只看有哪些生成器、各有多少张，不跑推理
    python scripts/eval_cross_generator.py --list-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.datasets import IMG_EXT, _read_image  # noqa: E402
from src.data.transforms import build_transform          # noqa: E402
from src.engine.metrics import average_precision as cls_average_precision  # noqa: E402
from src.engine.metrics import roc_auc as cls_roc_auc                      # noqa: E402


# --------------------------------------------------------------------------
# NumPy 2.0 兼容：`np.trapz` 在 NumPy 2.0 中已被**移除**，改名 `np.trapezoid`。
# 本机 numpy 2.5.3 上直接 `np.trapz(...)` 会抛
# `AttributeError: module 'numpy' has no attribute 'trapz'`。
# ⚠️ 这个错**发生在收集完全部预测、开始算指标时** —— 前一版两条臂各跑了
#    约 9 分钟推理，然后在算 AUC 的那一行崩掉，整份跨生成器结果全丢。
#    所以这里做版本兼容，而不是简单改名（云端 numpy 版本可能更老）。
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# --------------------------------------------------------------------------
def scan_generators(test_root: str) -> dict:
    """返回 {生成器名: {"real": [路径...], "fake": [路径...]}}。

    ⚠ 层级不统一，必须递归找 `0_real` / `1_fake`：
        biggan/0_real                ← 一层
        cyclegan/apple/0_real        ← 两层（按类别再分）
        stylegan2/car/0_real         ← 两层
    只认"直接子目录"的话，cyclegan / progan / stylegan / stylegan2 这四个会被
    整个漏掉 —— 而它们恰好是体量最大的几类（stylegan2 有 15976 张）。
    """
    out = {}
    if not os.path.isdir(test_root):
        return out
    for gen in sorted(os.listdir(test_root)):
        gdir = os.path.join(test_root, gen)
        if not os.path.isdir(gdir):
            continue
        rec = {"real": [], "fake": []}
        for cur, dirs, _files in os.walk(gdir):
            base = os.path.basename(cur)
            if base not in ("0_real", "1_fake"):
                continue
            key = "real" if base == "0_real" else "fake"
            files = []
            for e in IMG_EXT:
                files.extend(glob(os.path.join(cur, e)))
                files.extend(glob(os.path.join(cur, e.upper())))
            rec[key].extend(files)
            dirs[:] = []                     # 不再往下钻
        rec["real"] = sorted(set(rec["real"]))
        rec["fake"] = sorted(set(rec["fake"]))
        if rec["real"] or rec["fake"]:
            out[gen] = rec
    return out


def norm_key(p: str) -> str:
    """路径归一化键：绝对路径 + Windows 大小写不敏感。"""
    return os.path.normcase(os.path.abspath(p))


def load_exclude_list(path: str) -> set:
    """读去泄漏排除清单（每行一个路径，`#` 开头为注释）。

    清单由 scripts/audit_train_eval_overlap.py --write-exclude 生成，
    里面是**与训练集内容逐字节相同**的评测图。
    """
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(norm_key(os.path.join(ROOT, line) if not os.path.isabs(line)
                             else line))
    return out


def build_plan(gens: dict, per_class: int, dedup_real: bool, seed: int,
               exclude: set | None = None):
    """按生成器分层挑选评测样本。

    返回 [(gen, path, label)]，label 1=伪造 0=真实。
    dedup_real=True 时，同一张真图只在其首次出现的生成器里计入。
    exclude 非空时，**在挑选之前**就把这些路径从候选里去掉（用于去泄漏：
    排除与训练集内容相同的图），从而不降低 per_class 的取数。
    """
    rng = random.Random(seed)
    seen_real = set()
    plan, per_gen, n_ex = [], {}, 0
    for gen in sorted(gens):
        rec = gens[gen]
        fr = sorted(rec["fake"])
        rl = sorted(rec["real"])
        if exclude:
            f0, r0 = len(fr), len(rl)
            fr = [p for p in fr if norm_key(p) not in exclude]
            rl = [p for p in rl if norm_key(p) not in exclude]
            n_ex += (f0 - len(fr)) + (r0 - len(rl))
        rng.shuffle(fr)
        rng.shuffle(rl)
        picks_f = fr[:per_class]
        if dedup_real:
            picks_r = [p for p in rl if os.path.basename(p) not in seen_real][:per_class]
            seen_real.update(os.path.basename(p) for p in picks_r)
        else:
            picks_r = rl[:per_class]
        per_gen[gen] = {"n_real": len(picks_r), "n_fake": len(picks_f),
                        "avail_real": len(rl), "avail_fake": len(fr),
                        "excluded": (len(rec["real"]) - len(rl))
                        + (len(rec["fake"]) - len(fr))}
        plan += [(gen, p, 0) for p in picks_r]
        plan += [(gen, p, 1) for p in picks_f]
    if exclude:
        per_gen["_excluded_total"] = n_ex
    return plan, per_gen


# --------------------------------------------------------------------------
def load_model(config: str, ckpt: str, device: str):
    import torch

    from src.models.spatial_branch import align_cfg_to_checkpoint
    from src.models.vibnet import build_model, load_config

    cfg = load_config(config)
    state = None
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        align_cfg_to_checkpoint(cfg, state.get("model", state))
        print(f"[eval] 载入权重 {ckpt}（epoch={state.get('epoch')}, "
              f"stage={state.get('stage')}）")
    else:
        print(f"[eval] ⚠️ 找不到权重 {ckpt} —— 将使用随机初始化模型，"
              f"指标无意义，仅用于验证脚本通路。")
    model = build_model(cfg)
    if state is not None:
        miss, unexp = model.load_state_dict(state.get("model", state), strict=False)
        if miss or unexp:
            print(f"[eval] 权重键对齐：missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(dev)
    return model, cfg, torch, dev


def run_inference(plan, model, cfg, torch, dev, size, batch_size, quiet=False):
    """批量推理，返回 [(idx, prob_fake)]。"""
    tf = build_transform(size, train=False)
    probs = np.zeros(len(plan), dtype=np.float32)

    def collate(batch):
        xs = []
        for p in batch:
            img = _read_image(p)
            x, _ = tf(img, None)
            xs.append(np.asarray(x))
        return torch.from_numpy(np.stack(xs)).float()

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for i in range(0, len(plan), batch_size):
            chunk = plan[i:i + batch_size]
            paths = [c[1] for c in chunk]
            xb = collate(paths).to(dev)
            out = model(xb, sample_vib=False, beta=0.0)
            p = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
            probs[i:i + len(chunk)] = p
            done += len(chunk)
            if not quiet and (i // batch_size) % 5 == 0:
                el = time.time() - t0
                eta = el / max(done, 1) * (len(plan) - done)
                print(f"\r    推理 {done}/{len(plan)}  "
                      f"{el:.0f}s 已用 / 预计还需 {eta:.0f}s   ", end="", flush=True)
    if not quiet:
        print()
    return probs, time.time() - t0


# --------------------------------------------------------------------------
def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    """用纯 numpy 算指标，不依赖 sklearn。"""
    out = {"n": int(len(y))}
    if len(y) == 0:
        return out
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out.update(tp=tp, tn=tn, fp=fp, fn=fn,
               acc=(tp + tn) / len(y),
               real_acc=(tn / (tn + fp)) if (tn + fp) else float("nan"),
               fake_acc=(tp / (tp + fn)) if (tp + fn) else float("nan"),
               prec=(tp / (tp + fp)) if (tp + fp) else 0.0,
               rec=(tp / (tp + fn)) if (tp + fn) else 0.0)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 and n0:
        # ★ AUC / AP 一律走 `src/engine/metrics.py` 里的**唯一实现**。
        #   本文件曾有自己的一份，与主评测路径的定义不同（11 点插值 vs ΣP·ΔR），
        #   导致两张表的 `ap` 不可比；并列分数上两者也都偏低。
        #   现在口径唯一，由 `scripts/test_metric_definitions.py` 钉死。
        out["auc"] = float(cls_roc_auc(y, s))
        out["ap"] = float(cls_average_precision(y, s))
        # 保留"不做并列校正的梯形法"仅供对照，显式暴露偏差（通常偏低）
        order = np.argsort(-s)
        ys = y[order]
        cum_tp = np.cumsum(ys == 1)
        cum_fp = np.cumsum(ys == 0)
        fpr = np.concatenate([[0.0], cum_fp / n0])
        tpr = np.concatenate([[0.0], cum_tp / n1])
        out["auc_trapz"] = float(_trapz(tpr, fpr))
    else:
        out["auc"] = out["auc_trapz"] = out["ap"] = float("nan")
    return out


# --------------------------------------------------------------------------
def summarize(names: np.ndarray, y: np.ndarray, probs: np.ndarray,
              per_gen_meta: dict, meta: dict):
    """按生成器聚合指标并渲染 Markdown。

    抽成独立函数是为了让 `scripts/recompute_cross_gen.py` 复用**同一段聚合逻辑** ——
    否则"重算"会和"实跑"各走一套代码，又制造出一类新的口径分叉。
    """
    per_gen = {}
    for g in sorted(per_gen_meta):
        sel = names == g
        if not sel.any():
            continue
        m = metrics(y[sel], probs[sel])
        m.update({k: per_gen_meta[g][k] for k in
                  ("n_real", "n_fake", "avail_real", "avail_fake")})
        per_gen[g] = m
    overall = metrics(y, probs)
    overall.update(n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()))
    for k in ("acc", "real_acc", "fake_acc", "auc", "ap"):
        vals = [v[k] for v in per_gen.values() if k in v and not np.isnan(v[k])]
        overall[f"{k}_macro"] = float(np.mean(vals)) if vals else float("nan")
    return per_gen, overall, render(per_gen, overall, meta)


def render(per_gen: dict, overall: dict, meta: dict) -> str:
    L = []
    L.append(f"# 跨生成器泛化评测（{meta['ckpt_name']}）\n")
    L.append(f"- 配置：`{meta['config']}`　权重：`{meta['ckpt']}`")
    L.append(f"- 采样：每生成器每类 ≤{meta['per_class']} 张；"
             f"真图去重：{'是' if meta['dedup_real'] else '否'}；随机种子 {meta['seed']}")
    # 生成器筛选是本表的口径来源：不写出来的话，同一份 config 在不同筛选下
    # 会产出「同名不同义」的两张表（例如 val_cross 只覆盖 3 个留出生成器）。
    if meta.get("include_generators") or meta.get("exclude_generators"):
        _inc = ",".join(meta.get("include_generators") or []) or "全部"
        _exc = ",".join(meta.get("exclude_generators") or []) or "无"
        L.append(f"- **生成器筛选**：include=`{_inc}`　exclude=`{_exc}`"
                 f"（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）")
    L.append(f"- 推理：{meta['n_total']} 张，用时 {meta['seconds']:.0f} 秒，"
             f"batch={meta['batch_size']}\n")
    L.append("| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |")
    L.append("|---|---|---|---|---|---|---|")
    for gen in sorted(per_gen):
        m = per_gen[gen]
        L.append(f"| {gen} | {m.get('n_real', 0)}/{m.get('n_fake', 0)} "
                 f"| {m.get('acc', float('nan')):.4f} "
                 f"| {m.get('real_acc', float('nan')):.4f} "
                 f"| {m.get('fake_acc', float('nan')):.4f} "
                 f"| {m.get('auc', float('nan')):.4f} "
                 f"| {m.get('ap', float('nan')):.4f} |")
    L.append(f"| **宏平均** | — | **{overall.get('acc_macro', float('nan')):.4f}** "
             f"| **{overall.get('real_acc_macro', float('nan')):.4f}** "
             f"| **{overall.get('fake_acc_macro', float('nan')):.4f}** "
             f"| **{overall.get('auc_macro', float('nan')):.4f}** "
             f"| **{overall.get('ap_macro', float('nan')):.4f}** |")
    L.append(f"| **微平均（合并全部样本）** | {overall['n_real']}/{overall['n_fake']} "
             f"| **{overall['acc']:.4f}** | {overall['real_acc']:.4f} "
             f"| {overall['fake_acc']:.4f} | {overall['auc']:.4f} "
             f"| {overall['ap']:.4f} |")
    L.append("")
    L.append("> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；"
             "微平均 = 把所有样本合在一起算。")
    L.append("> 两者差异大说明模型在不同生成器上表现严重不均。")
    if meta.get("cmd"):
        L.append("")
        L.append(f"> 复现命令：`{meta['cmd']}`")
    return "\n".join(L)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="跨生成器泛化评测")
    ap.add_argument("--config", default="configs/lite_realval.yaml")
    ap.add_argument("--ckpt", default="checkpoints/realval_best.pt")
    ap.add_argument("--data-root", default="data/Datasets/ForenSynths/test")
    ap.add_argument("--per-class", type=int, default=150,
                    help="每个生成器的真/假各取多少张")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-dedup-real", action="store_true",
                    help="不去重真图（默认去重；见文件头说明）")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--exclude-list", default=None,
                    help="去泄漏排除清单（每行一个路径）。由 "
                         "scripts/audit_train_eval_overlap.py --write-exclude 生成；"
                         "清单里的图在**挑选之前**就从候选里剔除，"
                         "因此不会降低 --per-class 的取数")
    ap.add_argument("--generators", default=None,
                    help="只评测这些生成器（逗号分隔）。默认全部。"
                         "用于单独评留出的 val_cross 生成器")
    ap.add_argument("--exclude-generators", default=None,
                    help="剔除这些生成器（逗号分隔）。用于报**严格跨生成器**宏平均时"
                         "把「与训练同源的 progan」和「已留作验证集的生成器」踢出去")
    ap.add_argument("--out", default="outputs/cross_gen")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scores-out", default=None,
                    help="把逐样本 (生成器, 标签, 分数) 存成 .npz。"
                         "★ 改指标定义后可用它免推理重算（推理是本脚本最贵的一步）")
    args = ap.parse_args()
    args.generators = [g.strip() for g in (args.generators or "").split(",") if g.strip()]
    args.exclude_generators = [g.strip() for g in
                               (args.exclude_generators or "").split(",") if g.strip()]

    exclude = load_exclude_list(args.exclude_list) if args.exclude_list else None

    root_abs = args.data_root if os.path.isabs(args.data_root) \
        else os.path.join(ROOT, args.data_root)
    gens = scan_generators(root_abs)
    if not gens:
        print(f"❌ 没在 {root_abs} 下找到任何生成器目录。")
        return 1

    # ---- 生成器级筛选（可选）-------------------------------------------------
    # 两个用途：
    #   ① 只评「被留出的 val_cross 生成器」——用来验证新验证集是否真的不再饱和；
    #   ② 报严格跨生成器宏平均时，把「与训练同源的 progan」和
    #      「已留作验证集的生成器」从测试集里剔除。
    # ★ 这里刻意做成**硬失败**而不是静默过滤：生成器名写错（拼写/大小写/名字不存在）
    #   会让宏平均在一个意外的小集合上算出来，数字还更好看 —— 那是彻头彻尾的
    #   静默失效。所以名字对不上就直接报错并把盘上实际有的列出来。
    if args.generators or args.exclude_generators:
        inc = {g.lower() for g in args.generators}
        exc = {g.lower() for g in args.exclude_generators}
        have = {g.lower() for g in gens}
        unknown = sorted((inc | exc) - have)
        if unknown:
            print(f"❌ 筛选条件里有不存在的生成器：{unknown}")
            print(f"   盘上实际有：{sorted(gens)}")
            return 1
        kept, dropped = {}, []
        for g, rec in gens.items():
            gl = g.lower()
            if (inc and gl not in inc) or (gl in exc):
                dropped.append(g)
                continue
            kept[g] = rec
        if not kept:
            print("❌ 筛选后没有剩下任何生成器，请检查 --generators / "
                  "--exclude-generators。")
            return 1
        print(f"\n[filter] 生成器筛选：保留 {len(kept)} 个，剔除 {len(dropped)} 个"
              f"（{'、'.join(sorted(dropped))}）")
        gens = kept

    print("=" * 74)
    print(" 跨生成器泛化评测")
    print("=" * 74)
    print(f"\n找到 {len(gens)} 个生成器：")
    tot_r = tot_f = 0
    for g in sorted(gens):
        r, f = len(gens[g]["real"]), len(gens[g]["fake"])
        tot_r += r
        tot_f += f
        print(f"  {g:18s} 真 {r:6d}  假 {f:6d}")
    print(f"  {'合计':18s} 真 {tot_r:6d}  假 {tot_f:6d}")
    if args.list_only:
        return 0

    plan, per_gen_meta = build_plan(gens, args.per_class,
                                    not args.no_dedup_real, args.seed,
                                    exclude=exclude)
    if exclude:
        n_ex = per_gen_meta.pop("_excluded_total", 0)
        print(f"\n去泄漏：排除清单 {args.exclude_list}"
              f"（{len(exclude)} 条），从候选里剔除 {n_ex} 张与训练集内容相同的图")
    y = np.array([c[2] for c in plan], dtype=int)
    print(f"\n评测计划：{len(plan)} 张（真 {int((y == 0).sum())} / "
          f"假 {int((y == 1).sum())}）")

    model, cfg, torch, dev = load_model(args.config, args.ckpt, args.device)
    probs, secs = run_inference(plan, model, cfg, torch, dev,
                               args.size, args.batch_size)

    names = np.array([c[0] for c in plan])
    # ★ 先把原始分数落盘，再算指标。指标实现将来若有修正，可直接重算而无需重跑推理；
    #   本轮就吃过一次亏：两条臂各推理约 9 分钟后在算 AUC 时崩溃，结果全丢。
    if args.scores_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.scores_out)) or ".", exist_ok=True)
        # ★ 连**文件基名**一起存：只存 `names`（生成器名）的话，事后无法证明
        #   「两个变体评测的是同一批样本」，before/after 的差值就归因不了（docs/08 D28）。
        #   ⚠ 注意 `names` 是**生成器**名，不是文件名 —— 别拿它当文件清单用。
        paths = np.array([os.path.basename(c[1]) for c in plan])
        np.savez_compressed(args.scores_out, names=names, y=y, probs=probs,
                            paths=paths,
                            ckpt=os.path.basename(args.ckpt),
                            per_class=args.per_class, size=args.size,
                            per_gen_meta=json.dumps(per_gen_meta, ensure_ascii=False))
        print(f"逐样本分数已存盘：{args.scores_out}"
              f"（含文件基名，可用 scripts/report_third_domain.py 校验样本一致性；"
              f"改指标定义后可用 scripts/recompute_cross_gen.py 免推理重算）")
    # ★ 这个 dict 必须绑定成变量：下面写 JSON 时要用到它。
    #   曾经写成字面量直接传给 summarize()，结果 main() 里没有 meta，
    #   写 JSON 时抛 NameError —— 而 .md 已经先落盘了，于是留下一个「md 正常、
    #   json 是 0 字节」的假成功（下游 run_ablation.py 读 json 会读到空）。
    meta = {
        "config": args.config, "ckpt": args.ckpt,
        "ckpt_name": os.path.basename(args.ckpt),
        "per_class": args.per_class, "seed": args.seed,
        "dedup_real": not args.no_dedup_real,
        "n_total": len(plan), "seconds": secs,
        "batch_size": args.batch_size,
        # 口径溯源：生成器筛选与原始命令行。缺了这两项，报告头只写 config
        # 路径，读的人无法判断这张表覆盖的是哪几个生成器（曾经踩过）。
        "include_generators": list(args.generators or []),
        "exclude_generators": list(args.exclude_generators or []),
        "cmd": "python " + " ".join([os.path.relpath(sys.argv[0], ROOT)]
                                    + sys.argv[1:]),
    }
    per_gen, overall, md = summarize(names, y, probs, per_gen_meta, meta)
    print("\n" + md)

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    md_path = os.path.join(args.out, f"cross_generator{suffix}.md")
    js_path = os.path.join(args.out, f"cross_generator{suffix}.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    # 先写临时文件再原子替换：避免中途出错留下 0 字节的 json（下游会当成有效结果读）
    tmp_js = js_path + ".tmp"
    with open(tmp_js, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "per_generator": per_gen, "overall": overall},
                  f, ensure_ascii=False, indent=2, default=float)
    os.replace(tmp_js, js_path)
    # 自查：产物非空，否则明确报错而不是静默返回 0
    for p in (md_path, js_path):
        if os.path.getsize(p) == 0:
            print(f"❌ 产物为空：{p}")
            return 1
    print(f"\n已写出：\n  {md_path}\n  {js_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
