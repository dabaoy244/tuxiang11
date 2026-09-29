"""诊断：分层 VIB 的 KL 实际落在哪、梯度范数是否真的触发裁剪。

背景（2026-09-29）
------------------
`docs/13` 把「β 退火 / KL 散度裁剪 / 梯度裁剪」写成三重稳定训练策略，
但正式训练的日志里 **只有 β 被记录**（`beta=0.000 → 0.100` 共 100 行），
`kl` 与 `grad_norm` 一次都没打印（`train_log.txt` 全文 grep 结果为 0）。

于是一个尴尬的局面：三个机制里只有一个有可观测证据。
本脚本用产出权重在 CPU 上补测另两个量，回答三个问题：

  1. **KL 常态值是多少？** `[0,10]` 的裁剪（公式 25）是"安全阀"还是"常态生效"？
     若是常态生效，`torch.clamp` 会把 VIB 正则项的梯度清零 ⇒ **信息瓶颈被静默关闭**，
     而这在论文里完全看不出来（这才是本脚本存在的理由）。
  2. **梯度总范数有没有超过 `grad_clip_norm=5.0`？** 即 `clip_grad_norm_` 是否真的
     在起作用，还是从来没触发过（后者说明这个"策略"是装饰）。
  3. **β=0（stage1 设定）与 β=0.1（stage3 设定）下，上面两项有无差别？**

做法上刻意与训练一致：
  * 直接用权重里存的 `ckpt["cfg"]` 建模（不是去读 configs/，避免口径漂移）；
  * 恢复 `normalizer` 与 `loss_weights`（z-score 统计量与不确定性加权都存了），
    否则损失量级与训练时不同，梯度范数就没有可比性；
  * `load_state_dict(strict=False)` 是**显式打印 missing/unexpected 数量**的
    （本项目的高频坑：它会静默吞掉没对上的键，ACC 照样有输出）。

产物：`outputs/diag/vib_kl_gradnorm{_tag}.json` / `.md`
（默认写独立目录，不污染正式产物；`--tag` 可再加后缀）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.transforms import build_transform          # noqa: E402
from src.losses.multi_task import build_loss             # noqa: E402
from src.models.vib import beta_schedule                 # noqa: E402
from src.models.vibnet import build_model                # noqa: E402

CKPT_DEFAULT = "outputs/upload/results_20260928/_pack/vibnet_best.pt"


# --------------------------------------------------------------------------
def collect_images(data_root: str, gens, per_gen: int, seed: int = 3407):
    """按生成器分层取 (path, label)：0_real 记 0，1_fake 记 1。

    ⚠ ForenSynths/test 有**两种**布局，必须都支持（2026-09-29 实测）：
        test/biggan/{0_real,1_fake}/...            ← 扁平
        test/cyclegan/apple/{0_real,1_fake}/...    ← 多一层类别（progan/cyclegan/
                                                      stylegan/stylegan2 都是这种）
    只按 `<gen>/0_real` 找会把这 4 个生成器静默当成"空的"，样本量少一半还不报错。
    因此这里对生成器目录做递归 glob，取到所有 `0_real` / `1_fake` 目录。
    """
    rng = np.random.RandomState(seed)
    out = []
    for g in gens:
        gdir = os.path.join(data_root, g)
        if not os.path.isdir(gdir):
            print(f"  [warn] 生成器目录不存在，跳过：{gdir}")
            continue
        for sub, lab in (("0_real", 0), ("1_fake", 1)):
            ds = sorted(glob(os.path.join(gdir, "**", sub), recursive=True))
            if not ds:
                print(f"  [warn] {g}: 找不到任何 {sub} 目录")
                continue
            fs = []
            for d in ds:
                fs += [os.path.join(d, f) for f in os.listdir(d)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
            if len(fs) > per_gen:
                fs = [fs[i] for i in rng.choice(len(fs), per_gen, replace=False)]
            out += [(f, lab) for f in fs]
    return out


def load_batch(items, transform):
    import cv2
    imgs, labs = [], []
    for p, lab in items:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(transform(img)[0])
        labs.append(lab)
    x = torch.from_numpy(np.stack(imgs)).float()
    return x, torch.tensor(labs, dtype=torch.long)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="VIB 的 KL 与梯度范数诊断")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--data-root", default="data/Datasets/ForenSynths/test")
    ap.add_argument("--generators", default="biggan,cyclegan,stylegan2,crn",
                    help="分层取图的生成器（同时含真图与假图）")
    ap.add_argument("--per-gen", type=int, default=16, help="每个生成器每类取几张")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4, help="做几次 backward 测梯度范数")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/diag")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    dev = torch.device(args.device)

    print("=" * 74)
    print(" VIB KL / 梯度范数诊断")
    print("=" * 74)
    print(f" 权重      : {args.ckpt}")
    print(f" 设备      : {dev}")
    print(f" 生成器    : {gens}（每生成器每类 {args.per_gen} 张）")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    print(f" 权重内记录: epoch={ckpt.get('epoch')} stage={ckpt.get('stage')} "
          f"β={ckpt.get('record', {}).get('beta')}")

    # ---- 建模 + 严格核对键 -------------------------------------------
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print()
    print(f" [键核对] missing={len(missing)} unexpected={len(unexpected)}"
          f"  （应为 0/0；非 0 说明有模块没装上还被静默放过）")
    for k in list(missing)[:8]:
        print(f"    missing   : {k}")
    for k in list(unexpected)[:8]:
        print(f"    unexpected: {k}")
    if missing or unexpected:
        print("  ⚠ 键未完全对上，下面的数值可能不代表训练时的模型。")
    model.to(dev).eval()

    # ---- 恢复损失状态（与 trainer.load 一致）--------------------------
    criterion = build_loss(cfg)
    if "loss_weights" in ckpt:
        criterion.load_state_dict(ckpt["loss_weights"])
    if "normalizer" in ckpt:
        criterion.normalizer.load_state_dict(ckpt["normalizer"])
    lcfg = cfg["loss"].get("vib", {})
    vcfg = cfg["model"].get("vib", {})
    kl_max = float(vcfg.get("kl_clip_max", 10.0))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 5.0))
    print(f"\n [设定] kl_clip=[{vcfg.get('kl_clip_min', 0.0)}, {kl_max}]"
          f" kl_reduction={vcfg.get('kl_reduction')}"
          f" kl_clip_grad_through={vcfg.get('kl_clip_grad_through')}")
    print(f" [设定] grad_clip_norm={grad_clip}"
          f"  beta_max={vcfg.get('beta_max')}"
          f"  warmup={vcfg.get('beta_warmup_start')}→{vcfg.get('beta_warmup_end')}")

    # ---- 数据 --------------------------------------------------------
    items = collect_images(args.data_root, gens, args.per_gen)
    print(f"\n [数据] 候选 {len(items)} 张（真 {sum(1 for _, l in items if l == 0)}"
          f" / 假 {sum(1 for _, l in items if l == 1)}）")
    if len(items) < args.batch_size:
        print(" ✗ 样本不足以组成一个 batch，退出。")
        return 2
    transform = build_transform(size=cfg["data"].get("image_size", 224), train=False)

    batches = []
    for i in range(0, len(items) - args.batch_size + 1, args.batch_size):
        batches.append(items[i:i + args.batch_size])
    batches = batches[:max(args.steps, 3)]

    # ---- 阶段一：前向，看 KL ------------------------------------------
    def probe_forward(beta: float, n_forward: int):
        rows = []
        with torch.no_grad():
            for bi in batches[:n_forward]:
                x, y = load_batch(bi, transform)
                out = model(x.to(dev), sample_vib=True, beta=beta)
                kl_raw = out["kl_raw"]
                rows.append({
                    "kl_raw": float(kl_raw),
                    "kl_clipped": float(out["kl"]),
                    "clamp_binds": bool(float(kl_raw) > kl_max),
                    "sigma_min": float(out["sigma"].min()),
                    "sigma_max": float(out["sigma"].max()),
                    "sigma_mean": float(out["sigma"].mean()),
                    "mu_abs_mean": float(out["mu"].abs().mean()),
                })
        return rows

    print("\n" + "-" * 74)
    print(" ① KL 分布：β 取 stage1(0.0) 与 stage3(0.1) 两种设定")
    print("-" * 74)
    fwd = {}
    for beta in (0.0, 0.1):
        rows = probe_forward(beta, len(batches))
        fwd[f"beta_{beta}"] = rows
        kr = np.array([r["kl_raw"] for r in rows])
        sm = np.array([r["sigma_mean"] for r in rows])
        print(f"  β={beta:<4} kl_raw  mean={kr.mean():.4f}  min={kr.min():.4f} "
              f" max={kr.max():.4f}   命中裁剪上限次数={sum(r['clamp_binds'] for r in rows)}"
              f"/{len(rows)}   σ̄={sm.mean():.4f}")
    all_kl = np.array([r["kl_raw"] for rows in fwd.values() for r in rows])
    n_bind = sum(r["clamp_binds"] for rows in fwd.values() for r in rows)
    print(f"  → 合计 {len(all_kl)} 个 batch：kl_raw 均值 {all_kl.mean():.4f}，"
          f"最大 {all_kl.max():.4f}，触发 [0,{kl_max}] 上限 {n_bind} 次")

    # ---- 阶段二：反向，看梯度范数 -------------------------------------
    print("\n" + "-" * 74)
    print(" ② 梯度范数：backward 后、clip 前的总范数（与 grad_clip 比）")
    print("-" * 74)
    grad = {}
    model.train()
    for beta in (0.0, 0.1):
        norms = []
        for bi in batches[:args.steps]:
            x, y = load_batch(bi, transform)
            model.zero_grad(set_to_none=True)
            out = model(x.to(dev), sample_vib=True, beta=beta)
            losses = criterion(out, y.to(dev), None, beta=beta,
                               active_tasks=["cls"], update_norm=True)
            loss = losses["total"]
            if losses.get("empty"):
                continue
            loss.backward()
            # 返回值即"裁剪前"的总范数
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            norms.append(float(total_norm))
        if norms:
            a = np.array(norms)
            grad[f"beta_{beta}"] = norms
            over = int((a > grad_clip).sum())
            print(f"  β={beta:<4} grad_norm  mean={a.mean():.3f}  min={a.min():.3f} "
                  f" max={a.max():.3f}   > {grad_clip} 的次数={over}/{len(a)}"
                  f"   裁剪缩放比={min(1.0, grad_clip / a.max()):.3f}")
        else:
            print(f"  β={beta:<4} 无有效 batch（全部 empty）")
    model.eval()

    # ---- 汇总 --------------------------------------------------------
    verdict = {
        "kl_clamp_is_active_safeguard":
            bool(n_bind == 0),
        "grad_clip_binds": bool(any(np.array(v) .max() > grad_clip
                                    for v in grad.values()) if grad else False),
    }
    print("\n" + "=" * 74)
    print(" 判读")
    print("=" * 74)
    print(f"  KL 裁剪：{'从未触发（安全阀，VIB 未被静默关闭）' if verdict['kl_clamp_is_active_safeguard'] else '★ 常态触发 —— clamp 会清零 VIB 正则梯度，机制被静默旁路'}")
    print(f"  梯度裁剪：{'确实触发（策略在起作用）' if verdict['grad_clip_binds'] else '★ 从未触发 —— 该设置在本模型上不起作用'}")

    os.makedirs(args.out, exist_ok=True)
    suf = f"_{args.tag}" if args.tag else ""
    js = os.path.join(args.out, f"vib_kl_gradnorm{suf}.json")
    payload = {
        "ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "stage": ckpt.get("stage"),
        "key_check": {"missing": len(missing), "unexpected": len(unexpected)},
        "settings": {"kl_clip_max": kl_max, "grad_clip_norm": grad_clip,
                     "beta_max": vcfg.get("beta_max"),
                     "beta_warmup": [vcfg.get("beta_warmup_start"), vcfg.get("beta_warmup_end")],
                     "beta_schedule_probe": {
                         "t=0": beta_schedule(0), "t=19": beta_schedule(19),
                         "t=20": beta_schedule(20), "t=39": beta_schedule(39),
                         "t=40": beta_schedule(40), "t=54": beta_schedule(54)}},
        "forward": fwd, "grad_norm": grad,
        "kl_summary": {"n_batches": int(len(all_kl)), "mean": float(all_kl.mean()),
                       "min": float(all_kl.min()), "max": float(all_kl.max()),
                       "n_clamp_bind": int(n_bind)},
        "verdict": verdict,
    }
    with open(js, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n 已写出：{js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
