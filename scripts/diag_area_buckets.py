#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-① 定位归因诊断：按 GT 篡改面积分桶的逐图 IoU。

为什么必须先做这一步
--------------------
09-28 那次实测 `miou_tampered_only = 0.2668`（目标 0.56），差得很远。
但"差得远"不是原因，只是结果。可能的根因至少有两条**对策完全相反**：
    (a) 小篡改区域被压掉  → 对策是提输入分辨率 / 加小目标监督；
    (b) 定位头整体容量不足 → 对策是加宽解码器 / 加边缘监督 / 换损失。
不区分这两者就去重训，等于用一次 9 卡时（≈¥17 + 一晚）在猜。

判据（写死在脚本里，避免事后解释）
--------------------------------
    * 小面积桶 IoU ≈ 0、大面积桶正常  → 归因「小目标」→ 提分辨率
    * 各面积桶都低                     → 归因「定位头能力」→ 加监督/换解码器
    * 各面积桶都还行但池化高得多       → 归因「口径」→ 差距来自像素加权

指标口径
--------
沿用 `src/engine/metrics.py` 的定义，不自创：
    池化 IoU   = Σinter / Σunion            （按像素加权，最宽松）
    逐图 IoU   = mean(inter_i / union_i)    （只在含 GT 篡改的图上，★文献标准口径）
本脚本额外给出**每个面积桶内的**池化与逐图两种值，以及像素级 recall/precision，
用来区分"整块没检出来"还是"检出来了但边界不准"。

用法
----
    # 用 09-28 这次的权重（注意：不是 checkpoints/vibnet_best.pt，那是 9/26 的旧权重）
    python scripts/diag_area_buckets.py \
        --ckpt outputs/upload/results_20260928/_pack/vibnet_best.pt \
        --config configs/default.yaml --dataset CASIAv2 --split test

    # 先小跑验证流程通不通
    python scripts/diag_area_buckets.py --ckpt ... --limit 128

输出
----
    outputs/diag/diag_area_buckets.md    报告（可直接贴进 docs）
    outputs/diag/diag_area_buckets.json  逐图明细（供后续画图）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from src.data.datasets import TamperDataset  # noqa: E402
from src.engine.metrics import (  # noqa: E402
    ClassificationMetrics,
    LocalizationMetrics,
)
from src.models.spatial_branch import align_cfg_to_checkpoint  # noqa: E402
from src.models.vibnet import build_model, load_config  # noqa: E402


# 面积桶：左开右闭 (lo, hi]，单位 = GT 篡改像素占全图比例
BUCKETS: List[Tuple[float, float, str]] = [
    (0.0, 0.005, "≤0.5%（极小）"),
    (0.005, 0.02, "0.5%~2%（小）"),
    (0.02, 0.05, "2%~5%（中小）"),
    (0.05, 0.10, "5%~10%（中）"),
    (0.10, 0.25, "10%~25%（大）"),
    (0.25, 1.01, ">25%（极大）"),
]


def _bucket_of(area: float) -> Optional[int]:
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo < area <= hi:
            return i
    return None


@torch.no_grad()
def run(cfg: dict, ckpt: str, dataset: str, split: str, batch_size: int,
        workers: int, limit: Optional[int], thr: float,
        device: str) -> Dict[str, object]:
    dev = torch.device(device)
    state = None
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        align_cfg_to_checkpoint(cfg, state.get("model", state))
        print(f"[diag] 已加载权重 {ckpt}")
    else:
        raise SystemExit(f"[diag] 找不到权重：{ckpt}")

    size = cfg["data"].get("image_size", 224)
    model = build_model(cfg)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[diag] 模型参数量 {n_par:.2f} M  image_size={size}  device={dev}")

    # 用数据集原样（不设 max_samples），保证与正式评测同一批样本
    ds = TamperDataset(cfg["data"]["root"], dataset, split, size, None, train=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=False, collate_fn=_collate,
    )

    cls_m = ClassificationMetrics()
    loc_m = LocalizationMetrics(thr)
    rows: List[dict] = []
    t_start = time.time()
    n_seen = 0

    for bi, batch in enumerate(loader):
        if limit and n_seen >= limit:
            break
        images = batch["image"].to(dev)
        out = model(images, sample_vib=False, beta=0.0)
        prob = out["mask_prob"].float().cpu().numpy()[:, 0]          # (B,H,W)
        gt = batch["mask"].numpy()[:, 0]                             # (B,H,W) {0,1}
        cls_m.update(out["cls_logits"].float().cpu().numpy(),
                     batch["label"].numpy())

        for i in range(prob.shape[0]):
            if limit and n_seen >= limit:
                break
            n_seen += 1
            g = gt[i] > 0.5
            p = prob[i] >= thr
            g_area = float(g.mean())
            p_area = float(p.mean())
            inter = float((p & g).sum())
            union = float((p | g).sum())
            iou = inter / union if union > 0 else float("nan")
            n_gt = int(g.sum())
            recall = inter / n_gt if n_gt > 0 else float("nan")
            prec = inter / float(p.sum()) if p.sum() > 0 else float("nan")
            rows.append({
                "name": batch["name"][i],
                "gt_area": g_area,
                "pred_area": p_area,
                "iou": iou,
                "px_recall": recall,
                "px_precision": prec,
                "bucket": _bucket_of(g_area),
                # 桶内池化 IoU 需要绝对像素量（Σinter / Σunion），
                # 由 iou/area 反推不精确，故原样留存
                "_inter": inter,
                "_union": union,
                "_n_pixels": float(g.size),
            })

        # 与正式评测口径对齐：只把 mask_valid 的样本喂给 loc_m
        mv = batch["mask_valid"].numpy()
        if mv.any():
            loc_m.update(prob[mv][:, None], gt[mv][:, None])

        if bi % 5 == 0:
            el = time.time() - t_start
            rate = n_seen / el if el > 0 else 0
            print(f"[diag] batch {bi:4d}  已处理 {n_seen:5d} 张  "
                  f"{rate:5.2f} 张/秒  已用 {el:6.1f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"[diag] 完成 {n_seen} 张，用时 {elapsed:.1f}s "
          f"({n_seen / max(elapsed, 1e-9):.2f} 张/秒)")
    return {
        "rows": rows,
        "cls": cls_m.compute(thr),
        "loc": loc_m.compute(),
        "meta": {
            "ckpt": ckpt, "dataset": dataset, "split": split,
            "image_size": size, "threshold": thr, "n": n_seen,
            "params_M": n_par, "elapsed_s": elapsed,
        },
    }


def _collate(batch):
    from src.data.datasets import collate_multitask
    return collate_multitask(batch)


# --------------------------------------------------------------------------
def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def build_report(res: Dict[str, object]) -> str:
    rows: List[dict] = res["rows"]          # type: ignore[assignment]
    loc: Dict[str, float] = res["loc"]      # type: ignore[assignment]
    cls: Dict[str, float] = res["cls"]      # type: ignore[assignment]
    meta: Dict[str, object] = res["meta"]   # type: ignore[assignment]

    real = [r for r in rows if r["gt_area"] <= 0]
    tam = [r for r in rows if r["gt_area"] > 0]

    L: List[str] = []
    L.append("# P0-① 定位归因诊断：按 GT 篡改面积分桶")
    L.append("")
    L.append(f"- 权重：`{meta['ckpt']}`")
    L.append(f"- 数据：`{meta['dataset']}/{meta['split']}`，"
             f"共 {meta['n']} 张（其中含篡改 {len(tam)} 张、真实 {len(real)} 张）")
    L.append(f"- 输入 {meta['image_size']}×{meta['image_size']}，"
             f"判定阈值 {meta['threshold']}，参数量 {meta['params_M']:.2f} M")
    L.append(f"- 耗时 {meta['elapsed_s']:.1f}s（CPU）")
    L.append("")

    # ---- 全量口径复现（与 gpu_eval_log 对拍）----
    L.append("## 一、全量口径复现（用于确认与云端评测同口径）")
    L.append("")
    L.append("| 口径 | 数值 |")
    L.append("|---|---|")
    L.append(f"| ★ 逐图（仅含篡改图，文献标准）`miou_tampered_only` | "
             f"**{_fmt(loc.get('miou_tampered_only'))}** |")
    L.append(f"| 池化 `miou` | {_fmt(loc.get('miou'))} |")
    L.append(f"| 全部非空图 `miou_per_sample` | {_fmt(loc.get('miou_per_sample'))} |")
    L.append(f"| Pixel Acc | {_fmt(loc.get('pixel_acc'))} |")
    L.append(f"| Dice | {_fmt(loc.get('dice'))} |")
    L.append(f"| n_tampered | {loc.get('n_tampered')} |")
    L.append("")
    L.append(f"分类（同批样本）：ACC {_fmt(cls.get('acc'))} / "
             f"F1 {_fmt(cls.get('f1'))} / AUC {_fmt(cls.get('auc'))} / "
             f"Precision {_fmt(cls.get('precision'))} / Recall {_fmt(cls.get('recall'))}")
    L.append("")
    L.append("> 云端 `eval_casia.json` 的对应值为 "
             "`miou_tampered_only = 0.2668`、池化 0.4570、逐图 0.2525。"
             "本表与之一致才能继续往下看分桶结论。")
    L.append("")

    # ---- 分桶主表 ----
    L.append("## 二、★ 按 GT 篡改面积分桶")
    L.append("")
    L.append("| 面积桶 | n | 逐图 IoU 均值 ★ | 池化 IoU | 像素召回 | 像素精度 | "
             "预测面积/GT 面积 | IoU≥0.1 占比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, (_, _, lab) in enumerate(BUCKETS):
        b = [r for r in rows if r["bucket"] == i]
        if not b:
            L.append(f"| {lab} | 0 | — | — | — | — | — | — |")
            continue
        ious = [r["iou"] for r in b if not np.isnan(r["iou"])]
        # 池化需 Σinter/Σunion 的绝对像素量，见 rows 里的 _inter/_union
        pooled = _bucket_pooled(b)
        pxr = np.nanmean([r["px_recall"] for r in b]) if b else float("nan")
        pxp = np.nanmean([r["px_precision"] for r in b]) if b else float("nan")
        ratio = np.nanmean([r["pred_area"] / max(r["gt_area"], 1e-9) for r in b]) \
            if b else float("nan")
        hit = np.mean([1.0 if (not np.isnan(r["iou"]) and r["iou"] >= 0.1) else 0.0
                       for r in b]) if b else float("nan")
        L.append(f"| {lab} | {len(b)} | **{_fmt(np.mean(ious))}** | {_fmt(pooled)} | "
                 f"{_fmt(pxr)} | {_fmt(pxp)} | {_fmt(ratio, 2)} | {_fmt(hit, 3)} |")
    L.append("")
    L.append("> 「预测面积/GT 面积」= 平均(预测为正的像素比例 ÷ GT 篡改比例)。"
             "远小于 1 = 系统性**欠分割**（该检的没检出来）；远大于 1 = 过分割/误报。")
    L.append("")

    # ---- 归因判定 ----
    L.append("## 三、归因判定（自动给出）")
    L.append("")
    L.extend(_verdict(rows, tam, loc))
    L.append("")

    # ---- 真实图误报 ----
    if real:
        fp_rate = np.mean([1.0 if r["pred_area"] > 0.005 else 0.0 for r in real])
        L.append("## 四、真实图误报（定位支路的假阳性）")
        L.append("")
        L.append(f"- 真实图 {len(real)} 张，其中 {fp_rate * 100:.1f}% 的图预测出了"
                 f"超过 0.5% 面积的\"篡改区域\"")
        L.append(f"- 平均预测面积占比 {_fmt(np.mean([r['pred_area'] for r in real]))}")
        L.append("")
        L.append("> 真实图上定位支路本应输出全零。误报率高会同时拖累"
                 "`miou_per_sample` 与人工复核体验。")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"*由 `scripts/diag_area_buckets.py` 生成 · {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(L)


def _bucket_pooled(b: List[dict]) -> float:
    """桶内池化 IoU。用 gt_area/pred_area/iou 无法精确还原 union，
    故由脚本在 rows 里另存 inter/union 的绝对值（见 run()）。"""
    num = sum(r.get("_inter", 0.0) for r in b)
    den = sum(r.get("_union", 0.0) for r in b)
    return num / den if den > 0 else float("nan")


def _verdict(rows, tam, loc) -> List[str]:
    """自动判定归因方向 —— 判据写死在代码里，避免事后挑解释。"""
    out: List[str] = []
    small = [r for r in rows if r["gt_area"] > 0 and r["gt_area"] <= 0.02]
    large = [r for r in rows if r["gt_area"] > 0.10]
    s_iou = np.mean([r["iou"] for r in small]) if small else float("nan")
    l_iou = np.mean([r["iou"] for r in large]) if large else float("nan")
    overall = loc.get("miou_tampered_only", float("nan"))
    pooled = loc.get("miou", float("nan"))

    out.append("| 现象 | 实测 |")
    out.append("|---|---|")
    out.append(f"| 小面积桶（≤2%）逐图 IoU | {_fmt(s_iou)} （n={len(small)}） |")
    out.append(f"| 大面积桶（>10%）逐图 IoU | {_fmt(l_iou)} （n={len(large)}） |")
    out.append(f"| 全体逐图 IoU | {_fmt(overall)} |")
    out.append(f"| 池化 IoU | {_fmt(pooled)} |")
    out.append("")

    ok_s = not np.isnan(s_iou) and s_iou >= 0.30
    ok_l = not np.isnan(l_iou) and l_iou >= 0.30
    if not ok_s and ok_l:
        out.append("**→ 归因：小目标被压掉。** 大面积桶正常而小面积桶塌陷。")
        out.append("对策优先级：① 输入分辨率 224→320（小区域像素数 ×2，代价单轮 ≈2.2×）；"
                   "② 定位头输出层上采样到更高分辨率；③ 边缘损失权重上调。")
    elif ok_s and not ok_l:
        out.append("**→ 归因：大面积反而做不好**（少见）。检查掩码是否被整体偏置、"
                   "或损失被小区域主导。")
    elif not ok_s and not ok_l:
        out.append("**→ 归因：定位头整体能力不足（各面积桶都不高）。**")
        out.append("对策优先级：① 加宽 `refine_channels` / 解码器通道；"
                   "② 提高边缘监督权重；③ 检查 stage2 的解冻范围与学习率。")
        out.append("**注意：这条结论下，单纯提分辨率收益有限** —— "
                   "不是分辨率不够，是定位能力本身不够。")
    else:
        out.append("**→ 归因：各面积桶均在自己可接受范围**，"
                   "与目标 0.56 的差距主要来自边界精度而非检出率。")

    if not np.isnan(pooled) and not np.isnan(overall) and (pooled - overall) > 0.15:
        out.append("")
        out.append(f"**⚠ 口径差距显著**：池化 {_fmt(pooled)} 比逐图 {_fmt(overall)} "
                   f"高 {pooled - overall:.3f} —— 说明**大区域图在拉高池化值**，"
                   "论文里必须报 `miou_tampered_only` 并注明，不要挑池化值。")
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="CASIAv2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 张（自检用）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/diag")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    res = run(cfg, args.ckpt, args.dataset, args.split, args.batch_size,
              args.workers, args.limit, args.threshold, args.device)

    md = build_report(res)
    md_path = os.path.join(args.out, "diag_area_buckets.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    json_path = os.path.join(args.out, "diag_area_buckets.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)

    print("\n" + "=" * 70)
    print(md)
    print("=" * 70)
    print(f"[diag] 报告已写入 {md_path}")
    print(f"[diag] 明细已写入 {json_path}")


if __name__ == "__main__":
    main()
