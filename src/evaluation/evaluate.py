"""主评测脚本：真伪检测 + 篡改定位 + 推理性能（申报书 6.3）

用法：
    python -m src.evaluation.evaluate --ckpt checkpoints/vibnet_best.pt --split test
    python -m src.evaluation.evaluate --ckpt checkpoints/vibnet_best.pt --demo
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.datasets import build_dataloaders
from ..engine.metrics import ClassificationMetrics, LocalizationMetrics
from ..models.spatial_branch import align_cfg_to_checkpoint
from ..models.vibnet import build_model, load_config


@torch.no_grad()
def evaluate_model(
    cfg: dict,
    ckpt: Optional[str] = None,
    split: str = "test",
    use_demo: bool = False,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    save_dir: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Dict[str, object]:
    """在指定 split 上评测，返回 {cls, loc, perf} 三组指标。

    `dataset` 用于把该 split 限定为**单个数据集**（按配置里的 `name` 匹配）。
    为什么需要：多个数据集混合时是按顺序拼接、且 test loader `shuffle=False`，
    排在后面的大数据集可能整个批不到（实测 `default.yaml` 的 test_sets 里
    ForenSynths 在前，`--max-batches` 小的时候 CASIAv2 一张都没轮到，
    定位指标静默为空）。测定位指标时应显式 `--dataset CASIAv2`。
    """
    if dataset:
        key = {"train": "train_sets", "val": "val_sets",
               "test": "test_sets"}.get(split, "test_sets")
        sets = cfg["data"].get(key) or []
        keep = [s for s in sets
                if str(s.get("name", "")).lower() == dataset.lower()]
        if not keep:
            raise SystemExit(
                f"[eval] {key} 里没有名为 {dataset} 的数据集；"
                f"现有：{[s.get('name') for s in sets]}")
        import copy as _copy
        cfg = _copy.deepcopy(cfg)
        cfg["data"][key] = keep
        # val_sets 常由 train_sets 派生；若评测 val 也一并收窄，避免又混进来
        print(f"[eval] 已把 {key} 限定为 {dataset}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = None
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # 骨干档位可能与当前 YAML 不一致（见 spatial_branch.infer_backbone_variant），
        # 先按 checkpoint 反推并对齐，避免抛出难以理解的 size mismatch
        align_cfg_to_checkpoint(cfg, state.get("model", state))
    model = build_model(cfg)
    if state is not None:
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"[eval] 已加载 {ckpt}")
    else:
        print(f"[eval] ⚠ 未找到 {ckpt}，使用随机初始化模型（指标无意义，仅供流程自检）")
    model.to(dev).eval()

    loaders = build_dataloaders(cfg, use_demo=use_demo)
    loader = loaders[split if split in loaders else "test"]

    thr = cfg["eval"].get("threshold", 0.5)
    cls_m = ClassificationMetrics()
    loc_m = LocalizationMetrics(thr)
    latencies: List[float] = []
    n_loc_samples = 0

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = batch["image"].to(dev)
        labels = batch["label"].to(dev)
        mask = batch["mask"].to(dev) if batch["mask"] is not None else None
        mvalid = batch["mask_valid"].to(dev) if "mask_valid" in batch else None

        t0 = time.perf_counter()
        out = model(images, sample_vib=False, beta=0.0)
        latencies.append((time.perf_counter() - t0) / max(1, images.shape[0]) * 1000)

        cls_m.update(out["cls_logits"].float().cpu().numpy(), labels.cpu().numpy())
        if mask is not None and mvalid is not None and bool(mvalid.any()):
            idx = mvalid.cpu().numpy()
            loc_m.update(out["mask_prob"].float().cpu().numpy()[idx], mask.cpu().numpy()[idx])
            n_loc_samples += int(idx.sum())

    # ---- 定位指标为空时，给出**具体原因**（别让报告只丢一句"无像素级 GT"）
    #
    # 实测踩到：`--config configs/default.yaml --split test` 的 test_sets 里
    # ForenSynths 排在 CASIAv2 前面，而 test loader 是 shuffle=False，
    # 于是前 N 批全是整图真伪样本 → n_loc_samples=0 → 定位指标**静默消失**。
    # 报告里只写"该 split 无像素级 GT"会让人以为数据集本身没有掩码。
    loc_absent = None
    if n_loc_samples == 0:
        sets = cfg["data"].get(
            {"train": "train_sets", "val": "val_sets", "test": "test_sets"}
            .get(split, "test_sets"), []) or []
        # kind 是配置里显式声明的（CASIAv2 / COVERAGE 都写 kind: tamper）；
        # 名字兜底是为了容忍没写 kind 的手写配置。
        _TAMPER_HINT = ("casiav2", "casia2", "casia", "coverage")
        tamper_sets = [s.get("name", "?") for s in sets
                       if s.get("kind") == "tamper"
                       or str(s.get("name", "")).lower() in _TAMPER_HINT]
        if tamper_sets:
            loc_absent = (
                f"配置里**有**带掩码的数据集 {tamper_sets}，但一条像素级样本都没轮到。"
                f"常见原因：① 多个数据集混合时按顺序拼接、test loader `shuffle=False`，"
                f"前若干批全是排在前面的大数据集（如 ForenSynths）；"
                f"② `--max-batches` 设得太小。"
                f"→ 想测定位指标，请让 test_sets **只含**带掩码的数据集"
                f"（把 test_sets 收窄成单个 tamper 数据集即可；"
                f"外部域 COVERAGE 用 scripts/eval_external_domain.py 单独评）。"
            )
        else:
            loc_absent = ("该 split 配置的数据集都不含像素级掩码"
                          "（ForenSynths 是整图真伪数据集，只有 0_real/1_fake，没有 mask）。")

    result: Dict[str, object] = {
        "split": split,
        "ckpt": ckpt,
        "device": str(dev),
        "cls": cls_m.compute(thr),
        "loc": loc_m.compute() if n_loc_samples > 0 else {},
        "n_loc_samples": n_loc_samples,
        "loc_absent_reason": loc_absent,
        "perf": {
            "latency_ms_per_image": float(np.mean(latencies)) if latencies else None,
            "fps": float(1000.0 / np.mean(latencies)) if latencies else None,
            "model_params_M": model.summary()["total(M)"],
            "model_fp32_MB": model.summary()["fp32_size(MB)"],
        },
    }

    # ---- 达标核对（申报书 6.3 预期成果）
    #
    # ⚠ 两处口径陷阱（答辩前必看 docs/09 第 5 节）：
    #   1) use_demo=True 时用的是**程序化合成数据**，不是 ForenSynths，
    #      把它的 ACC 对标 "ForenSynths >= 91%" 是错的 —— 这里只作为"流程是否跑通"的信号；
    #   2) mIoU 有三种口径，**数值能差一倍以上**，必须标明报的是哪一个：
    #        miou                = 全局像素汇总后算 IoU（池化，被大区域图主导，最宽松）
    #        miou_tampered_only  = 只在含篡改区域的图上逐图算 IoU 再平均（文献标准口径）
    #        miou_per_sample     = 在所有"预测或 GT 非空"的图上平均（含误报的真实图，最严格）
    #
    #    ⚠️ 曾误以为池化值偏高是因为"真实图全零掩码按 IoU=1 计入"——**这是错的**：
    #       `LocalizationMetrics.update()` 里 `if union > 0` 才计入逐图列表，
    #       全零预测的真实图是**被跳过**的。池化值偏高的真实原因是**按像素加权**：
    #       大篡改区域定位准则 tp 贡献大，而小区域图在逐图平均里权重相同、IoU 常很低。
    #       三种口径的定义由 `scripts/test_metric_definitions.py` 用手算样例钉死。
    exp = {
        "ForenSynths 检测准确率 >= 91%": None,
        "CASIAv2 篡改定位 mIoU >= 56%": None,
        "CPU 推理 >= 8 张/秒(224x224)": result["perf"]["fps"] >= 8
        if result["perf"]["fps"] else None,
        "模型体积 <= 120MB": round(result["perf"]["model_fp32_MB"], 1) <= 120,
    }
    if not use_demo and split.lower().startswith("foren"):
        exp["ForenSynths 检测准确率 >= 91%"] = result["cls"].get("acc", 0) >= 0.91
    if not use_demo and result["loc"]:
        # 以文献标准口径（只在篡改图上平均）判定；三种口径全部写进 miou_conventions
        exp["CASIAv2 篡改定位 mIoU >= 56%"] = (
            result["loc"].get("miou_tampered_only", 0) >= 0.56
        )
    result["targets"] = exp
    if result["loc"]:
        result["miou_conventions"] = {
            "miou": result["loc"].get("miou"),
            "miou_tampered_only": result["loc"].get("miou_tampered_only"),
            "miou_per_sample": result["loc"].get("miou_per_sample"),
            "n_tampered": result["loc"].get("n_tampered"),
            "判定所用口径": "miou_tampered_only（含篡改区域的图逐图平均，文献标准）",
            "说明": "另外两个口径仅作参考；**不要挑最高的那个报**。",
        }
    result["is_synthetic_demo"] = bool(use_demo)
    if use_demo:
        result["demo_caveat"] = (
            "本次评测使用程序化合成数据，精度指标不可与申报书真实数据集指标对标；"
            "仅用于验证评测流程与指标计算是否正确。"
        )
    if result["loc"] and result["loc"].get("miou", 0) - result["loc"].get("miou_per_sample", 0) > 0.15:
        result["miou_caveat"] = (
            f"三种 mIoU 口径差异较大（池化 {result['loc']['miou']:.3f} / "
            f"仅篡改图 {result['loc'].get('miou_tampered_only', 0):.3f} / "
            f"全部非空图 {result['loc']['miou_per_sample']:.3f}）。"
            "原因是池化口径**按像素加权**（大篡改区域主导），不是"
            "『真实图全零掩码按 IoU=1 计入』——后者是错的，全零图是被跳过的。"
            "论文请报 miou_tampered_only 并注明口径；不要挑最高的那个报。"
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"eval_{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=float)
        print(f"[eval] 结果已写入 {path}")
    return result


def format_report(result: Dict[str, object]) -> str:
    """把评测结果格式化成可直接贴进论文/技术报告的表格式文本。"""
    cls = result.get("cls", {}) or {}
    loc = result.get("loc", {}) or {}
    perf = result.get("perf", {}) or {}
    lines = [
        "=" * 62,
        f"评测报告  split={result.get('split')}  device={result.get('device')}",
        "=" * 62,
        "【真伪检测指标】",
        f"  ACC={cls.get('acc', float('nan')):.4f}   "
        f"Precision={cls.get('precision', float('nan')):.4f}   "
        f"Recall={cls.get('recall', float('nan')):.4f}",
        f"  F1={cls.get('f1', float('nan')):.4f}   "
        f"AUC={cls.get('auc', float('nan')):.4f}   "
        f"AP(mAP)={cls.get('ap', float('nan')):.4f}",
        f"  TP={cls.get('tp')} FP={cls.get('fp')} FN={cls.get('fn')} TN={cls.get('tn')}",
    ]
    if loc:
        lines += [
            "【篡改定位指标】（三种 mIoU 口径都列出，报哪个必须写明）",
            f"  ① 池化 mIoU（像素加权，最宽松）        = "
            f"{loc.get('miou', float('nan')):.4f}",
            f"  ② 仅篡改图 mIoU（文献标准，★判定用）  = "
            f"{loc.get('miou_tampered_only', float('nan')):.4f}"
            f"   （n_tampered={loc.get('n_tampered')}）",
            f"  ③ 全部非空图 mIoU（含误报，最严格）    = "
            f"{loc.get('miou_per_sample', float('nan')):.4f}",
            f"  Pixel Acc={loc.get('pixel_acc', float('nan')):.4f}   "
            f"Dice={loc.get('dice', float('nan')):.4f}   F1={loc.get('f1', float('nan')):.4f}",
            f"  评测样本数={result.get('n_loc_samples')}",
        ]
    else:
        lines.append("【篡改定位指标】该 split 没有产生任何像素级样本（n_loc_samples=0）")
        why = result.get("loc_absent_reason")
        if why:
            lines.append(f"  ⚠ {why}")
    lines += [
        "【推理性能】",
        f"  单张延迟={perf.get('latency_ms_per_image', float('nan')):.2f} ms   "
        f"吞吐={perf.get('fps', float('nan')):.2f} 张/秒",
        f"  参数量={perf.get('model_params_M', float('nan')):.2f} M   "
        f"FP32 体积≈{perf.get('model_fp32_MB', float('nan')):.1f} MB",
    ]
    if result.get("is_synthetic_demo"):
        lines.append("  ⚠ 本次为合成演示数据，吞吐是 PyTorch eager 口径，"
                     "验收请用 scripts/benchmark_cpu.py（ONNX 路径）")
    if result.get("miou_caveat"):
        lines.append(f"  ⚠ {result['miou_caveat']}")
    lines.append("【申报书预期指标核对】")
    for k, v in (result.get("targets") or {}).items():
        mark = "达标" if v else ("未达标" if v is False else "无数据")
        lines.append(f"  [{mark}] {k}")
    if result.get("is_synthetic_demo"):
        lines.append("  ⚠ 演示数据集下的精度项不作为验收依据（无像素级真实标注语义）")
    lines.append("=" * 62)
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/vibnet_best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--demo", action="store_true", help="使用离线演示数据集")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--save-dir", default="outputs")
    ap.add_argument("--dataset", default=None,
                    help="把该 split 限定为单个数据集（按配置里的 name 匹配），"
                         "例如 --dataset CASIAv2。测定位 mIoU 时建议显式指定，"
                         "否则混合 test_sets 里排在后面的带掩码数据集可能批不到。")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.demo:
        cfg["data"]["image_size"] = 224
    res = evaluate_model(cfg, args.ckpt, args.split, args.demo, args.device,
                         args.max_batches, args.save_dir, args.dataset)
    print(format_report(res))


if __name__ == "__main__":
    main()
