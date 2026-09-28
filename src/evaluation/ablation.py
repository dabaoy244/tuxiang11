"""消融实验（申报书 6.3 "对比实验与消融实验"）

逐项移除关键模块，验证每个模块的有效性：
    full                  完整模型
    no_learnable_mask     可学习频域掩码 -> 固定中频掩码
    no_phase              去掉相位支路（仅用幅值）
    no_cross_attention    CS-CAM -> 直接相加
    no_vib                去掉分层 VIB（分类支路直接用 F_fusion）
    no_grad_stop          去掉梯度停止层
    no_edge               去掉 Canny 边缘监督

⚠ 正确的消融流程（写论文必须这么做）：
    每一项都要**重新训练**，然后用同一套测试集评测。
    只加载同一个 ckpt 再关掉模块，只能说明"该模块对推理有影响"，
    不能说明"该模块对训练/泛化有贡献"，审稿人会直接指出这一点。

因此本脚本提供两种模式：
    --mode eval   ：快速自检（加载 full 的 ckpt，推理时关模块）；仅用于流程验证
    --mode train  ：每项独立训练 + 评测；论文里用的就是这个结果

用法：
    python -m src.evaluation.ablation --mode eval  --demo
    python -m src.evaluation.ablation --mode train --epochs-scale 1.0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List, Optional

from ..models.vibnet import build_model, load_config

# 每个消融项对应的配置覆盖（点号路径）
ABLATION_PRESETS: Dict[str, Dict[str, object]] = {
    "full": {},
    "no_learnable_mask": {"model.freq.use_learnable_mask": False},
    "no_phase": {"model.freq.use_phase": False},
    "no_cross_attention": {"model.fusion.use_cross_attention": False},
    "no_vib": {"model.vib.use_vib": False},
    "no_grad_stop": {"model.localization.use_grad_stop": False},
    "no_edge": {"model.localization.use_edge_head": False},
}


def _set_nested(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def apply_preset(cfg: dict, preset: str) -> dict:
    if preset not in ABLATION_PRESETS:
        raise KeyError(f"未知消融项 {preset}，可选：{list(ABLATION_PRESETS)}")
    out = copy.deepcopy(cfg)
    for k, v in ABLATION_PRESETS[preset].items():
        _set_nested(out, k, v)
    return out


def describe_preset(preset: str) -> str:
    overrides = ABLATION_PRESETS.get(preset, {})
    if not overrides:
        return "完整模型"
    return "；".join(f"{k} -> {v}" for k, v in overrides.items())


# ==========================================================================
def run_ablation(
    cfg: dict,
    presets: Optional[List[str]] = None,
    mode: str = "eval",
    ckpt: Optional[str] = None,
    use_demo: bool = False,
    device: Optional[str] = None,
    save_dir: str = "outputs/ablation",
    epochs_scale: float = 1.0,
) -> Dict[str, object]:
    from ..evaluation.evaluate import evaluate_model

    presets = presets or list(ABLATION_PRESETS.keys())
    os.makedirs(save_dir, exist_ok=True)
    table: Dict[str, object] = {}

    for p in presets:
        sub_cfg = apply_preset(cfg, p)
        sub_cfg["project"]["output_dir"] = os.path.join(save_dir, p)
        sub_cfg["project"]["ckpt_dir"] = os.path.join(save_dir, p, "ckpt")
        print(f"\n{'='*70}\n[ablation] {p}: {describe_preset(p)}\n{'='*70}")

        sub_ckpt = ckpt
        if mode == "train":
            from ..data.datasets import build_dataloaders
            from ..engine.trainer import Trainer, set_seed
            import torch

            set_seed(sub_cfg["project"].get("seed", 3407))
            if epochs_scale != 1.0:
                for s in sub_cfg["train"]["stages"]:
                    s["epochs"] = max(1, int(round(s["epochs"] * epochs_scale)))
            loaders = build_dataloaders(sub_cfg, use_demo=use_demo)
            model = build_model(sub_cfg)
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            trainer = Trainer(sub_cfg, model, dev, loaders["train"], loaders["val"],
                              ckpt_prefix=f"ablation_{p}")
            trainer.fit()
            sub_ckpt = os.path.join(sub_cfg["project"]["ckpt_dir"], f"ablation_{p}_best.pt")

        res = evaluate_model(sub_cfg, sub_ckpt, "test", use_demo, device,
                             save_dir=sub_cfg["project"]["output_dir"])
        table[p] = {
            "config": describe_preset(p),
            "cls": {k: res["cls"].get(k) for k in ("acc", "precision", "recall", "f1", "auc", "ap")},
            "loc": {k: res["loc"].get(k) for k in ("miou", "pixel_acc", "dice", "f1")} if res["loc"] else {},
            "perf": res["perf"],
        }

    # ---- 与 full 的差值（消融表的核心列）
    base = table.get("full", {})
    for p, row in table.items():
        d_acc = (base.get("cls", {}).get("acc") or 0) - (row["cls"].get("acc") or 0)
        d_miou = (base.get("loc", {}).get("miou") or 0) - (row["loc"].get("miou") or 0)
        row["delta_vs_full"] = {"acc": round(d_acc, 4), "miou": round(d_miou, 4)}

    out_path = os.path.join(save_dir, "ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2, default=float)
    md_path = os.path.join(save_dir, "ablation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| 消融项 | 说明 | ACC | F1 | AUC | mIoU | Dice | ΔACC | ΔmIoU |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for p, r in table.items():
            c, l = r["cls"], r["loc"]
            f.write(f"| {p} | {r['config']} | {c.get('acc', 0):.4f} | {c.get('f1', 0):.4f} | "
                    f"{c.get('auc', 0):.4f} | {l.get('miou', 0):.4f} | {l.get('dice', 0):.4f} | "
                    f"{r['delta_vs_full']['acc']:+.4f} | {r['delta_vs_full']['miou']:+.4f} |\n")
    print(f"\n[ablation] 结果已写入 {out_path}\n           Markdown 表格：{md_path}")
    if mode == "eval":
        print("[ablation] ⚠ 当前为 eval 模式（复用同一 ckpt），结果只能用于流程自检；"
              "论文数据请用 --mode train 重新训练每项。")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", default="eval", choices=["eval", "train"])
    ap.add_argument("--presets", nargs="*", default=None)
    ap.add_argument("--ckpt", default="checkpoints/vibnet_best.pt")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs-scale", type=float, default=1.0,
                    help="train 模式下按比例缩短各阶段轮数（演示用，如 0.05）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_ablation(cfg, args.presets, args.mode, args.ckpt, args.demo, args.device,
                 epochs_scale=args.epochs_scale)


if __name__ == "__main__":
    main()
