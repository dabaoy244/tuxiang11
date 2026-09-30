"""β 生效版受控对照实验（full + no_vib）—— 产物隔离，零覆盖主链路结果。

为什么要单独一个入口
    `src/evaluation/ablation.py::main()` **不暴露** `save_dir`，产物固定写
    `outputs/ablation/<preset>/`。而 `run_ablation()` 本身是**有** `save_dir` 参数的。
    ⇒ 直接跑 `-m src.evaluation.ablation` 会覆盖 2026-09-30 那次 `full` 的权重目录
      （`outputs/ablation/full/ckpt/ablation_full_best.pt`），而那正是跨生成器双行
      与 COVERAGE 外部域行的来源。本入口直接调 `run_ablation()` 并指定独立 save_dir，
      实现**零覆盖**（并有硬断言拦住误用）。

它回答什么问题
    主消融表（`default_crossval.yaml`，β 升温窗 [20,40]）在实况早停下 β 峰值只有
    0.0050（= 目标的 5%）⇒ VIB 的 KL 正则**几乎没参与训练**。
    本实验用 `default_crossval_betafull.yaml`（窗口下移到 [11,18]，零成本验算显示
    β 可达 0.1000 并保持饱和）重跑 `full` 与 `no_vib`，构成**同配置受控对照**：
    唯一变量是 VIB 开关 ⇒ 回答「信息瓶颈真正生效时，VIB 是否带来增益」。
    ⚠ 本组结果**不可与主消融表混表并列**（协议不同），须单独成表。

用法
    python -m src.evaluation.beta_control \
        --config configs/default_crossval_betafull.yaml \
        --presets full no_vib --epochs-scale 0.82
"""

from __future__ import annotations

import argparse

from ..models.vibnet import load_config
from .ablation import run_ablation

DEFAULT_SAVE_DIR = "outputs/ablation_betafull"
# 主消融表的产物根：任何情况下都不允许写入（会覆盖 full 的权威权重）
FORBIDDEN_SAVE_DIR = "outputs/ablation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_crossval_betafull.yaml")
    ap.add_argument("--presets", nargs="*", default=["full", "no_vib"])
    ap.add_argument("--epochs-scale", type=float, default=0.82)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dir", default=DEFAULT_SAVE_DIR,
                    help="产物根目录；默认与主消融表隔离")
    args = ap.parse_args()

    if args.save_dir.rstrip("/") == FORBIDDEN_SAVE_DIR:
        raise SystemExit(
            f"[beta_control] 拒绝把产物写进 {FORBIDDEN_SAVE_DIR} —— 那会覆盖主消融表"
            "（含 full 的权威权重，跨生成器/COVERAGE 两处结果都由它复现）。"
        )

    cfg = load_config(args.config)
    print(f"[beta_control] 配置      = {args.config}")
    print(f"[beta_control] 臂        = {args.presets}")
    print(f"[beta_control] 产物根    = {args.save_dir}")
    print(f"[beta_control] epochs_scale = {args.epochs_scale}")
    run_ablation(cfg, args.presets, "train", None, False, args.device,
                 save_dir=args.save_dir, epochs_scale=args.epochs_scale)


if __name__ == "__main__":
    main()
