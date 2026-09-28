#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「预训练 vs 随机初始化」对照实验的两份配置。

为什么必须成对生成
------------------
这个对照实验的全部说服力都来自**只差一个变量**。手写两份 YAML 必然出现
笔误（少改一个 lr、多留一个 max_samples），而这种差异会让结论作废。
所以两份配置由同一个脚本、同一份基线、同一组覆盖项生成，
唯一的区别就是 `backbone_weights` 有没有值。

实验设计
--------
  A  arm=random     : backbone_variant=b16，骨干**随机初始化**（复现当前失败现象）
  B  arm=pretrained : backbone_variant=b16 + pretrained/vit_b16_imagenet.pt

  两臂：同数据、同轮数、同学习率、同随机种子、同评测子集。
  唯一差异 → 骨干是否有 ImageNet 预训练权重。

  期望：A 的 val_cls_acc 停在 ~0.5（全判同一类），B 明显超过 0.5。
  这就把「模型训不起来」定位到"缺预训练先验"，而不是代码 bug。

用法
----
    python scripts/make_ablation_configs.py                     # 用默认规模
    python scripts/make_ablation_configs.py --train-samples 1600 --epochs 3
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER = """# ============================================================================
# 对照实验配置：{arm_cn}（{arm}）
# ----------------------------------------------------------------------------
# ⚠ 本文件由 scripts/make_ablation_configs.py 自动生成，请勿手改 ——
#   手改会让两臂失去可比性，直接改生成脚本再重跑。
#
# 实验协议（两臂完全一致，只有「骨干是否加载预训练权重」不同）：
#   训练集 : ForenSynths/val  {train_samples} 张（ProGAN 20 类保留集）
#   验证集 : ForenSynths/test {eval_samples} 张
#            ⚠ 此处原写「与训练集图像无交集」，**该说法是错的**（2026-09-19 实测，
#              见 scripts/audit_train_eval_overlap.py）：pool 级有 400 张真图
#              与训练集 val **内容逐字节相同**（cat 200/200、horse 200/200），
#              占 val 真图的 10%；当前 {eval_samples} 张子采样恰好命中 3 张
#              （0.23%），对指标影响可忽略，但**换子采样规模就会放大**。
#              结论口径：只能说"本次子采样实测重叠 3 张"，不能说"无交集"。
#   骨干   : ViT-B/16（768 维 / 12 层 / 12 头）
#   轮数   : {epochs}   batch={batch}   随机种子={seed}
#   阶段   : 仅 stage1_cls_pretrain（分类分支预训练）
#
# 骨干权重 : {weights_desc}
#
# 已知口径限制（写论文/中期材料时必须带上）：
#   * 训练集是 ProGAN 的**验证划分**，非官方 train 划分 —— 官方 train
#     (progan_train.7z, 70GB) 在受限网络下未能取得。二者同为 ProGAN 生成、
#     与测试集无图像重叠，可作为小规模替代，但**不能声称复现了官方协议**。
#   * 评测在 ForenSynths/test 上做，但为控制 CPU 成本做了子采样，
#     逐生成器结论请用 scripts/eval_cross_generator.py 分层评测。
# ============================================================================
"""


def build(arm: str, args) -> dict:
    base = os.path.join(ROOT, "configs", "default.yaml")
    with open(base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- 骨干：走 tiny_vit 通道（离线可构造 + 支持 backbone_weights） --------
    sp = cfg["model"]["spatial"]
    sp["backbone"] = "tiny_vit"
    sp["backbone_variant"] = "b16"
    sp["global_dim"] = 768
    sp.pop("clip_pretrained", None)
    sp.pop("clip_local_dir", None)
    pretrained_rel = "pretrained/vit_b16_imagenet.pt"
    if arm == "pretrained":
        sp["backbone_weights"] = pretrained_rel
        weights_desc = f"加载 `{pretrained_rel}`（ImageNet-1K 预训练 ViT-B/16）"
    else:
        sp.pop("backbone_weights", None)
        weights_desc = "**不加载**（骨干随机初始化，复现失败现象）"

    # --- 数据 -------------------------------------------------------------
    cfg["data"]["root"] = "data/Datasets"
    cfg["data"]["num_workers"] = 0          # Windows + 小数据，避免多进程开销
    cfg["data"]["train_sets"] = [
        {"name": "ForenSynths", "kind": "gensynth", "split": "val",
         "max_samples": args.train_samples}]
    cfg["data"]["val_sets"] = [
        {"name": "ForenSynths", "kind": "gensynth", "split": "test",
         "max_samples": args.eval_samples}]
    cfg["data"]["test_sets"] = [
        {"name": "ForenSynths", "kind": "gensynth", "split": "test",
         "max_samples": args.eval_samples}]

    # --- 训练：只留 stage1 -------------------------------------------------
    st = cfg["train"]["stages"][0]
    st["name"] = "stage1_cls_pretrain"
    st["epochs"] = args.epochs
    st["lr"] = args.lr
    st["lr_min"] = args.lr / 100.0
    st["freeze"] = ["localization"]
    st["train"] = ["spatial", "freq", "fusion", "vib", "cls_head"]
    st["use_tasks"] = ["cls"]
    st["beta_epoch_offset"] = 0
    cfg["train"]["stages"] = [st]
    cfg["train"]["batch_size"] = args.batch
    cfg["train"]["log_every"] = max(1, args.log_every)
    cfg["train"]["save_every"] = 1
    cfg["train"]["eval_every"] = 1
    cfg["train"]["monitor"] = "val_cls_acc"
    cfg["train"]["early_stop_patience"] = 99      # 固定轮数，不做早停（保证两臂同轮）
    cfg["train"]["amp"] = False

    # --- 项目 -------------------------------------------------------------
    cfg["project"]["name"] = f"VIB-Net 对照实验 ({arm})"
    cfg["project"]["seed"] = args.seed
    cfg["project"]["output_dir"] = f"outputs/ablation_{arm}"
    cfg["project"]["ckpt_dir"] = "checkpoints"

    hdr = HEADER.format(arm=arm,
                        arm_cn="随机初始化" if arm == "random" else "ImageNet 预训练",
                        train_samples=args.train_samples,
                        eval_samples=args.eval_samples,
                        epochs=args.epochs, batch=args.batch, seed=args.seed,
                        weights_desc=weights_desc)
    return hdr, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="生成对照实验配置")
    ap.add_argument("--train-samples", type=int, default=1600)
    ap.add_argument("--eval-samples", type=int, default=1300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    for arm in ("random", "pretrained"):
        hdr, cfg = build(arm, args)
        out = os.path.join(ROOT, "configs", f"ablation_b16_{arm}.yaml")
        buf = io.StringIO()
        yaml.safe_dump(cfg, buf, allow_unicode=True, sort_keys=False,
                       default_flow_style=False)
        with open(out, "w", encoding="utf-8") as f:
            f.write(hdr + buf.getvalue())
        sp = cfg["model"]["spatial"]
        print(f"✅ {out}")
        print(f"   backbone_variant = {sp['backbone_variant']}"
              f"   backbone_weights = {sp.get('backbone_weights', '(无)')}")
    print(f"\n两臂协议：train={args.train_samples} 张 / eval={args.eval_samples} 张"
          f" / {args.epochs} 轮 / batch={args.batch} / seed={args.seed}")
    print("差异项：仅 model.spatial.backbone_weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
