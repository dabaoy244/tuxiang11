#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外部域（零样本 / 跨数据集）评测：把某个「完全没参与训练」的数据集当**独立基准行**来评。

为什么单独写这个脚本
--------------------
1. **不复制配置。** 模型 / 损失 / 评测段一律在运行时从
   `configs/default_crossval.yaml` 现读，只覆盖 `data.root` 与三个 split 列表。
   写第二份「几乎一样」的 YAML 必然漂移，本项目已经吃过一次「同名不同义」的亏
   （docs/08 D23）。单源真相比什么都重要。
2. **外部域必须与训练域物理隔离。** 数据放 `data/external/<域>/`，**不在**
   `data/Datasets/` 下。理由：`build_dataloaders` 会把配置里出现的每个数据集
   都按 `root` 找一遍，一旦外部域落进 `data/Datasets/`，正在排队的消融臂就会被
   悄悄多喂一个数据集 —— 各臂 val/test 口径不再一致，整张消融表作废。
3. **样本数硬校验。** `--expect 191` 对不上就**报错退出**，绝不静默产出一个
   「看起来正常」的小样本数。`TamperDataset` 在掩码配不上时会跳过样本，
   跳过多少它自己会打印，但汇总表不会体现 —— 所以这里再加一道闸。

用法
----
    python scripts/eval_external_domain.py --domain COVERAGE \
        --ckpt checkpoints/vibnet_best.pt --expect 191 \
        --out outputs/coverage_ext

    # 只看会加载到多少张、真假各多少，不跑推理
    python scripts/eval_external_domain.py --domain COVERAGE --list-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: 外部域登记表。键 = 目录名（= 报告里出现的名字），值 = 该域的元信息。
#: `kind` 用 datasets.py 认识的两种：tamper（有像素掩码）/ gensynth（只有真伪）。
EXTERNAL_DOMAINS = {
    "COVERAGE": {
        "kind": "tamper",
        "expect_total": 191,
        "expect_real": 100,
        "expect_fake": 91,
        "note": "复制-移动（copy-move）篡改，官方 100 对；"
                "与训练域（ForenSynths 生成图 / CASIAv2 拼接·复制移动）**不同源**，"
                "全程未参与训练 ⇒ 这是**零样本跨数据集**结果。",
    },
    # ---- 批次 1-B：四个新增评测域（2026-10-10 接入）--------------------------
    # 都是「只评测、不训练」；放置要求同 COVERAGE：data/external/<域>/test/{image,mask}。
    "DEFACTO": {
        "kind": "tamper",
        "expect_total": 8000,
        "expect_real": 4000,
        "expect_fake": 4000,
        "note": "DEFACTO（Kaggle defacto-test + defacto-test-pristine）："
                "4 类伪造（splicing / copy-move / removal / face-swap）混杂。"
                "★ 取它的第一条理由：本仓 `diag_area_buckets` 早先实测「篡改区 ≤2% 的图」"
                "mIoU 只有 0.046，而 DEFACTO 的篡改区占比集中在这个最弱的档 ⇒ "
                "**这是对定位短板最有针对性的一域**，不是随手多的一个数据集。"
                "真实图来自 COCO（`au_*`），带**全零掩码文件**（同 COVERAGE 的坑）。",
    },
    "IMD2020": {
        "kind": "tamper",
        "expect_total": 3500,
        "expect_real": 0,
        "expect_fake": 3500,
        "note": "IMD2020（HF KhangTruong/IMD2020 test 划分）：真实场景的**相机传感器级**"
                "篡改（splicing/removal/copy-move 等），CAT-Net(IJCV2022) 报过的 IML 基准。"
                "⚠ **没有真实图**（Au=0）⇒ 本域的 ACC / ACC@0.5 在语义上是"
                "「篡改召回率」，不是平衡准确率，**不得**与含真图的域并列比 ACC；"
                "跨域可比的是 AUC/AP（阈无关）与 miou_tampered_only。",
    },
    "CocoGlide": {
        "kind": "tamper",
        "expect_total": 1024,
        "expect_real": 512,
        "expect_fake": 512,
        "note": "CocoGlide（HF nebula/CocoGlide，TruFor 论文的扩散修复定位基准）："
                "**GLIDE 文本引导 inpainting** ⇒ 伪造机制是生成式的，与 CASIAv2/COVERAGE 的"
                "「手工拼接·复制移动」、IMD2020 的「传感器级」都不同源。真实图带全零掩码。",
    },
    "CASIA1.0": {
        "kind": "tamper",
        "expect_total": 920,
        "expect_real": 0,
        "expect_fake": 920,
        "note": "CASIA v1.0（从 IML-ViT.zip 内 images/CASIA1.0/{Tp,Gt} 抽取）："
                "ManTra-Net(CVPR2019)/SPAN(ECCV2020) 报过的**老基准**。"
                "⚠ 只有 Tp+Gt，**无 Au**（同 IMD2020 的注意事项）；"
                "另注意它与训练域 CASIA v2 同源（v1 是 v2 的子集系）⇒ "
                "**这不是零样本跨域**，是「同源早期版本」行，报数时必须标注，"
                "不得当作跨数据集泛化证据。",
    },
}


def build_cfg(base_config: str, domain: str, external_root: str) -> dict:
    """从主配置派生「只含该外部域」的评测配置（内存内覆盖，不落盘、不漂移）。"""
    from src.models.vibnet import load_config

    if domain not in EXTERNAL_DOMAINS:
        raise SystemExit(f"[ext] 未登记的外部域 {domain}；已登记：{list(EXTERNAL_DOMAINS)}")
    meta = EXTERNAL_DOMAINS[domain]

    cfg = load_config(base_config)
    entry = {"name": domain, "kind": meta["kind"], "split": "test", "max_samples": None}

    cfg["data"]["root"] = external_root
    # ★ 三个 split 全指向外部域：`build_dataloaders` 会把 train/val/test 三个 loader
    #   一次性建好，任一 split 为空就直接 RuntimeError（这是刻意的硬失败）。
    #   本脚本只用 test，另外两个建了不迭代；num_workers=0 避免凭空起一堆 worker。
    cfg["data"]["train_sets"] = [dict(entry)]
    cfg["data"]["val_sets"] = [dict(entry)]
    cfg["data"]["test_sets"] = [dict(entry)]
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="外部域零样本评测（独立基准行）")
    ap.add_argument("--domain", default="COVERAGE")
    ap.add_argument("--base-config", default="configs/default_crossval.yaml",
                    help="模型/评测段的唯一来源；本脚本只覆盖 data 段")
    ap.add_argument("--external-root", default="data/external",
                    help="外部域根目录（**必须**与训练域 data/Datasets 分离）")
    ap.add_argument("--ckpt", default="checkpoints/vibnet_best.pt")
    ap.add_argument("--device", default=None, help="默认 CPU（不干扰同机训练占卡）")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="CPU 推理线程数上限。0=不限。云端与训练共存时建议 8，"
                         "避免 torch 默认吃满所有核、拖慢正在跑的 GPU 训练。")
    ap.add_argument("--expect", type=int, default=None,
                    help="期望的评测样本数；对不上直接报错退出（防静默少样本）")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--out", default="outputs/coverage_ext")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ext_root = args.external_root
    if os.path.isabs(ext_root) is False:
        ext_root = os.path.join(ROOT, ext_root)
    data_dir = os.path.join(ext_root, args.domain)

    print("=" * 74)
    print(f" 外部域零样本评测  domain={args.domain}")
    print("=" * 74)
    print(f" 外部域目录：{os.path.relpath(data_dir, ROOT)}")
    if not os.path.isdir(data_dir):
        print(f"❌ 外部域目录不存在：{data_dir}")
        return 1
    if "Datasets" in os.path.relpath(data_dir, ROOT).split(os.sep):
        print("❌ 外部域**不能**放在 data/Datasets/ 下：那会被训练/消融链路按 root 扫到，"
              "把外部域混进 train/val/test，毁掉协议可比性。请改放到 data/external/。")
        return 1

    cfg = build_cfg(args.base_config, args.domain, ext_root)

    if args.list_only:
        from src.data.datasets import TamperDataset, GenSynthsDataset, IMG_EXT  # noqa: F401
        ds = (GenSynthsDataset(ext_root, args.domain, "test", 224, None, False)
              if EXTERNAL_DOMAINS[args.domain]["kind"] == "gensynth"
              else TamperDataset(ext_root, args.domain, "test", 224, None, False))
        n_pos = sum(1 for s in ds.samples if s["label"] == 1)
        print(f" 载入 {len(ds)} 条：篡改 {n_pos} / 真实 {len(ds) - n_pos}")
        return 0

    from src.evaluation.evaluate import evaluate_model, format_report

    if args.cpu_threads > 0:
        import torch as _t
        _t.set_num_threads(args.cpu_threads)
        print(f"[ext] torch CPU 线程数上限设为 {args.cpu_threads}")

    res = evaluate_model(cfg, args.ckpt, split="test", use_demo=False,
                         device=args.device or "cpu", max_batches=None,
                         save_dir=args.out, dataset=None)
    print(format_report(res))

    cls = res.get("cls") or {}
    n = int(cls.get("tp", 0) + cls.get("fp", 0) + cls.get("fn", 0) + cls.get("tn", 0))
    n_real = int(cls.get("tn", 0) + cls.get("fp", 0))
    n_fake = int(cls.get("tp", 0) + cls.get("fn", 0))

    meta = EXTERNAL_DOMAINS[args.domain]
    expect = args.expect if args.expect is not None else meta.get("expect_total")
    bad = []
    if expect and n != expect:
        bad.append(f"样本数 {n} ≠ 期望 {expect}")
    if meta.get("expect_real") and n_real != meta["expect_real"]:
        bad.append(f"真实样本 {n_real} ≠ 期望 {meta['expect_real']}")
    if meta.get("expect_fake") and n_fake != meta["expect_fake"]:
        bad.append(f"篡改样本 {n_fake} ≠ 期望 {meta['expect_fake']}")
    loc = res.get("loc") or {}
    n_loc = res.get("n_loc_samples") or 0
    # ⚠ 两个数不是一回事，别混：
    #   `n_loc_samples` = 被标了 mask_valid 的样本（COVERAGE 的 100 张真实图**也有**
    #     一个全零掩码文件 ⇒ 也算"有效"）⇒ 这里 = 191。
    #   `loc.n_tampered` = GT 真有前景的图（`gt.sum()>0` 才计入）⇒ 这里 = 91。
    #   只有后者等于篡改图数；`miou_tampered_only` 用的就是后者。用错了会把
    #   一个本来就正确的结果判成失败（这次就踩了）。
    n_tamp = int((loc or {}).get("n_tampered") or 0)
    if n_loc and n_loc != n:
        bad.append(f"定位有效样本 {n_loc} ≠ 分类样本总数 {n}")
    # ★ 2026-10-10：这里原来写的是 `n_tamp != expect_fake` ⇒ 直接判失败。
    #   实测 DEFACTO 会差 6 张（改 transform 前）/ 443 张（改之后），原因**不是**
    #   数据错：篡改区极小的图，短边缩到 224 后区域整块消失（INTER_NEAREST 采不到点），
    #   `n_tampered` 统计的是「GT 真有前景」的图，于是必然**少于**篡改图数。
    #   所以改成分两个方向判：
    #     n_tamp > expect_fake  →  不可能（比篡改图还多）⇒ 报错
    #     n_tamp < expect_fake  →  可解释的短缺 ⇒ 打印差额与原因，不报错
    size_hint = cfg["data"].get("image_size", 224)
    if meta.get("expect_fake"):
        if n_tamp > meta["expect_fake"]:
            bad.append(f"含真实前景的定位样本 {n_tamp} **多于**该域篡改图数 "
                       f"{meta['expect_fake']}——这是不可能的，说明计数或数据有问题")
        elif n_tamp < meta["expect_fake"]:
            gap = meta["expect_fake"] - n_tamp
            print(f"ℹ 含真实前景的定位样本 {n_tamp} < 该域篡改图数 {meta['expect_fake']}，"
                  f"差 {gap} 张")
            print(f"  原因：这 {gap} 张图的篡改区在缩到 {size_hint}×{size_hint} 后消失"
                  f"（区域太小，INTER_NEAREST 采不到点）。不是数据错误。")
            print(f"  影响：它们的逐图 IoU 不计入 miou_tampered_only ⇒ 该口径的分母"
                  f"是 {n_tamp} 而非 {meta['expect_fake']}，报数时必须写明。")

    loc = res.get("loc") or {}
    row = {
        "domain": args.domain,
        "ckpt": args.ckpt,
        "device": res.get("device"),
        "n_total": n, "n_real": n_real, "n_fake": n_fake, "n_loc_samples": n_loc,
        "n_loc_tampered": n_tamp,
        "cls": cls, "loc": loc,
        "zero_shot": True,
        "note": meta.get("note", ""),
        # ★ 口径红旗：COVERAGE 的 100 张真实图带**全零掩码文件** ⇒ 它们被标了
        #   mask_valid=True 并进入定位池；而 CASIAv2 的真实图没有掩码文件、不进池。
        #   所以 `miou`(池化) / `miou_per_sample` / dice / pixel_acc **跨数据集不可比**；
        #   只有 `miou_tampered_only` 两边都是"仅 GT 有前景的图"，才可比。
        "loc_pool_warning": (
            f"定位池含 {n_loc - n_tamp} 张全零 GT 的真实图（掩码文件存在但无前景）。"
            "只有 miou_tampered_only 跨数据集可比；其余三个含真实图像素，"
            "受误报影响，不可与 CASIAv2 的同名指标并列。"
        ),
        "cmd": "python " + " ".join([os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]),
    }

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    js_path = os.path.join(ROOT, args.out, f"external_{args.domain}{suffix}.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2, default=float)

    def g5(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "—"

    # 真实图召回（真判真率）与篡改图召回：cls 汇总口径里 `recall` = 正类(篡改)召回，
    # 真实图那一侧必须自己从混淆矩阵算，别拿 recall 冒充（会正好反了）。
    real_recall = (cls.get("tn", 0) / (cls.get("tn", 0) + cls.get("fp", 0))) if n_real else float("nan")

    md = [
        f"# 外部域独立基准行：{args.domain}（零样本，未参与训练）",
        "",
        f"- 权重：`{args.ckpt}`　设备：`{res.get('device')}`",
        f"- 样本：{n_real} 真实 + {n_fake} 篡改 = **{n}** 张"
        f"（定位池 {n_loc} 张，其中 GT 真有前景 {n_tamp} 张）",
        f"- 口径来源：`{args.base_config}` 的模型/评测段；仅 `data.root` 换成"
        f"`{os.path.relpath(data_dir, ROOT)}`",
        f"- ⚠ {meta.get('note', '')}",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 真伪 ACC | **{g5(cls.get('acc'))}** |",
        f"| 真伪 AUC | **{g5(cls.get('auc'))}** |",
        f"| AP | {g5(cls.get('ap'))} |",
        f"| F1 | {g5(cls.get('f1'))} |",
        f"| 真实图召回（真判真） | {g5(real_recall)} |",
        f"| 篡改图召回（假判假） | {g5(cls.get('recall'))} |",
        f"| 定位 mIoU（池化，宽松） | {g5(loc.get('miou'))} |",
        f"| 定位 mIoU（仅篡改图，★文献标准） | **{g5(loc.get('miou_tampered_only'))}** |",
        f"| 定位 mIoU（全部非空图，严格） | {g5(loc.get('miou_per_sample'))} |",
        f"| **像素级 AUC**（阈无关，ManTra-Net/SPAN 第一位指标） | "
        f"**{g5(loc.get('pixel_auc'))}** |",
        f"| Dice | {g5(loc.get('dice'))} |",
        f"| Pixel Acc | {g5(loc.get('pixel_acc'))} |",
        "",
        "> 复现命令：`" + row["cmd"] + "`",
        "",
        "> ⚠️ **口径红旗**：" + row["loc_pool_warning"],
        "> 报数请只报 `miou_tampered_only`（= 上表★行），并注明它是"
        f"{n_tamp} 张篡改图上的逐图 IoU 平均。",
    ]
    md_path = os.path.join(ROOT, args.out, f"external_{args.domain}{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print()
    print("\n".join(md))
    print(f"\n已写出：\n  {os.path.relpath(md_path, ROOT)}\n  {os.path.relpath(js_path, ROOT)}")

    if bad:
        print("\n❌ 样本构成校验未通过（**不要**采信上面的指标）：")
        for b in bad:
            print(f"   - {b}")
        return 1
    print("\n✅ 样本构成校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
