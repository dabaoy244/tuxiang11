"""训练入口（三阶段分任务训练）

用法：
    # 用离线演示数据集快速跑通全流程（CPU 可跑，几分钟）
    python scripts/train.py --demo --epochs-scale 0.05

    # 正式训练（需要三套公开数据集，建议 GPU）
    python scripts/train.py --config configs/default.yaml

    # 从断点继续
    python scripts/train.py --resume checkpoints/vibnet_last.pt

    # 跳过已跑完的阶段（stage1 已训完 20 轮，直接接着训 stage2）
    python scripts/train.py --resume checkpoints/vibnet_best.pt --start-stage stage2_loc_pretrain

    # 速度探针：只跑很少的 batch，用 iter 行的耗时定位"哪个阶段慢"
    python scripts/train.py --epochs-scale 0.05 --limit-batches 20 --out-dir outputs/probe
    python scripts/train.py --epochs-scale 0.05 --limit-batches 20 --cudnn off      # 对照

常用参数：
    --epochs-scale  按比例缩放各阶段轮数（0.05 表示 70 轮 -> 约 4 轮，用于冒烟测试）
    --limit-batches 限制每个 epoch 的批次数（进一步加速自检 / 实测速度）
    --device        cpu / cuda
    --workers       DataLoader 进程数
    --amp           开启 bf16 混合精度（云端 GPU 必备；见 docs/18）
    --cudnn         on / off / bench —— 卷积后端（慢路径排查用）
    --amp-dtype     bfloat16 / float16 —— 覆盖 amp_dtype（慢路径排查用）
    --data-root     数据目录，须指向 <data>/Datasets（误传 <data> 会自动补一级）
    --out-dir       输出目录，让 probe 短跑不污染正式 run 的日志
    --start-stage   跳过前面的阶段，从该阶段开始（名字或序号）；配合 --resume 使用
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from src.data.datasets import build_dataloaders  # noqa: E402
from src.engine.trainer import Logger, Trainer, set_seed  # noqa: E402
from src.models.vibnet import build_model, load_config  # noqa: E402


def apply_runtime_flags(cfg: dict, cudnn: str = "on",
                        amp_dtype: str | None = None) -> None:
    """把 ``--cudnn`` / ``--amp-dtype`` 真正落到运行期状态与 cfg 上。

    单独成函数只为一件事：能被 ``scripts/smoke_test.py`` 直接 import 断言。
    教训：这两个开关曾经**只 add_argument、没有任何消费点** —— 命令行上写着
    ``--cudnn off``，实际根本没关 cuDNN，于是"两条对照探针耗时完全相同"，
    看上去像"cuDNN 慢路径假设被否定"，其实实验根本没做（docs/08 D42）。
    参数声明了却不生效，属于最坏的一类静默失真：它给出的结论是**反向**的。
    """
    if cudnn == "off":
        torch.backends.cudnn.enabled = False
    elif cudnn == "bench":
        torch.backends.cudnn.benchmark = True
    if amp_dtype:
        cfg.setdefault("train", {})["amp_dtype"] = amp_dtype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/default.yaml"))
    ap.add_argument("--demo", action="store_true", help="使用离线演示数据集")
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--tag", default="vibnet")
    ap.add_argument("--amp", action="store_true",
                    help="开启 bf16 混合精度（cloud GPU 上建议开；4090 上实测提速明显且省显存）")
    ap.add_argument("--no-amp", action="store_true", help="强制关闭混合精度")
    ap.add_argument("--data-root", default=None,
                    help="覆盖配置里的 data.root。它必须指向 <data>/Datasets"
                         "（ForenSynths / CASIAv2 的父目录），"
                         "如 /root/autodl-tmp/data/Datasets；"
                         "若误传 <data> 会自动补一级并提示")
    ap.add_argument("--out-dir", default=None,
                    help="覆盖配置里的 project.output_dir（让 probe 短跑不污染正式 run 的日志）")
    ap.add_argument("--start-stage", default=None,
                    help="跳过前面的阶段、从该阶段开始训练（写阶段名或 0 基序号），"
                         "配合 --resume 用：--resume checkpoints/vibnet_best.pt "
                         "--start-stage stage2_loc_pretrain。"
                         "注意 --resume 本身只加载权重，不推进阶段与轮次。")
    ap.add_argument("--cudnn", choices=["on", "off", "bench"], default="on",
                    help="cuDNN 卷积后端选择。on=默认；off=torch.backends.cudnn.enabled=False"
                         "（回退到 PyTorch 原生实现）；bench=benchmark=True（自动调优）。"
                         "日志里出现 CUDNN_STATUS_NOT_SUPPORTED 且单轮耗时异常（比同类阶段"
                         "慢一个数量级）时，用它做对照：同一份权重、同一批数据跑 20 个 batch，"
                         "比较 iter 行的耗时即可定位是不是反向卷积掉进了慢路径。")
    ap.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default=None,
                    help="覆盖配置里的 train.amp_dtype。bf16 在部分 cuDNN 版本上"
                         "卷积反向会回退到极慢路径，可用 float16 对照。")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.amp:
        cfg["train"]["amp"] = True
    if args.no_amp:
        cfg["train"]["amp"] = False
    # 卷积后端 / 精度覆盖：必须在这里就生效（模型还没建、任何卷积都还没跑）
    apply_runtime_flags(cfg, args.cudnn, args.amp_dtype)
    if args.data_root:
        # 只改"数据在哪"：GenSynthsDataset/TamperDataset 拿到的 root 就是这一项，
        # 它必须直指 <data>/Datasets。数据盘挂载点变了不必改 yaml（见 docs/18）。
        r = args.data_root
        # 容错：最常见的写法错误是把 <data>（Datasets 的上一级）传进来。
        # 这里按"目录里有没有 ForenSynths"自动判定，避免白跑一次才发现找不到数据。
        if (not os.path.isdir(os.path.join(r, "ForenSynths"))
                and os.path.isdir(os.path.join(r, "Datasets", "ForenSynths"))):
            fixed = os.path.join(r, "Datasets")
            print(f"[train] --data-root={r} 是数据根，已自动更正为 {fixed}")
            r = fixed
        cfg["data"]["root"] = r
    if args.out_dir:
        cfg["project"]["output_dir"] = args.out_dir
    if args.demo:
        cfg["project"]["output_dir"] = os.path.join(ROOT, "outputs", "run_demo")
        cfg["project"]["ckpt_dir"] = os.path.join(ROOT, "checkpoints")

    if args.epochs_scale != 1.0:
        for s in cfg["train"]["stages"]:
            s["epochs"] = max(1, int(round(s["epochs"] * args.epochs_scale)))
        cfg["train"]["log_every"] = 5
        cfg["train"]["save_every"] = 999
        print(f"[train] 轮数已按 {args.epochs_scale} 缩放 -> "
              f"{[s['epochs'] for s in cfg['train']['stages']]}")

    # ---- --start-stage：跳过已完成的阶段（配合 --resume 用）
    # 这一步必须放在 epochs-scale 之后：跳过时要累加的正是**实际**轮数，
    # 顺序反了会让 global_epoch 偏移，beta 退火（公式 24）跟着错。
    stage_names = [s["name"] for s in cfg["train"]["stages"]]
    start_stage = 0
    if args.start_stage is not None:
        raw = str(args.start_stage).strip()
        if raw.lstrip("-").isdigit():
            start_stage = int(raw)
        elif raw in stage_names:
            start_stage = stage_names.index(raw)
        else:
            raise SystemExit(
                f"[train] --start-stage={raw} 既不是序号也不是阶段名。"
                f"可选：{list(enumerate(stage_names))}")
        if not 0 <= start_stage < len(stage_names):
            raise SystemExit(
                f"[train] --start-stage={start_stage} 越界，本配置只有 "
                f"{len(stage_names)} 个阶段：{list(enumerate(stage_names))}")
        print(f"[train] --start-stage：跳过 {stage_names[:start_stage]}，"
              f"从 '{stage_names[start_stage]}' 开始"
              f"（请配合 --resume 指向该阶段之前的权重，否则是随机初始化从头训）")
        if not args.resume:
            print("[train] ⚠ 你用了 --start-stage 但没给 --resume：前面阶段的权重不会被加载，"
                  "当前是从随机初始化直接训该阶段。多数情况下这不是你想要的。")

    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    set_seed(cfg["project"].get("seed", 3407))

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] 设备：{dev}")
    # 把后端选择写进日志：否则"这次是用哪套卷积实现跑的"只存在于命令行历史里，
    # 事后对比两次 run 的耗时（例如查 cuDNN 慢路径）会无从下手。
    print(f"[train] cuDNN enabled={torch.backends.cudnn.enabled} "
          f"benchmark={torch.backends.cudnn.benchmark} ｜ "
          f"amp={cfg['train'].get('amp')} amp_dtype={cfg['train'].get('amp_dtype')}")

    loaders = build_dataloaders(cfg, use_demo=args.demo)
    print(f"[train] 训练集 {len(loaders['train'].dataset)} 张，"
          f"验证集 {len(loaders['val'].dataset)} 张")

    if args.limit_batches:
        from itertools import islice

        class _Limited:
            """只取前 n 批（冒烟用）。"""

            def __init__(self, loader, n):
                self.loader, self.n = loader, n

            def __iter__(self):
                return islice(iter(self.loader), self.n)

            def __len__(self):
                return min(len(self.loader), self.n)

        loaders["train"] = _Limited(loaders["train"], args.limit_batches)
        # ★ 故意**不**限制验证集（2026-09-27 修，云上冒烟时踩到）。
        #   曾经的写法是 `_Limited(val, max(1, limit//2))`，它在带掩码的 stage2 上
        #   必然崩，原因有两层，两层缺一不可：
        #     ① 验证集的顺序是 ForenSynths(val, 8000 张、无掩码) 在前、
        #        CASIAv2(val, 1261 张、带掩码) 在后，且 shuffle=False
        #        ⇒ **前几百批一张带掩码的样本都没有**（不是"运气不好"，是必然）；
        #     ② validate() 只在 has_loc_pred（至少有一批带掩码样本）时才产出
        #        loc_* 指标，于是 stage2 的 monitor=val_loc_miou 解析不到，
        #        trainer._monitor_value 按设计抛
        #            KeyError: monitor='val_loc_miou' 解析为 'val_loc_miou'，
        #                      但验证记录里没有该指标
        #        —— 这个报错本身是**故意**的：宁可报错，也不要静默换成别的指标
        #        去选最优权重（那属于"以为在优化 A、实际在优化 B"）。
        #   顺带：截断验证集还会让 best.pt 由 3% 的数据选出，指标本身就不可信。
        #   而验证集只有前向，跑满一轮约 35 s（3080 Ti），冒烟完全付得起。
        #   真要用极少批次快速验证，请配单阶段临时 yaml 把 monitor 改成 val_cls_auc。


    model = build_model(cfg)
    for k, v in model.summary().items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")

    logger = Logger(os.path.join(cfg["project"]["output_dir"], "train_log.txt"))
    trainer = Trainer(cfg, model, dev, loaders["train"], loaders["val"],
                      logger=logger, ckpt_prefix=args.tag)
    if args.resume and os.path.exists(args.resume):
        ckpt = trainer.load(args.resume)
        if args.start_stage is None:
            print(f"[train] 注意：--resume 只加载权重（该 ckpt 属于阶段 "
                  f"{ckpt.get('stage')}、epoch {ckpt.get('epoch')}），"
                  f"训练仍会从 '{stage_names[0]}' 第 1 轮重新开始 —— 已完成的阶段会被重跑。"
                  f"要接着下一阶段训，请加 --start-stage <阶段名>。")

    trainer.fit(start_stage=start_stage)
    print(f"\n[train] 完成。最佳 {cfg['train']['monitor']}="
          f"{trainer.best_raw:.4f} @ epoch {trainer.best_epoch}")
    print(f"[train] 权重目录：{cfg['project']['ckpt_dir']}")
    print(f"[train] 日志：{cfg['project']['output_dir']}/train_log.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
