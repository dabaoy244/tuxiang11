"""三阶段分任务训练引擎（申报书 1.4(3)）

阶段划分（对应 configs/default.yaml -> train.stages）：
    ① 分类预训练  0-20  epoch  冻结定位支路；AdamW lr=1e-4 余弦退火 -> 1e-6
    ② 定位预训练 20-40  epoch  冻结分类支路；AdamW lr=5e-5 余弦退火 -> 5e-7
    ③ 联合微调   40-70  epoch  全部解冻；AdamW lr=1e-5 余弦退火 -> 1e-7

同时落实申报书的稳定训练三件套：
    * beta 退火（公式 24）—— 由 `beta_schedule(global_epoch)` 计算，阶段间可续接
    * KL 散度裁剪（公式 25）—— 在 models/vib.py 内完成
    * 梯度裁剪 L2 <= 5.0 —— 本文件的 `clip_grad_norm_`
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..losses.multi_task import MultiTaskLoss
from ..models.vib import beta_schedule
from .metrics import ClassificationMetrics, LocalizationMetrics


# ==========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class StageConfig:
    name: str
    epochs: int
    lr: float
    lr_min: float
    weight_decay: float = 1e-5
    freeze: List[str] = field(default_factory=list)
    train: List[str] = field(default_factory=list)
    use_tasks: List[str] = field(default_factory=lambda: ["cls", "loc", "edge"])
    beta_epoch_offset: int = 0
    # 本阶段早停所依据的验证指标；None → 用 train.monitor 的全局值。
    # 必须能分阶段指定：stage2 只训定位（分类头冻结），用 val_cls_auc 早停会
    # 在 8 轮后误判"没提升"而砍掉本阶段。2026-09-25 云上实测踩到。
    monitor: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "StageConfig":
        return StageConfig(
            name=d["name"], epochs=d["epochs"], lr=d["lr"],
            lr_min=d.get("lr_min", d["lr"] * 0.01),
            weight_decay=d.get("weight_decay", 1e-5),
            freeze=d.get("freeze", []), train=d.get("train", ["all"]),
            use_tasks=d.get("use_tasks", ["cls", "loc", "edge"]),
            beta_epoch_offset=d.get("beta_epoch_offset", 0),
            monitor=d.get("monitor"),
        )


# ==========================================================================
class Logger:
    """同时输出到控制台和 log.txt。"""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")
        else:
            self.fh = None

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()


# ==========================================================================
def optimization_parameters(model: nn.Module, criterion: nn.Module):
    """收集「必须交给优化器」的参数，返回 (参数列表, 额外纳入的 criterion 参数组数)。

    为什么单独抽出来
    ----------------
    不确定性加权的 s_i = log(σ²) 是 **criterion 里的 nn.Parameter**，不在 model 里。
    如果优化器只收 `model.trainable_parameters()`，这些 s_i 永远不会更新，
    权重就恒等于初始值 0.5*exp(0)=0.5 ——「可学习不确定性加权」静默退化成固定等权，
    而且训练照跑、日志照打，不报任何错。

    症状识别：权重文件/日志里 `w_* = 0.5`、`sigma_* = 1.0` 从头到尾不变。

    抽成模块级函数是为了让 Trainer 和 smoke_test 用同一份逻辑，
    避免两边各写一遍后再次出现"测试绿的、训练错的"。
    """
    params = list(model.trainable_parameters())
    n_extra = 0
    uw = getattr(criterion, "weighting", None)
    if uw is not None and getattr(criterion, "use_uncertainty", False):
        extra = [p for p in uw.parameters() if p.requires_grad]
        params += extra
        n_extra = len(extra)
    return params, n_extra


# ==========================================================================
def resolve_amp(tcfg: dict, device_type: str):
    """决定混合精度怎么开。返回 ``(enabled, amp_dtype, need_grad_scaler)``。

    抽成模块级函数，是为了让本机（无 GPU）也能验证这张真值表 —— 否则
    "上云第一次启用 AMP"就成了它唯一一次测试，而 AMP 配错会静默地毁掉
    一整轮云端训练（几十卡时）。

    两条规则，都是踩出来的：

    1) **bf16 不需要 GradScaler。** bf16 的指数位与 fp32 相同（都是 8 位），
       动态范围不会溢出，loss scaling 纯属多余；开着它只会多一次 scale/unscale、
       触发 UserWarning，并给出"已防溢出"的假安全感。历史实现不区分 dtype
       一律建 GradScaler —— 等于"看起来在防溢出、其实没防"（见 docs/08 D35）。

    2) **只在 cuda 上启用。** torch 确实支持 CPU 上的 bf16 autocast，但本机
       （无 AMX/VNNI 的普通 x86）实测比 fp32 **慢 30 倍以上**：同一个 demo 任务
       fp32 约 25 s/epoch，开了 amp 后 15 分钟连一个 epoch 都跑不完。
       这种"能跑但会把人坑死"的配置，不如显式关掉并告知。
    """
    want = bool(tcfg.get("amp", False))
    _d = str(tcfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.float16 if _d in ("float16", "fp16", "half") else torch.bfloat16
    enabled = want and device_type == "cuda"
    need_scaler = enabled and amp_dtype is torch.float16
    return enabled, amp_dtype, need_scaler


# ==========================================================================
class Trainer:
    def __init__(self, cfg: dict, model: nn.Module, device: torch.device,
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 logger: Optional[Logger] = None, ckpt_prefix: str = "vibnet"):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_prefix = ckpt_prefix

        tcfg = cfg["train"]
        self.output_dir = cfg["project"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(cfg["project"]["ckpt_dir"], exist_ok=True)
        self.log = logger or Logger(os.path.join(self.output_dir, "train_log.txt"))

        self.criterion = MultiTaskLoss(cfg).to(device)
        self.stages = [StageConfig.from_dict(s) for s in tcfg["stages"]]
        self.grad_clip = tcfg.get("grad_clip_norm", 5.0)
        self.amp, self.amp_dtype, _need_scaler = resolve_amp(tcfg, device.type)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True) if _need_scaler else None
        if tcfg.get("amp") and not self.amp:
            self.log("    [amp] 配置里 amp=true，但当前设备是 cpu —— 已自动关闭。"
                     "CPU 上的 bf16 autocast 实测慢 30 倍以上（见 docs/08 C19）")
        self.monitor = tcfg.get("monitor", "val_total")
        self.early_stop_patience = tcfg.get("early_stop_patience", 8)
        self.threshold = cfg["eval"].get("threshold", 0.5)

        self.best_metric = -math.inf
        self.best_epoch = -1
        self.best_raw = float("nan")     # 监控指标的原始值（未取负）
        self.history: List[dict] = []

    # ==================================================================
    def _optimizer(self, stage: StageConfig) -> torch.optim.Optimizer:
        params, n_extra = optimization_parameters(self.model, self.criterion)
        if n_extra:
            self.log(f"    [opt] 已把不确定性加权参数 {n_extra} 组（log σ²）纳入优化器")
        if not params:
            raise RuntimeError(f"阶段 {stage.name} 没有可训练参数，请检查 freeze/train 配置")
        betas = tuple(self.cfg["train"].get("betas", [0.9, 0.999]))
        # ★ 优先用 fused AdamW（2026-09-27 实测，scripts/diag_seg.py）：
        #   默认实现走 `_single_tensor_adam`，**每个标量参数都做一次 `.item()`**
        #   —— 本模型实测 402 次/步（stage2），等于每步 402 个 device→host 同步点；
        #   fused 实现是 0 次。实测 step 耗时相当（7.7ms vs 7.2ms，本就不是瓶颈），
        #   但少 400 个同步点能减少对数据加载/H2D 拷贝流水线的干扰。
        #   不可用时（非 CUDA、旧版 torch、个别 dtype 组合）静默回退到默认实现，
        #   并明确打出用了哪条路径 —— 否则"这次是哪种优化器"只存在于命令行历史里。
        if self.device.type == "cuda":
            try:
                opt = torch.optim.AdamW(params, lr=stage.lr,
                                        weight_decay=stage.weight_decay,
                                        betas=betas, fused=True)
                self.log("    [opt] AdamW(fused=True)：每步 0 次 .item() 同步")
                return opt
            except (TypeError, RuntimeError, ValueError) as exc:  # noqa: BLE001
                self.log(f"    [opt] fused AdamW 不可用（{type(exc).__name__}: "
                         f"{exc}），回退默认实现（每步约 400 次 .item() 同步）")
        return torch.optim.AdamW(
            params, lr=stage.lr, weight_decay=stage.weight_decay,
            betas=betas,
        )

    def _scheduler(self, opt: torch.optim.Optimizer, stage: StageConfig):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, stage.epochs), eta_min=stage.lr_min
        )

    # ==================================================================
    def fit(self, start_epoch: int = 0, start_stage: int = 0) -> List[dict]:
        """三阶段训练主循环。

        `start_stage` 用于**跳过已完成的阶段**（配合 `--resume` 的权重）。
        为什么需要它：--resume 只把权重读回来，训练仍会从 stage1 第 1 轮重新跑。
        stage1 分类预训练在 4090 上实测约 100 分钟（≈¥3），明明已经跑完并选出
        best.pt 却要重跑一遍，是纯粹的浪费；而 global_epoch 又不能简单归零，
        否则 beta 退火（公式 24，t≥20 才开始升温）会被重新按 t=0 计算，
        阶段三的 β 就永远起不来。所以这里按被跳过阶段的轮数把 global_epoch 顶上。
        """
        set_seed(self.cfg["project"].get("seed", 3407))
        start_stage = max(0, min(int(start_stage), len(self.stages) - 1))
        global_epoch = start_epoch
        if start_stage:
            skipped = self.stages[:start_stage]
            n_skip = sum(s.epochs for s in skipped)
            global_epoch += n_skip
            self.log(f"[resume] 跳过已完成的 {len(skipped)} 个阶段 "
                     f"{[s.name for s in skipped]}（共 {n_skip} 轮），"
                     f"从「{self.stages[start_stage].name}」开始；"
                     f"global_epoch={global_epoch}（保证 beta 退火接得上）")
        vcfg = self.cfg["model"]["vib"]

        # ★ β 退火的"锚"（2026-09-29 修，见 docs/22 §2.2(b)）
        #   原实现把 β 锚在 global_epoch 上，于是"升温的 20 轮"（t=20..39）
        #   恰好整个落在 stage2 —— 而 stage2 是 use_tasks=["loc","edge"]、
        #   freeze=["vib","cls_head"]，**VIB 既不进损失、参数也被冻结**。
        #   结果"退火"实跑成 stage1→stage3 之间的一次 0→0.1 阶跃，而且
        #   发生在解冻那一瞬间（正是退火本来要避免的情形）。
        #   现在默认锚在 vib_epoch（只在"VIB 真的进损失"的轮次上计步）。
        beta_anchor = str(vcfg.get("beta_anchor", "vib")).lower()
        if beta_anchor not in ("vib", "global"):
            raise ValueError(f"model.vib.beta_anchor 只能是 'vib' 或 'global'，收到 {beta_anchor!r}")
        vib_epoch = 0
        self._check_beta_window(vcfg, beta_anchor)

        for si, stage in enumerate(self.stages):
            if si < start_stage:
                continue
            # 早停依据的指标**按阶段解析**：stage2 只训定位（分类头冻结），
            # 若沿用分类指标会连续 8 轮"未提升"而被误砍。见 StageConfig.monitor。
            mon = stage.monitor or self.monitor
            self.log(f"{'='*70}")
            self.log(f"阶段 {si+1}/{len(self.stages)}: {stage.name}  "
                     f"epochs={stage.epochs}  lr={stage.lr:.1e}->{stage.lr_min:.1e}  "
                     f"监控={mon}")
            self.model.apply_stage(stage.freeze, stage.train)
            info = self.model.summary()
            self.log(f"  可训练参数 {info['trainable(M)']:.2f}M / 总计 {info['total(M)']:.2f}M"
                     f" | 冻结={stage.freeze} | 训练={stage.train} | 任务={stage.use_tasks}")

            opt = self._optimizer(stage)
            sched = self._scheduler(opt, stage)
            patience = 0
            # 各阶段监控的指标不同，分数不可跨阶段比较 → 本阶段重新起算
            self.best_metric = -math.inf
            self.best_epoch = -1
            self.best_raw = float("nan")
            stopped_early = False

            for ep in range(stage.epochs):
                t0 = time.time()
                # anchor="vib"   → 只在 VIB 参与损失（use_tasks 含 cls）的轮次上推进计步
                # anchor="global"→ 旧行为（锚在全局轮次），保留以便复现 09-28 那次训练
                anchor = vib_epoch if beta_anchor == "vib" else global_epoch
                beta = beta_schedule(
                    anchor + stage.beta_epoch_offset,
                    vcfg.get("beta_max", 0.1),
                    vcfg.get("beta_warmup_start", 20),
                    vcfg.get("beta_warmup_end", 40),
                )
                if "cls" in stage.use_tasks:
                    vib_epoch += 1
                tr = self._train_one_epoch(opt, stage, beta, global_epoch)
                sched.step()
                lr_now = opt.param_groups[0]["lr"]

                if tr["n_batches"] == 0:
                    # 整个 epoch 一个有效批次都没有 → 本阶段是**空转**，
                    # 权重一步都不会动，但日志会照打、显存照占、钱照烧。
                    # 必须立刻硬失败：正常情形（掩码样本占比 ~13%）空批次只占 10%，
                    # 100% 全空只可能是"带掩码的篡改数据集根本没加载进来"。
                    msg = (f"阶段「{stage.name}」整个 epoch 没有任何一个批次含有效监督："
                           f"本阶段启用任务={stage.use_tasks}，其中 loc/edge 的监督"
                           f"全部来自像素掩码，而这一轮 {tr['empty_batches']} 个批次全无掩码。"
                           f"请先核对 DataLoader 打印的 "
                           f"\"[data] split=train 共 N 条 ← ...\"："
                           f"CASIAv2 / COVERAGE 是否为 0 条（即为根因），"
                           f"再检查 data/Datasets 下的目录名与布局。")
                    self.log(f"    [x] {msg}")
                    raise RuntimeError(msg)
                if tr["empty_batches"]:
                    self.log(f"    [!] 本 epoch {tr['empty_batches']} 个批次无可用监督"
                             f"（整批都不带掩码）已跳过；有效批次 {tr['n_batches']}"
                             f"/{tr['n_batches'] + tr['empty_batches']}")

                msg = (f"[{stage.name}] epoch {ep+1}/{stage.epochs} (global {global_epoch}) "
                       f"lr={lr_now:.2e} beta={beta:.3f} loss={tr['loss']:.4f} "
                       f"raw={ {k: round(v,4) for k,v in tr['raw'].items()} } "
                       f"kl_raw={tr['diag'].get('kl_raw', float('nan')):.4f} "
                       f"kl_clip={tr['diag'].get('kl_clipped', float('nan')):.4f} "
                       f"grad_norm={tr['grad_norm']:.1f} "
                       f"({time.time()-t0:.1f}s)")
                self.log(msg)

                record = {"stage": stage.name, "epoch": global_epoch, "lr": lr_now,
                          "beta": beta, **{f"train_{k}": v for k, v in tr.items()
                                           if k not in ("raw", "normed", "weights")},
                          "train_raw": tr["raw"], "train_weights": tr["weights"]}

                if self.val_loader is not None and (global_epoch + 1) % self.cfg["train"].get("eval_every", 1) == 0:
                    val = self.validate(beta)
                    record.update({f"val_{k}": v for k, v in val.items()
                                   if not isinstance(v, dict)})
                    record["val_detail"] = {k: v for k, v in val.items() if isinstance(v, dict)}
                    self.log(f"    val: " + "  ".join(
                        f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, (int, float))))
                    metric = self._monitor_value(record, mon)
                    if metric > self.best_metric:
                        self.best_metric, self.best_epoch = metric, global_epoch
                        patience = 0
                        self.save("best", global_epoch, stage.name, record)
                        # 打印原始值（metric 对损失类取过负号，直接打印会误导）
                        raw = record.get(self._monitor_key(mon), metric)
                        self.best_raw = float(raw)
                        # 2026-09-29：原来这里用 '✔'，中文控制台字体没这个字形，
                        # 截图里渲染成方框（取证材料上很难看）→ 统一改 ASCII 标记。
                        self.log(f"    [+] 新的最佳 {mon}={self.best_raw:.4f}，"
                                 f"已保存 best.pt")
                    else:
                        patience += 1
                        if patience >= self.early_stop_patience:
                            # 只结束**本阶段**：早停是"这一段的指标到顶了"，
                            # 不是"整个训练该停了"。早期版本这里是 return，
                            # 会在 stage1 就把 stage2/3 一起带走（2026-09-25 云上踩到，
                            # 表现为训练 1 小时就结束、随后 AUTOSHUTDOWN 关机）。
                            self.log(f"    早停：{mon} 连续 {self.early_stop_patience} 轮未提升"
                                     f" → 结束本阶段「{stage.name}」")
                            self.history.append(record)
                            self.save("last", global_epoch, stage.name, record)
                            stopped_early = True
                            # 已消耗一轮，必须推进全局计数：否则下一阶段的第 1 轮
                            # 会复用同一个 epoch 号（history.json 出现重复 epoch，
                            # epochN 快照还可能互相覆盖）
                            global_epoch += 1
                            break

                self.history.append(record)
                if (global_epoch + 1) % self.cfg["train"].get("save_every", 5) == 0:
                    self.save(f"epoch{global_epoch+1}", global_epoch, stage.name, record)

                global_epoch += 1

            if stopped_early:
                self.log(f"    本阶段提前结束，继续下一阶段；"
                         f"最终 best.pt 由最后阶段的 {self.stages[-1].monitor or self.monitor} 选出。")
            elif si < len(self.stages) - 1:
                # 阶段正常结束时落一次 last.pt：中途异常退出也能断点续训
                self.save("last", global_epoch - 1 if global_epoch else 0,
                          stage.name, self.history[-1] if self.history else {})

        self.save("last", global_epoch - 1 if global_epoch else 0,
                  self.stages[-1].name, self.history[-1] if self.history else {})
        self._dump_history()
        self.log.close()
        return self.history

    # ==================================================================
    #: 简写 -> 验证记录里的真实键名
    _MONITOR_ALIASES = {
        "val_acc": "val_cls_acc",
        "val_miou": "val_loc_miou",
        "val_total": "val_loss",      # 历史写法：验证损失
    }

    def _monitor_key(self, name: Optional[str] = None) -> str:
        """把 monitor 的写法解析成验证记录里的扁平键名。"""
        n = str(self.monitor if name is None else name)
        return self._MONITOR_ALIASES.get(n, n)

    def _monitor_value(self, record: dict, name: Optional[str] = None) -> float:
        """监控指标统一成「越大越好」的分数。

        ⚠ 这里**不能**对不认识的键名静默回退到 -val_loss：那样配置里写
        monitor: val_cls_acc 实际却在按验证损失选最优权重，而且日志还照抄
        val_cls_acc 的名字，属于最贵的一类错误（以为在优化 A，其实在优化 B）。
        所以键名解析不出来时直接抛错，并列出可用键供修正。
        """
        key = self._monitor_key(name)
        if key not in record:
            candidates = sorted(k for k, v in record.items()
                                if isinstance(v, (int, float)))
            hint = ""
            if key.startswith("val_loc_"):
                # 最常见的原因不是配置写错，而是"这一轮验证集里没有带掩码的样本"
                # —— validate() 只在 has_loc_pred 时才产出 loc_* 指标。
                # 冒烟跑（--limit-batches）时把验证集截得太小就会这样。
                hint = ("\n  ⚠ 定位类指标缺失最常见的原因是：本轮验证集中**没有一张"
                        "带掩码的样本**（validate() 只在出现掩码样本时才计算 loc_*）。"
                        "若你在跑冒烟测试（--limit-batches），请调大该值或不要截断验证集。")
            raise KeyError(
                f"monitor='{self.monitor if name is None else name}' 解析为 '{key}'，"
                f"但验证记录里没有该指标。可用键：{candidates}{hint}")
        v = float(record[key])
        return -v if "loss" in key else v        # 损失类越小越好 → 取负

    # ==================================================================
    def _check_beta_window(self, vcfg: dict, beta_anchor: str) -> None:
        """启动自检：β 的升温区间必须与「VIB 真的进损失」的轮次有交集。

        把 docs/22 §2.2(b) 那个失效模式变成**硬失败**的防线。
        原实现里 β 升满 20 轮，而 VIB 全程被冻结且不进损失 ——
        配置合法（beta_max / beta_warmup_start / beta_warmup_end 都写了）、
        日志里 β 轨迹也漂亮，但退火对任何东西都没有影响。
        "配置合法、日志好看、语义为空"这类错必须启动即报错，不能等答辩时被问穿。
        """
        beta_max = float(vcfg.get("beta_max", 0.1))
        warm_s = float(vcfg.get("beta_warmup_start", 20))
        warm_e = float(vcfg.get("beta_warmup_end", 40))
        if warm_e <= warm_s:
            raise ValueError(f"beta_warmup_end({warm_e}) 必须大于 beta_warmup_start({warm_s})")

        # 枚举每个全局轮次：它的锚值、以及该轮 VIB 是否参与损失
        rows = []          # (global_epoch, stage_name, anchor, cls_on)
        vib_ep = 0
        for stage in self.stages:
            for _ in range(stage.epochs):
                cls_on = "cls" in stage.use_tasks
                anchor = (vib_ep if beta_anchor == "vib" else len(rows)) + stage.beta_epoch_offset
                rows.append((len(rows), stage.name, anchor, cls_on))
                if cls_on:
                    vib_ep += 1

        # 升温 = β 严格介于 0 与 beta_max 之间（在 warm_s / warm_e 两端是平的）
        warming = [r for r in rows if warm_s < r[2] < warm_e]
        max_anchor = max(r[2] for r in rows)
        if not warming:
            if max_anchor < warm_s:
                # 锚值根本没走到 warmup 就结束了 —— 这是"退火**未启用**"，
                # 与"退火被跳过 / 是空操作"是两回事。demo / --epochs-scale 0.05
                # 这类短跑天然如此，只警告、不失败（否则冒烟测试全被打死）。
                self.log(f"    [beta] [!] 锚={beta_anchor}，本轮最大锚值 {max_anchor:g} < "
                         f"warmup_start {warm_s:g} ⇒ β 全程恒为 0，退火未启用"
                         f"（短跑 / 小 epochs-scale 时正常；正式训练请核对）")
                return
            # 锚值已经越过整个升温区间，却没有任何一轮落在区间内 ⇒ 退火被跳过。
            # 这是 2026-09-29 实测踩到的：只把 beta_anchor 改成 'vib'、却留着
            # stage3.beta_epoch_offset=20，于是 stage3 的锚从 20 直接跳到 40
            # （= warm_e），β 一上来就是 0.1 —— 症状和原来一模一样。
            raise RuntimeError(
                f"β 退火被**跳过**：锚值越过了整个升温区间 [{warm_s:g},{warm_e:g})，"
                f"其中一轮都没落进去。\n"
                f"  锚模式      : {beta_anchor}\n"
                f"  锚值取值    : {sorted({r[2] for r in rows})[:8]} ... 最大 {max_anchor:g}\n"
                f"  典型原因    : stages[].beta_epoch_offset 过大 —— 它在 'global' 锚下是空操作，"
                f"在 'vib' 锚下却会把 stage3 的起始锚直接推过 warm_e。\n"
                f"  修法        : 把各 stage 的 beta_epoch_offset 置 0，并保持 beta_anchor='vib'。")
        overlap = [r for r in warming if r[3]]
        if not overlap:
            first = warming[0]
            raise RuntimeError(
                f"β 的升温区间与「VIB 参与损失」的轮次**没有交集** —— 退火对任何东西都不起作用。\n"
                f"  锚模式      : {beta_anchor}\n"
                f"  升温区间    : {warm_s:g} < anchor < {warm_e:g}（共 {len(warming)} 轮，"
                f"如 global {warming[0][0]}..{warming[-1][0]}，落在阶段 {warming[0][1]}）\n"
                f"  VIB 参与轮次: 共 {sum(1 for r in rows if r[3])} 轮"
                f"（阶段 {sorted({r[1] for r in rows if r[3]})}）\n"
                f"  → 升温那 {len(warming)} 轮里 VIB 不进损失（use_tasks 无 cls），"
                f"参数还可能被 freeze —— 这正是 09-28 那次训练的问题。\n"
                f"  两种修法：① model.vib.beta_anchor='vib'（默认，把锚移到 VIB 参与的轮次）；"
                f"② 调整 warmup 区间或 stages 轮数使其有交集。")

        self.log(f"    [beta] 锚={beta_anchor}  升温区间 {warm_s:g}<anchor<{warm_e:g} → "
                 f"global {warming[0][0]}~{warming[-1][0]}（{len(warming)} 轮）；"
                 f"与 VIB 参与轮次交集 {len(overlap)} 轮 [OK]")

    # ==================================================================
    def _train_one_epoch(self, opt, stage: StageConfig, beta: float, global_epoch: int) -> dict:
        self.model.train()
        # 冻结的 BN / Dropout 仍需正确的 train/eval 语义：冻结模块强制 eval
        for name in stage.freeze:
            mod = getattr(self.model, name, None)
            if isinstance(mod, nn.Module):
                mod.eval()

        total_loss = 0.0
        nb = 0
        n_empty = 0                      # 无任何可用监督项的批次（整批无掩码）
        raw_acc: Dict[str, float] = {}
        weight_last: Dict[str, float] = {}
        # ★ 可观测性（2026-09-29，docs/22 §2.2）：此前 train_log 全文搜 `kl` /
        #   `grad_norm` / `clip` 均 0 次命中 —— 三个稳定机制里只有 β 有日志，
        #   另外两个只能事后重算（每次重新加载 362MB 权重）。这里把
        #   ① KL 原始值/裁剪后值（来自 criterion 的 diag）
        #   ② 梯度总范数（clip_grad_norm_ 的返回值 = 裁剪**前**的范数）
        #   逐 epoch 平均后写进日志与 history.json。
        diag_acc: Dict[str, float] = {}
        grad_norm_acc = 0.0
        log_every = self.cfg["train"].get("log_every", 50)
        t0 = time.time()

        for it, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True) if batch["mask"] is not None else None
            mvalid = batch["mask_valid"].to(self.device, non_blocking=True) if "mask_valid" in batch else None

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp):
                out = self.model(images, sample_vib=True, beta=beta)
                losses = self.criterion(
                    out, labels, mask, beta=beta, active_tasks=stage.use_tasks,
                    update_norm=True, mask_valid=mvalid,
                )
                loss = losses["total"]

            # ★ 本批次没有任何启用中的监督项 → 整个 batch 对参数更新毫无贡献
            #   （典型：stage2/3 只训 loc/edge，监督全来自掩码，而这一批 16 张
            #    恰好全是 ForenSynths 无掩码样本，占比约 10%）。
            #   必须**在 backward 之前**跳过：counterfeit 一个 0 梯度 step
            #   不仅白算，还会把 zero-grad 当成一步 AdamW 更新推进动量。
            #   也不能计入 total_loss —— 否则 loss 曲线被 0 稀释成"看起来在变好"，
            #   正是 docs/08 反复出现的那类静默失真。
            if losses.get("empty"):
                n_empty += 1
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                # clip_grad_norm_ 的返回值 = 裁剪**前**的总范数。
                # 实测本项目为 89~419（阈值 5.0）⇒ 缩放比仅 0.012~0.038，
                # 即"每步都触发、但近乎空操作"。必须记下来才能支撑这句话。
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
            grad_norm_acc += float(gn)

            total_loss += float(loss.detach())
            nb += 1
            for k, v in losses["raw"].items():
                raw_acc[k] = raw_acc.get(k, 0.0) + v
            for k, v in losses.get("diag", {}).items():
                diag_acc[k] = diag_acc.get(k, 0.0) + float(v)
            weight_last = losses["weights"]

            if (it + 1) % log_every == 0:
                self.log(f"    iter {it+1}/{len(self.train_loader)}  "
                         f"loss={total_loss/nb:.4f}  {time.time()-t0:.1f}s")

        return {
            "loss": total_loss / max(1, nb),
            "raw": {k: v / max(1, nb) for k, v in raw_acc.items()},
            "weights": weight_last,
            "n_batches": nb,
            "empty_batches": n_empty,
            # 裁剪前梯度总范数（epoch 内平均）；无有效批次时记 0.0
            "grad_norm": grad_norm_acc / max(1, nb),
            # KL / β·KL 的 epoch 平均（该阶段未启用 cls 时为空 dict）
            "diag": {k: v / max(1, nb) for k, v in diag_acc.items()},
        }

    # ==================================================================
    @torch.no_grad()
    def validate(self, beta: float = 0.0) -> dict:
        self.model.eval()
        cls_m = ClassificationMetrics()
        loc_m = LocalizationMetrics(self.threshold)
        crit = MultiTaskLoss(self.cfg)
        crit.normalizer = self.criterion.normalizer       # 复用训练统计量
        crit = crit.to(self.device)
        total, nb = 0.0, 0
        has_loc_pred = False

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            mask = batch["mask"].to(self.device) if batch["mask"] is not None else None
            mvalid = batch["mask_valid"].to(self.device) if "mask_valid" in batch else None

            out = self.model(images, sample_vib=False, beta=beta)   # 推理用 mu（确定性）
            losses = crit(out, labels, mask, beta=beta, active_tasks=["cls", "loc", "edge"],
                          update_norm=False, mask_valid=mvalid)
            total += float(losses["total"])
            nb += 1

            cls_m.update(out["cls_logits"].float().cpu().numpy(), labels.cpu().numpy())
            if mask is not None and mvalid is not None and bool(mvalid.any()):
                idx = mvalid.cpu().numpy()
                loc_m.update(out["mask_prob"].float().cpu().numpy()[idx],
                             mask.cpu().numpy()[idx])
                has_loc_pred = True

        res = {"loss": total / max(1, nb)}
        cls_res = cls_m.compute(self.threshold)
        res.update({f"cls_{k}": v for k, v in cls_res.items() if isinstance(v, (int, float))})
        res["cls"] = cls_res
        res["val_loss"] = res["loss"]
        if has_loc_pred:
            loc_res = loc_m.compute()
            res.update({f"loc_{k}": v for k, v in loc_res.items()})
            res["loc"] = loc_res
        return res

    # ==================================================================
    def save(self, tag: str, epoch: int, stage: str, record: dict) -> str:
        path = os.path.join(self.cfg["project"]["ckpt_dir"], f"{self.ckpt_prefix}_{tag}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "loss_weights": self.criterion.state_dict(),
            "normalizer": self.criterion.normalizer.state_dict(),
            "epoch": epoch, "stage": stage, "record": record,
            "cfg": self.cfg,
        }, path)
        return path

    def load(self, path: str, strict: bool = False) -> dict:
        from ..models.spatial_branch import infer_backbone_variant

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # resume 时模型已经建好了，不能像 evaluate 那样自动改档位 ——
        # 所以这里显式比对档位，把 "size mismatch" 翻译成人能看懂的话。
        ck_variant = infer_backbone_variant(ckpt.get("model", {}))
        cur_variant = str(
            self.cfg.get("model", {}).get("spatial", {}).get("backbone_variant", "b16")
        ).lower()
        if ck_variant and ck_variant != cur_variant:
            raise RuntimeError(
                f"checkpoint 的骨干档位是 '{ck_variant}'，但当前配置是 '{cur_variant}'，"
                f"两者参数形状不兼容。\n"
                f"  解决方式一（推荐）：把 YAML 里的 backbone_variant 改成 '{ck_variant}' 后重跑；\n"
                f"  解决方式二：重新训练，或用 --resume 指向与当前配置同档位的 checkpoint。\n"
                f"  参见 docs/09_指标可达性与骨干选型.md"
            )

        try:
            missing = self.model.load_state_dict(ckpt["model"], strict=strict)
        except RuntimeError as e:
            raise RuntimeError(
                f"加载 {path} 失败：{e}\n"
                f"  常见原因：checkpoint 与 YAML 的骨干档位/融合维度不一致。"
                f"当前 backbone_variant='{cur_variant}'，"
                f"checkpoint 推断档位='{ck_variant or '未知'}'。"
            ) from e
        if "loss_weights" in ckpt:
            try:
                self.criterion.load_state_dict(ckpt["loss_weights"])
            except Exception:
                pass
        if "normalizer" in ckpt:
            self.criterion.normalizer.load_state_dict(ckpt["normalizer"])
        self.log(f"已加载权重 {path} (epoch={ckpt.get('epoch')}, stage={ckpt.get('stage')})")
        return ckpt

    def _dump_history(self) -> None:
        path = os.path.join(self.output_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2, default=str)
        self.log(f"训练历史已写入 {path}")
