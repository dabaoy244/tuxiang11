#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照实验驱动：**骨干随机初始化 vs ImageNet 预训练**（唯一变量 = 骨干权重）

为什么必须做这个实验
--------------------
项目早期所有训练都"训不动"：val acc 恒等于 0.5，AUC 游走在 0.49~0.52，
预测在两个极端之间来回翻（有时全判真 tp=0/fn=N，有时全判假 tp=N/fn=0）。
当时的怀疑是"骨干随机初始化 + 分类头太小，学不动"，
但怀疑不能写进论文/中期材料 —— 必须有**受控实验**支撑。

本脚本跑一对**除骨干权重外完全相同**的实验：

    A 臂 random     configs/ablation_b16_random.yaml     骨干随机初始化
    B 臂 pretrained configs/ablation_b16_pretrained.yaml 骨干加载 pretrained/vit_b16_imagenet.pt

两臂共用：同一份训练/验证数据、同一超参、同一随机种子(3407)、同一评测口径。
两配置文件由 `scripts/make_ablation_configs.py` 从同一模板生成，
过滤掉注释后 `diff` 只剩 3 处不同（`name` / `output_dir` / `backbone_weights`），
其中**只有 `backbone_weights` 影响训练** —— 另外两处只是实验标签与落盘路径。
这是"受控"的依据，报告里会连同复核命令一起打印出来。

数据取自 ForenSynths，**只能得出"可训练性"结论**
------------------------------------------------
ForenSynths 的 train 划分(progan_train.7z, 70GB)在受限网络下未取得，
本实验用 ProGAN 的 **val 划分**当训练集、**test 划分**当验证集（两者图像无交集）。
因此结论口径是："预训练权重决定模型能不能训起来"，
**不能**声称复现了 CNNDetection 的官方协议，也**不能**当作跨生成器泛化性能。

用法
----
    # 完整流程：两臂顺序训练 -> 逐生成器评测 -> 生成报告
    python scripts/run_ablation.py

    # 只跑其中一臂
    python scripts/run_ablation.py --arms pretrained

    # 不跑逐生成器评测（省时间）
    python scripts/run_ablation.py --skip-eval

    # 结果已存在，只重新汇总出报告
    python scripts/run_ablation.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --------------------------------------------------------------------------
# 两臂定义：唯一区别就是配置文件里的 backbone_weights 那一行
# --------------------------------------------------------------------------
ARMS = [
    {
        "key": "random",
        "title": "随机初始化",
        "config": "configs/ablation_b16_random.yaml",
        "out_dir": "outputs/ablation_random",
        "ckpt_tag": "abr",                  # -> checkpoints/abr_best.pt
        "backbone_weights": "（不加载）",
    },
    {
        "key": "pretrained",
        "title": "ImageNet 预训练",
        "config": "configs/ablation_b16_pretrained.yaml",
        "out_dir": "outputs/ablation_pretrained",
        "ckpt_tag": "abp",                  # -> checkpoints/abp_best.pt
        "backbone_weights": "pretrained/vit_b16_imagenet.pt",
    },
]
BY_KEY = {a["key"]: a for a in ARMS}

REPORT_PATH = os.path.join(ROOT, "docs", "14_对照实验_预训练权重消融.md")


# --------------------------------------------------------------------------
def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def log(msg: str) -> None:
    print(f"[ablation] {msg}", flush=True)


def run_streamed(cmd: list, log_path: str) -> int:
    """跑子进程并把输出实时写进日志文件（同时透传到本进程 stdout）。"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            line = line.rstrip()
            if line:
                print("    " + line, flush=True)
        return proc.wait()


def train_arm(arm: dict) -> bool:
    log(f"=== 训练 A/B 臂 [{arm['key']}] {arm['title']} ===")
    log(f"    配置：{arm['config']}")
    log(f"    骨干权重：{arm['backbone_weights']}")
    t0 = time.time()
    rc = run_streamed(
        [sys.executable, "-u", _p("scripts", "train.py"),
         "--config", _p(*arm["config"].split("/")),
         "--device", "cpu", "--tag", arm["ckpt_tag"]],
        _p(arm["out_dir"], "train_stdout.txt"))
    dt = (time.time() - t0) / 60
    log(f"    [{arm['key']}] 退出码 {rc}，耗时 {dt:.1f} 分钟")
    return rc == 0


def eval_arm(arm: dict, per_class: int, batch_size: int) -> bool:
    ckpt = _p("checkpoints", f"{arm['ckpt_tag']}_best.pt")
    if not os.path.exists(ckpt):
        log(f"    [{arm['key']}] 找不到 {ckpt}，跳过逐生成器评测")
        return False
    log(f"=== 逐生成器评测 [{arm['key']}] per_class={per_class} ===")
    scores_npz = _p("outputs", "cross_gen_scores", f"{arm['key']}.npz")
    rc = run_streamed(
        [sys.executable, "-u", _p("scripts", "eval_cross_generator.py"),
         "--config", _p(*arm["config"].split("/")),
         "--ckpt", ckpt,
         "--per-class", str(per_class),
         "--batch-size", str(batch_size),
         "--device", "cpu",
         "--out", _p("outputs", f"cross_gen_{arm['key']}"),
         "--tag", arm["key"],
         # ★ 把逐样本分数一并存盘：推理是本流程最贵的一步（约 9 分钟/臂），
         #   存下来以后改指标定义可以免推理重算（scripts/recompute_cross_gen.py）。
         "--scores-out", scores_npz],
        _p(arm["out_dir"], "cross_gen_stdout.txt"))
    log(f"    [{arm['key']}] 评测退出码 {rc}")
    return rc == 0


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
_EPOCH_RE = re.compile(
    r"epoch (\d+)/(\d+) \(global (\d+)\) lr=([0-9.eE+-]+) beta=([0-9.]+) "
    r"loss=(-?[0-9.]+|nan)")
# ⚠ 键名里**含数字**（`cls_f1`、`cls_mAP` 的 `mAP` 大小写、`epoch2` 之类），
#   所以字符类必须包含 0-9。早先写成 `[A-Za-z_]+`，结果 `cls_f1` 整个匹配不上、
#   被**静默丢弃** —— 报告里那一格显示 "—"，看起来像"没这项指标"，
#   实际是解析器漏了。这类"丢一个指标但不报错"的 bug 极难发现。
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9.]+|nan)")


def parse_train_log(path: str) -> list:
    """从 train_log.txt 解析逐 epoch 记录。

    为什么需要这个：`Trainer.save_history()` 只在**整个 fit() 结束时**才写
    `history.json`。所以"训练还在跑"或"训练中途崩了"这两种情况下，
    history.json 根本不存在 —— 而这两种情况恰恰最需要看到数字。
    训练日志是**边跑边刷**的，必须作为回退数据源。
    """
    recs = []
    cur = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _EPOCH_RE.search(line)
                if m:
                    cur = {"stage": "stage1_cls_pretrain",
                           "epoch": int(m.group(3)),
                           "epoch_in_stage": int(m.group(1)),
                           "lr": float(m.group(4)),
                           "beta": float(m.group(5)),
                           "train_loss": float(m.group(6))}
                    continue
                if cur is not None and "val:" in line:
                    kv = {k: float(v) for k, v in
                          _KV_RE.findall(line.split("val:", 1)[1])}
                    if "loss" in kv:
                        cur["val_loss"] = kv["loss"]
                    for k, v in kv.items():
                        if k.startswith("cls_"):
                            cur["val_" + k] = v
                    recs.append(cur)
                    cur = None
    except OSError:
        return []
    return recs


def load_arm_result(arm: dict) -> dict:
    """读取一臂的训练历史 + 逐生成器评测结果。"""
    res = {"arm": arm, "history": None, "cross_gen": None, "source": "无"}
    hp = _p(arm["out_dir"], "history.json")
    if os.path.exists(hp):
        try:
            with open(hp, encoding="utf-8") as f:
                res["history"] = json.load(f)
            res["source"] = "history.json（训练已完整结束）"
        except Exception as e:                              # noqa: BLE001
            log(f"    读取 {hp} 失败：{e}")
    if not res["history"]:
        recs = parse_train_log(_p(arm["out_dir"], "train_log.txt"))
        if recs:
            res["history"] = recs
            res["source"] = ("train_log.txt 实时解析"
                             "（history.json 尚未落盘 → 训练仍在进行，或曾中断）")
    # eval_cross_generator.py 落盘文件名：cross_generator{suffix}.json
    # suffix = f"_{tag}"，我们调用时传的 tag 就是臂名
    cg_dir = _p("outputs", f"cross_gen_{arm['key']}")
    for name in (f"cross_generator_{arm['key']}.json",
                 "cross_generator.json", "cross_gen.json", "report.json"):
        cp = os.path.join(cg_dir, name)
        if os.path.exists(cp):
            try:
                with open(cp, encoding="utf-8") as f:
                    res["cross_gen"] = json.load(f)
            except Exception as e:                          # noqa: BLE001
                log(f"    读取 {cp} 失败：{e}")
            break
    return res


def best_epoch(history: list) -> dict | None:
    """按 val_cls_auc 挑最佳 epoch（AUC 对类别不平衡不敏感，比 acc 更可靠）。"""
    if not history:
        return None
    key = "val_cls_auc"
    cand = [h for h in history if isinstance(h.get(key), (int, float))]
    if not cand:
        key = "val_cls_acc"
        cand = [h for h in history if isinstance(h.get(key), (int, float))]
    if not cand:
        return None
    return max(cand, key=lambda h: h[key])


def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def _i(v) -> str:
    """把混淆矩阵计数格式化成整数（训练日志里它们是浮点，如 618.0000）。"""
    if v is None:
        return "—"
    try:
        return str(int(round(float(v))))
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------
def find_cross_gen_rows(cg: dict) -> tuple:
    """从逐生成器评测结果里抽出 ([(生成器名, 指标dict)], overall dict)。

    eval_cross_generator.py 的落盘结构是：
        {"meta": {...}, "per_generator": {gen: {...}}, "overall": {...}}
    其中 overall 里既含全局指标(acc/auc/...)，也含各指标的宏平均(<k>_macro)。
    """
    if not isinstance(cg, dict):
        return [], {}
    per = None
    for k in ("per_generator", "per_generator_metrics", "per_gen", "generators"):
        if isinstance(cg.get(k), dict):
            per = cg[k]
            break
    if per is None:
        for k in ("rows", "results"):
            if isinstance(cg.get(k), list):
                per = {r.get("generator") or r.get("name"): r
                       for r in cg[k] if isinstance(r, dict)}
                break
    macro = {}
    if isinstance(cg.get("overall"), dict):
        macro = cg["overall"]
    else:
        for k in ("macro", "macro_avg", "macro_average", "summary"):
            if isinstance(cg.get(k), dict):
                macro = cg[k]
                break
    rows = []
    if isinstance(per, dict):
        for name, m in per.items():
            if isinstance(m, dict):
                rows.append((str(name), m))
    rows.sort(key=lambda t: t[0])
    return rows, macro


def pick(m: dict, *names, default=None):
    for n in names:
        if n in m and m[n] is not None:
            return m[n]
    return default


def effective_per_class(results: list, cli_value: int) -> int:
    """从各臂跨生成器 JSON 里读出**实际采样数**，而不是用命令行默认值。

    为什么必须这样：`build_report()` 原先把 CLI 的 `--per-class` 直接印进报告，
    于是 `--report-only`（不传该参数、走默认 120）重出报告时，
    报告会写"每生成器各取 120 张"，而数据其实是 100 张跑出来的 ——
    **报告里的数字和产出它的那次运行对不上**，正是本项目最忌讳的一类错。
    现在以 JSON 自带的 `meta.per_class` 为准（它是执行时的真值），
    各臂不一致时显式告警。
    """
    vals = []
    for r in results:
        cg = r.get("cross_gen") or {}
        v = (cg.get("meta") or {}).get("per_class")
        if isinstance(v, int) and v > 0:
            vals.append(v)
    if not vals:
        return cli_value
    uniq = sorted(set(vals))
    if len(uniq) > 1:
        log(f"    ⚠ 各臂的实际 per_class 不一致：{uniq}；"
            f"对照实验要求两臂采样一致，请检查是否重跑过单臂")
    if uniq and uniq[0] != cli_value:
        log(f"    报告按 JSON 记录的实际采样数 {uniq[0]} 生成"
            f"（命令行给的是 {cli_value}，已忽略）")
    return uniq[0]


def build_report(results: list, per_class: int) -> str:
    L = []
    A = L.append
    A("# 对照实验：骨干预训练权重消融（random vs ImageNet-pretrained）")
    A("")
    A("> 本文件由 `scripts/run_ablation.py` 自动生成，数字直接来自 "
      "`outputs/ablation_*/history.json` 与 `outputs/cross_gen_*/`。")
    A("")
    A("## 1. 实验动机")
    A("")
    A("本项目在拿到预训练权重之前，本机的小规模真实数据训练一直**训不出判决边界**。")
    A("为了不让这个现象被含糊地描述，先把它拆成两条**可证伪的判据**：")
    A("")
    A("| 判据 | 定义 | 说明 |")
    A("| --- | --- | --- |")
    A("| **退化**（degenerate） | 阈值判决下 `tn=0`（全判假）或 `tp=0`（全判真） | 判决边界不存在。此时 `acc` 必为 0.5 或类别占比，**ACC 完全不可用** |")
    A("| **有效**（non-degenerate） | 同一 epoch 内 `tn>0` **且** `tp>0` | 两类都有被正确判出的样本，边界已建立 |")
    A("")
    A("> 只看 AUC/ACC 会被误导：退化臂也可能有 `AUC>0.5`。本例中 b16 随机初始化臂")
    A("> 的 `AUC=0.63`，但其 `tn=0` —— 排序分数里有一点信号，阈值判决却完全塌到一侧。")
    A("> **所以判据必须是「同时看 AUC 与 min(tn,tp)」。**")
    A("")
    A("已有的两条观察（注意它们是**两个不同骨干**，不可混为一谈）：")
    A("")
    A("| 观察 | 骨干 | 现象 |")
    A("| --- | --- | --- |")
    A("| 本机 realval 训练（随机初始化） | **s16**（24.50M） | `AUC 0.49~0.52`（近随机排序）；epoch 间在「全判真」与「全判假」两个极端来回翻 |")
    A("| 本实验 random 臂（随机初始化） | **b16**（90.53M） | `AUC 0.62~0.63`，但 `acc` 恒为 0.5、`tn=0` —— **退化**，全部判假 |")
    A("")
    A("怀疑根因是「骨干随机初始化 + 分类头容量有限 → 学不动」。")
    A("本实验用一对**除骨干权重外完全相同**的配置来验证这个怀疑。")
    A("")
    A("## 2. 实验设置")
    A("")
    A("| 项 | 值 |")
    A("| --- | --- |")
    A("| 唯一自变量 | 骨干是否加载 ImageNet-1K 预训练权重 |")
    A("| A 臂 | `configs/ablation_b16_random.yaml`（无预训练） |")
    A("| B 臂 | `configs/ablation_b16_pretrained.yaml`（`pretrained/vit_b16_imagenet.pt`） |")
    A("| 骨干 | ViT-B/16（768 维 / 12 层 / 12 头，90.53M 参数） |")
    A("| 训练集 | ForenSynths `val` 划分 1600 张（ProGAN 20 类保留集） |")
    A("| 验证集 | ForenSynths `test` 划分 1300 张（与训练集图像无交集） |")
    A("| 阶段 | 仅 `stage1_cls_pretrain`，3 epoch |")
    A("| batch / 优化器 | 8 / AdamW，lr `1e-4 → 1e-6` 余弦，seed 3407 |")
    A("| 设备 | CPU（7 核，torch 2.14.0+cpu） |")
    A("")
    A("**两臂配置的一致性证据**：把注释过滤掉后，两个配置文件只剩 3 处不同 ——")
    A("")
    A("| 差异项 | random 臂 | pretrained 臂 | 是否影响训练 |")
    A("| --- | --- | --- | --- |")
    A("| `project.name` | `VIB-Net 对照实验 (random)` | `VIB-Net 对照实验 (pretrained)` | ❌ 仅实验标签 |")
    A("| `project.output_dir` | `outputs/ablation_random` | `outputs/ablation_pretrained` | ❌ 仅落盘路径 |")
    A("| **`model.spatial.backbone_weights`** | **（不设置）** | **`pretrained/vit_b16_imagenet.pt`** | ✅ **唯一自变量** |")
    A("")
    A("即：**除骨干权重外，两臂的模型结构、数据、超参、随机种子完全一致。** 复核命令：")
    A("")
    A("```bash")
    A('diff <(grep -v "^[[:space:]]*#" configs/ablation_b16_random.yaml) \\')
    A('     <(grep -v "^[[:space:]]*#" configs/ablation_b16_pretrained.yaml)')
    A("```")
    A("")
    A("### 口径限制（写论文/中期材料时必须带上）")
    A("")
    A("1. 训练集用的是 ForenSynths 的 **val 划分**，不是官方 train 划分。")
    A("   官方 `progan_train.7z`（70GB）在受限网络下未取得。二者同为 ProGAN 生成、")
    A("   且与测试集无图像重叠，可作为**小规模替代**，但**不能声称复现了官方协议**。")
    A("2. 骨干用的是 **ImageNet-1K 预训练 ViT-B/16**（torchvision 权重），")
    A("   不是申报书写的 `openai/clip-vit-base-patch16`。原因是本机网络对")
    A("   `huggingface.co` 间歇性完全阻断，CLIP 权重取不到。两者同为 ViT-B/16 结构、")
    A("   同为大规模图像预训练，作为**可训练性对照**是成立的；")
    A("   但正式论文里的「CLIP 语义先验」结论需要拿到 CLIP 权重后重跑。")
    A("3. 本实验只回答「**能不能训起来**」，不回答「跨生成器泛化到什么水平」。")
    A("   跨生成器结论请用 `scripts/eval_cross_generator.py` 与完整三阶段训练。")
    A("")

    # ---- 逐 epoch 对比 ----
    A("## 3. 逐 epoch 验证指标对比")
    A("")
    A(f"（验证集：ForenSynths test，共 1300 张；评测在 epoch 末进行）")
    A("")
    for i, r in enumerate(results, 1):
        arm = r["arm"]
        A(f"### 3.{i} {arm['title']}（`{arm['key']}`）")
        A("")
        A(f"数据来源：`{r.get('source', '无')}`")
        A("")
        h = r["history"]
        if not h:
            A("> ⚠ 未找到训练历史（`history.json` 与 `train_log.txt` 都读不到数据）。")
            A("")
            continue
        A("| epoch | lr | train_loss | val_acc | val_precision | val_recall | "
          "val_auc | val_ap | tn/fp/fn/tp |")
        A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in h:
            A("| {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{}/{} |".format(
                e.get("epoch"),
                f"{e.get('lr', 0):.2e}",
                fmt(e.get("train_loss")),
                fmt(e.get("val_cls_acc")),
                fmt(e.get("val_cls_precision")),
                fmt(e.get("val_cls_recall")),
                fmt(e.get("val_cls_auc")),
                fmt(e.get("val_cls_ap")),
                int(e.get("val_cls_tn") or 0), int(e.get("val_cls_fp") or 0),
                int(e.get("val_cls_fn") or 0), int(e.get("val_cls_tp") or 0)))
        A("")
    A("")

    # ---- 最佳 epoch 汇总 ----
    A("## 4. 主结果汇总")
    A("")
    A("### 4.1 按 val_cls_auc 挑选的最佳 epoch")
    A("")
    A("| 臂 | 骨干权重 | 最佳 epoch | val_acc | val_auc | val_ap | val_f1 |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    summary = {}
    for r in results:
        arm = r["arm"]
        b = best_epoch(r["history"] or [])
        summary[arm["key"]] = b
        A("| {} | {} | {} | {} | {} | {} | {} |".format(
            arm["title"], arm["backbone_weights"],
            b.get("epoch") if b else "—",
            fmt(b.get("val_cls_acc") if b else None),
            fmt(b.get("val_cls_auc") if b else None),
            fmt(b.get("val_cls_ap") if b else None),
            fmt(b.get("val_cls_f1") if b else None)))
    A("")
    for r in results:
        A(f"- `{r['arm']['key']}` 数据来源：{r.get('source', '无')}")
    A("")
    A("### 4.2 ★ 决定性判据：判决是否退化")
    A("")
    A("这是本实验最该看的一张表。`ACC` 在退化臂上无意义（全判假时 `acc` 恰等于负类占比），")
    A("所以要看的是**最佳 epoch 内 `tn` 与 `tp` 是否同时为正**。")
    A("")
    A("| 臂 | 最佳 epoch | tn | tp | fn | fp | **同时为正？** | 判决 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        arm = r["arm"]
        b = summary.get(arm["key"])
        if not b:
            A(f"| {arm['title']} | — | — | — | — | — | — | ⚠ 缺数据 |")
            continue
        tn = b.get("val_cls_tn")
        tp = b.get("val_cls_tp")
        ok = (isinstance(tn, (int, float)) and isinstance(tp, (int, float))
              and tn > 0 and tp > 0)
        A("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm["title"], b.get("epoch"),
            _i(tn), _i(tp), _i(b.get("val_cls_fn")), _i(b.get("val_cls_fp")),
            "**是**" if ok else "**否**",
            "有效（边界已建立）" if ok else "**退化**（全判一类）"))
    A("")
    A("> 判读规则：只要一臂的 `tn` 或 `tp` 为 0，该臂的 `acc`、`precision`、`f1`")
    A("> 都不能作为性能证据——它们在「全判假」下分别等于负类占比、负类占比、`2p/(1+p)`，")
    A("> 与模型无关。**唯一有意义的排序指标是 `AUC`/`AP`。**")
    A("")

    # ---- 逐生成器 ----
    A("## 5. 逐生成器跨模型评测")
    A("")
    A(f"每个生成器各取 {per_class} 张真 + {per_class} 张假；真图按文件名去重"
      "（ForenSynths 各生成器目录下的 `0_real` 来自同一批真实图，"
      "不去重会把同一张真图重复计入 13 次）。")
    A("")
    any_cg = any(r["cross_gen"] for r in results)
    if not any_cg:
        A("> ⚠ 未找到逐生成器评测结果。如果跳过了 `--skip-eval`，"
          "请重新运行脚本且不带该参数。")
        A("")
    else:
        tables = {}
        for r in results:
            rows, macro = find_cross_gen_rows(r["cross_gen"] or {})
            tables[r["arm"]["key"]] = (rows, macro)
        names = sorted({n for rows, _ in tables.values() for n, _ in rows})
        if names:
            header = ["生成器", "真/假样本数"]
            for r in results:
                k = r["arm"]["key"]
                header += [f"{k} AUC", f"{k} ACC"]
            A("| " + " | ".join(header) + " |")
            A("| " + " | ".join(["---"] * len(header)) + " |")
            for n in names:
                # 样本数取第一个有该生成器的臂（各臂采样应完全一致，见 effective_per_class）
                nrf = "—"
                for r in results:
                    rows, _ = tables[r["arm"]["key"]]
                    m = dict(rows).get(n, {})
                    nr, nf = m.get("n_real"), m.get("n_fake")
                    if isinstance(nr, (int, float)) and isinstance(nf, (int, float)):
                        nrf = f"{int(nr)}/{int(nf)}"
                        break
                line = [n, nrf]
                for r in results:
                    rows, _ = tables[r["arm"]["key"]]
                    m = dict(rows).get(n, {})
                    line += [fmt(pick(m, "auc", "AUC")),
                             fmt(pick(m, "acc", "accuracy", "ACC"))]
                A("| " + " | ".join(line) + " |")
            A("")
        # 宏/微平均
        A("### 5.1 宏平均与全局平均")
        A("")
        A("「宏平均」= 13 个生成器各自指标的算术平均（每个生成器等权，")
        A("不受大生成器样本量影响）；「全局」= 把所有样本混在一起算（大生成器主导）。")
        A("")
        A("| 臂 | 生成器数 | 宏平均 AUC | 宏平均 ACC | 全局 AUC | 全局 ACC |")
        A("| --- | --- | --- | --- | --- | --- |")
        for r in results:
            rows, macro = tables[r["arm"]["key"]]
            A("| {} | {} | {} | {} | {} | {} |".format(
                r["arm"]["title"],
                len(rows) if rows else pick(macro, "n_generators",
                                            "num_generators", "n_gen",
                                            default="—"),
                fmt(pick(macro, "auc_macro", "macro_auc", "auc")),
                fmt(pick(macro, "acc_macro", "macro_acc", "acc")),
                fmt(pick(macro, "auc")),
                fmt(pick(macro, "acc"))))
        A("")

    # ---- 结论 ----
    A("## 6. 结论")
    A("")
    br = summary.get("random")
    bp = summary.get("pretrained")
    ar = br.get("val_cls_auc") if br else None
    ap = bp.get("val_cls_auc") if bp else None

    def _deg(b):
        """返回 (tn, tp, 是否退化)。数据缺失返回 (None, None, None)。"""
        if not b:
            return None, None, None
        tn, tp = b.get("val_cls_tn"), b.get("val_cls_tp")
        if not isinstance(tn, (int, float)) or not isinstance(tp, (int, float)):
            return tn, tp, None
        return tn, tp, (tn <= 0 or tp <= 0)

    tnr, tpr, degr = _deg(br)
    tnp, tpp, degp = _deg(bp)

    if ar is None or ap is None:
        A("> 两臂结果不齐（可能有一臂尚未跑完），暂不自动判定。")
    else:
        gap = ap - ar
        A(f"- **排序指标**：随机初始化臂最佳 val AUC = **{ar:.4f}**"
          f"；ImageNet 预训练臂最佳 val AUC = **{ap:.4f}**；差值 **{gap:+.4f}**。")
        A("- **判决有效性（决定性）**：")
        A(f"  - 随机初始化臂：最佳 epoch `tn={_i(tnr)}` / `tp={_i(tpr)}` → "
          + ("**退化，全判一类**" if degr else
             "两类均判出" if degr is False else "数据缺失"))
        A(f"  - ImageNet 预训练臂：最佳 epoch `tn={_i(tnp)}` / `tp={_i(tpp)}` → "
          + ("**退化，全判一类**" if degp else
             "**有效，边界已建立**" if degp is False else "数据缺失"))
        A("")
        if degr is True and degp is False:
            A("- 判定：**预训练权重把模型从「判决退化」推进到「判决有效」**，"
              f"这是定性差别，不依赖 {gap:+.4f} 这个数值大小。")
            A("")
            A("  但要如实说明**这个对照的强度边界**：")
            A("")
            A(f"  1. 数值上的差距（AUC {gap:+.4f}）**不算大**。"
              f"b16 随机初始化臂的 `AUC={ar:.2f}`")
            A("     说明它的排序分数里**并非完全没有信号**——一个合理解释是：频域分支"
              "（2D-DFT 幅值/相位 + 可学习掩码）是**确定性变换**，其输出与骨干是否预训练无关，"
              "所以即便是随机初始化骨干，分类头仍能从中读到一点生成痕迹。")
            A("  2. 但它的**阈值判决完全塌到一侧**（`tn=0`，全部判假），说明"
            "「分数有微弱排序能力」与「边界能建立」是两件事，后者需要预训练权重。")
            A("  3. 因此本实验的结论应表述为：**预训练权重是判决边界能够建立的必要条件"
              "（在本机 CPU 小规模设定下），而非充分条件**——预训练臂的 "
              f"AUC {ap:.2f} 距离申报指标（ACC≥91%）仍很远，"
              "那需要完整三阶段训练 + 70GB 训练集 + GPU。")
        elif degr is True and degp is True:
            A("- 判定：**两臂都退化** —— 预训练权重在当前小规模设定下**不足以**建立判决边界。"
              "需要排查：预训练权重是否真的装进去了（看日志里的张量数）、"
              "学习率、VIB 的 KL 强度、以及训练集规模是否太小。")
        elif degr is False and degp is False:
            A(f"- 判定：两臂都建立了边界，预训练权重带来 **{gap:+.4f}** 的 AUC 改善，"
              "属**数值改善**而非定性差别。此时应把注意力转到数据规模与训练轮数上。")
        else:
            A("- 判定：数据不齐，见上表。")
        A("")
        A("- 口径提醒：本结论建立在 ForenSynths val→test 的**同生成器留出**设定上"
          "（训练与验证同为 ProGAN 生成、图像无交集），只能说明「可训练性」，"
          "不能外推为跨生成器泛化性能。")
    A("")
    A("## 7. 复现命令")
    A("")
    A("```bash")
    A("python scripts/make_ablation_configs.py           # 生成两臂配置")
    A("python scripts/run_ablation.py                    # 训练 + 评测 + 出本报告")
    A("python scripts/run_ablation.py --report-only      # 只重新汇总")
    A("```")
    A("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="预训练权重消融对照实验")
    ap.add_argument("--arms", default="random,pretrained",
                    help="要跑的臂：random,pretrained（逗号分隔）")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练，只评测")
    ap.add_argument("--skip-eval", action="store_true", help="跳过逐生成器评测")
    ap.add_argument("--report-only", action="store_true",
                    help="不训练不评测，只用已有结果重新出报告")
    ap.add_argument("--per-class", type=int, default=120,
                    help="逐生成器评测时每类取多少张")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    keys = [k.strip() for k in args.arms.split(",") if k.strip()]
    bad = [k for k in keys if k not in BY_KEY]
    if bad:
        log(f"未知的臂：{bad}，可选：{list(BY_KEY)}")
        return 2
    arms = [BY_KEY[k] for k in keys]

    t_start = time.time()
    if not args.report_only:
        for arm in arms:
            if not args.skip_train:
                train_arm(arm)
            if not args.skip_eval:
                eval_arm(arm, args.per_class, args.batch_size)

    results = [load_arm_result(a) for a in ARMS]
    md = build_report(results, effective_per_class(results, args.per_class))
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    log(f"报告已写入 {REPORT_PATH}")
    log(f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
