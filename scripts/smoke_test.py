"""端到端冒烟测试 —— 检查申报书里每一处公式对应的张量维度与梯度是否贯通

跑通即说明：模型能前向、能反向、损失都参与优化、各模块维度与公式一致。

用法：
    python scripts/smoke_test.py            # 只做维度/梯度自检（秒级）
    python scripts/smoke_test.py --full     # 额外生成演示数据 + 训练 1 轮 + 导出 ONNX
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.engine.trainer import Logger as TrainLogger, Trainer, optimization_parameters  # noqa: E402
from src.losses.multi_task import MultiTaskLoss, ZScoreNormalizer  # noqa: E402
from src.models.cs_cam import CSCAM  # noqa: E402
from src.models.freq_branch import FrequencyBranch  # noqa: E402
from src.models.gradient_stop import GradientStopLayer, grad_stop  # noqa: E402
from src.models.mobile_unetv2 import EdgeHead, LocalizationBranch  # noqa: E402
from src.models.spatial_branch import SpatialBranch  # noqa: E402
from src.models.vib import HierarchicalVIB, beta_schedule  # noqa: E402
from src.models.vibnet import build_model, load_config  # noqa: E402

OK, BAD = "\033[32m[OK]\033[0m", "\033[31m[FAIL]\033[0m"
results = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, cond))
    print(f"  {OK if cond else BAD} {name}" + (f"  {detail}" if detail else ""))


# ==========================================================================
def test_modules(cfg: dict) -> None:
    print("\n【1】单模块维度自检（对照申报书公式）")

    # ---- 1.1(1) 空域分支 ----
    print("\n  1.1(1) 空域特征提取分支  —— 公式 (1)(2)(3)")
    cfg["model"]["spatial"]["backbone"] = "tiny_vit"
    sb = SpatialBranch(cfg).eval()
    info = sb.sanity_check()
    for k, v in info.items():
        print(f"      {k}: {v}")
    check("phi(x) 为 768 维 (公式1)", info["phi(x) 维度"][-1] == 768)
    check("F_spa 通道 896 (公式3)", info["F_spa 特征图"][1] == 896)
    check("F_spa 空间 14x14", info["F_spa 特征图"][2:] == (14, 14))
    check("F_spa(GAP) 896 维", info["F_spa(GAP) 向量维度"][-1] == 896)

    # ---- 1.1(2) 频域分支 ----
    print("\n  1.1(2) 频域特征提取分支 —— 公式 (4)-(12)")
    fb = FrequencyBranch(cfg).eval()
    finfo = fb.sanity_check()
    for k, v in finfo.items():
        print(f"      {k}: {v}")
    check("A(u,v) 为 224x224 (公式5)", finfo["A(u,v)"] == (2, 1, 224, 224))
    check("P(u,v) 为 224x224 (公式6)", finfo["P(u,v)"] == (2, 1, 224, 224))
    check("F_freq 通道 256 (公式12)", finfo["F_freq 特征图"][1] == 256)

    # ---- 1.1(3) CS-CAM ----
    print("\n  1.1(3) 交叉注意力融合 —— 公式 (13)-(17)")
    cam = CSCAM(896, 256, 512).eval()
    cinfo = cam.sanity_check()
    for k, v in cinfo.items():
        print(f"      {k}: {v}")
    check("F_fusion 为 (2,512,14,14) (公式17)", cinfo["F_fusion"] == (2, 512, 14, 14))

    # ---- 1.2 分层 VIB ----
    print("\n  1.2 分层 VIB —— 公式 (20)-(25)")
    vib = HierarchicalVIB(512, 256, 256).eval()
    vinfo = vib.sanity_check()
    for k, v in vinfo.items():
        print(f"      {k}: {v}")
    check("mu 为 256 维 (公式20)", vinfo["mu"][-1] == 256)
    check("sigma 非负 (公式21)", vinfo["sigma_min/max"][0] >= 0)
    check("z 为 256 维 (公式23)", vinfo["z"][-1] == 256)
    check("KL 已裁剪到 <=10 (公式25)", vinfo["kl_clipped"] <= 10.0 + 1e-6)
    b = (beta_schedule(10), beta_schedule(30), beta_schedule(50))
    check("beta 退火分段正确 (公式24)",
          abs(b[0]) < 1e-9 and abs(b[1] - 0.05) < 1e-6 and abs(b[2] - 0.1) < 1e-6,
          f"beta(10),beta(30),beta(50)={b}")

    # ---- 1.3(1) 梯度停止 ----
    print("\n  1.3(1) 梯度停止层")
    v1 = GradientStopLayer.verify(stop=True)
    v2 = GradientStopLayer.verify(stop=False)
    check("stop=True 阻断梯度", v1["上游梯度全零"])
    check("stop=False 恒等传递", v2["上游梯度全零"] is False)

    # ---- 1.3(3) 定位支路 ----
    print("\n  1.3(3) 定位支路 Mobile-UNetv2 —— 公式 (28)(29)")
    loc = LocalizationBranch(cfg).eval()
    with torch.no_grad():
        lo = loc(torch.randn(2, 512, 14, 14))
    print(f"      mask_prob: {tuple(lo['mask_prob'].shape)}  "
          f"edge_prob: {tuple(lo['edge_prob'].shape)}  "
          f"参数量 {loc.num_millions():.2f} M")
    check("M_pred 为 224x224 (公式28)", tuple(lo["mask_prob"].shape) == (2, 1, 224, 224))
    check("M_edge_pred 为 224x224 (公式29)", tuple(lo["edge_prob"].shape) == (2, 1, 224, 224))
    check("定位头参数量 < 5M", loc.num_millions() < 5.0, f"{loc.num_millions():.2f}M")

    egt = EdgeHead.canny_edge(torch.zeros(2, 1, 224, 224))
    sq = torch.zeros(2, 1, 224, 224)
    sq[:, :, 50:150, 50:150] = 1.0
    egt2 = EdgeHead.canny_edge(sq)
    print(f"      全零掩码边缘像素={float(egt.sum()):.0f}，"
          f"方块掩码边缘像素={float(egt2.sum()):.0f}")
    check("Canny 提取到边缘 (公式29)", float(egt2.sum()) > 0 and float(egt.sum()) == 0)


# ==========================================================================
def test_end_to_end(cfg: dict) -> torch.nn.Module:
    print("\n【2】端到端前向 + 反向（整图真伪 + 像素定位 + 边缘）")
    cfg["model"]["spatial"]["backbone"] = "tiny_vit"
    model = build_model(cfg)
    s = model.summary()
    for k, v in s.items():
        print(f"      {k:18s} {v:.3f}" if isinstance(v, float) else f"      {k:18s} {v}")
    info = model.sanity_check(batch=2)
    for k, v in info.items():
        print(f"      {k}: {v}")
    check("cls_logits (B,2) (公式27)", info["cls_logits (公式27)"] == (2, 2))
    check("mask_prob (B,1,224,224) (公式28)", info["mask_prob (公式28)"] == (2, 1, 224, 224))

    from src.deploy.optimize_openvino import estimate_model_size
    size_info = estimate_model_size(s["total(M)"])
    print("      体积核算：" + "  ".join(
        f"{k}={v}" for k, v in size_info.items() if k.endswith("(MB)")))
    check("模型体积可满足 ≤120MB（INT8 量化后）", size_info["INT8达标"],
          f"FP32={size_info['FP32(MB)']}MB, "
          f"FP16={size_info['FP16(MB)']}MB, INT8={size_info['INT8(MB)']}MB"
          f" -> {size_info['建议']}")

    # ---- 损失与反向 ----
    print("\n【3】多任务损失 + 反向传播（公式 26/29/30/31）")
    model.train()
    crit = MultiTaskLoss(cfg)
    # ★ 必须用 Trainer 的同一份参数收集逻辑。以前这里写的是
    #   AdamW(model.trainable_parameters())，漏掉了 criterion 里的 log σ²，
    #   于是测试全绿、但真实训练里"可学习加权"从来没更新过。
    opt_params, n_extra = optimization_parameters(model, crit)
    opt = torch.optim.AdamW(opt_params, lr=1e-4)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 1, 0])
    mask = torch.zeros(4, 1, 224, 224)
    mask[1, :, 60:140, 60:140] = 1
    mask[2] = 1
    mvalid = torch.tensor([True, True, True, False])       # 第 4 个样本无掩码（模拟混合批次）

    out = model(x, sample_vib=True, beta=0.05)
    losses = crit(out, y, mask, beta=0.05, active_tasks=["cls", "loc", "edge"],
                  update_norm=True, mask_valid=mvalid)
    print(f"      raw losses: {losses['raw']}")
    print(f"      weights   : {losses['weights']}")
    print(f"      skipped   : {losses['skipped']}")
    check("四个损失项均参与计算 (公式31)", len(losses["raw"]) == 4)

    # 不确定性加权的参数必须真的进了优化器，否则权重会恒等于 0.5（假"可学习"）
    uw = crit.weighting
    check("不确定性加权参数已纳入优化器 (不是只收 model 参数)",
          n_extra > 0 and uw is not None and uw.log_var.requires_grad,
          f"额外纳入 {n_extra} 组 log σ²")
    s_before = uw.log_var.detach().clone() if uw is not None else None

    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # 申报书 1.2(3) 梯度裁剪
    grad_ok, no_grad = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and torch.isfinite(p.grad).all():
                grad_ok += 1
            else:
                no_grad += 1
    check("所有可训练参数梯度有效", no_grad == 0, f"{grad_ok} 个参数，{no_grad} 个异常")
    opt.step()
    check("优化器 step 未报错", True)

    # 真正验证"可学习"：一步之后 log σ² 必须发生变化。
    # 旧断言只检查权重 > 0（0.5*exp(-s) 恒为正），所以永远为真、什么也没验证住。
    if s_before is not None:
        moved = not torch.allclose(s_before, uw.log_var.detach())
        check("不确定性权重确实随梯度更新（不再恒为 0.5）", moved,
              f"log σ² 变化量 max|Δ|={float((uw.log_var.detach()-s_before).abs().max()):.3e}"
              if moved else "log σ² 一步未动，说明未进优化器")
        w_now = crit.weights()
        print(f"      σ after 1 step: "
              f"{ {k: round(float(v), 4) for k, v in w_now.items()} }")

    # ---- 三阶段冻结策略 ----
    print("\n【4】三阶段冻结/解冻策略")
    for stage in cfg["train"]["stages"]:
        model.apply_stage(stage.get("freeze", []), stage.get("train", ["all"]))
        n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"      {stage['name']:24s} 可训练 {n:8.3f} M  "
              f"freeze={stage.get('freeze')} train={stage.get('train')}")
        check(f"{stage['name']} 有可训练参数", n > 0)

    model.apply_stage([], ["all"])
    return model


# ==========================================================================
def test_full_pipeline(cfg: dict) -> None:
    print("\n【5】完整流水线：演示数据 -> 训练 -> 评测 -> ONNX 导出")
    from src.data.synth import generate_demo_dataset

    demo_root = os.path.join(ROOT, "data", "demo")
    if not (os.path.isdir(os.path.join(demo_root, "train", "image"))):
        info = generate_demo_dataset(demo_root, train=48, val=24, test=24)
        print(f"      演示数据集已生成：{info}")

    from src.data.datasets import build_dataloaders
    from src.engine.trainer import Logger, Trainer
    from src.models.vibnet import build_model as bm

    sub = load_config(os.path.join(ROOT, "configs/default.yaml"))
    sub["model"]["spatial"]["backbone"] = "tiny_vit"
    sub["data"]["num_workers"] = 0
    sub["train"]["batch_size"] = 8

    loaders = build_dataloaders(sub, use_demo=True)
    print(f"      train={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}"
          f"  test={len(loaders['test'].dataset)}")

    for s in sub["train"]["stages"]:
        s["epochs"] = 1
    sub["train"]["log_every"] = 3
    sub["train"]["save_every"] = 999
    sub["project"]["output_dir"] = os.path.join(ROOT, "outputs", "smoke")
    sub["project"]["ckpt_dir"] = os.path.join(ROOT, "checkpoints")

    model = bm(sub)
    logger = Logger(os.path.join(sub["project"]["output_dir"], "smoke_log.txt"))
    trainer = Trainer(sub, model, torch.device("cpu"), loaders["train"], loaders["val"],
                      logger=logger, ckpt_prefix="smoke")
    hist = trainer.fit()
    check("三阶段训练可运行", len(hist) > 0, f"{len(hist)} 条记录")

    best = os.path.join(sub["project"]["ckpt_dir"], "smoke_best.pt")
    check("最佳权重已保存", os.path.exists(best), best)

    from src.evaluation.evaluate import evaluate_model, format_report
    res = evaluate_model(sub, best, "test", use_demo=True, device="cpu",
                         max_batches=3, save_dir=sub["project"]["output_dir"])
    print(format_report(res))
    check("评测流程可运行", "cls" in res and bool(res["cls"]))

    from src.deploy.export_onnx import export_onnx, simplify_onnx
    onnx_path = os.path.join(ROOT, "deploy", "smoke.onnx")
    try:
        export_onnx(best, onnx_path, 17, 224, os.path.join(ROOT, "configs/default.yaml"))
        check("ONNX 导出成功", os.path.exists(onnx_path),
              f"{os.path.getsize(onnx_path)/1e6:.1f} MB")
        p2 = simplify_onnx(onnx_path)
        print(f"      simplify -> {p2}")
    except Exception as e:  # noqa: BLE001
        check("ONNX 导出成功", False, f"{type(e).__name__}: {e}")

    from src.deploy.optimize_openvino import convert_to_openvino, benchmark
    try:
        ir = convert_to_openvino(onnx_path, os.path.join(ROOT, "deploy", "smoke_ir"))
        if ir:
            benchmark(onnx_path, ir, 224, n_warmup=2, n_runs=5)
            check("OpenVINO 转换 + 性能测试", True)
        else:
            check("OpenVINO 转换 + 性能测试", False, "未安装 openvino（非致命）")
    except Exception as e:  # noqa: BLE001
        check("OpenVINO 转换 + 性能测试", False, f"{type(e).__name__}: {e}")

    # 统一推理封装（桌面工具用的就是它）
    try:
        from src.deploy.inference import Detector, list_available_backends
        print(f"      可用后端：{list_available_backends()}")
        det = Detector(backend="torch", ckpt=best,
                       config_path=os.path.join(ROOT, "configs/default.yaml"))
        sample = os.path.join(demo_root, "test", "image")
        any_img = sorted(os.listdir(sample))[0] if os.path.isdir(sample) else None
        if any_img:
            r = det.detect(os.path.join(sample, any_img))
            print(f"      Detector 推理：{r.summary()}")
            check("Detector 统一推理封装可用", 0.0 <= r.prob_fake <= 1.0)
    except Exception as e:  # noqa: BLE001
        check("Detector 统一推理封装可用", False, f"{type(e).__name__}: {e}")


# ==========================================================================
def test_metric_definitions() -> None:
    """指标口径回归测试（子进程调用，失败时回显尾部日志）。

    为什么必须挂在自检里：AUC/AP/mIoU 的实现一旦被改动（例如并列分数不做秩校正、
    把非标准积分当成 AP、把全零预测的真实图记成 IoU=1），数值会**静默变错**而
    不抛异常 —— 训练照样跑、报告照样出，只有答辩时才被发现。这里把它钉成一项检查。
    """
    script = os.path.join(ROOT, "scripts", "test_metric_definitions.py")
    if not os.path.exists(script):
        check("指标口径回归测试", False, "找不到 scripts/test_metric_definitions.py")
        return
    try:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        out = "\n".join(l for l in out.splitlines() if "crashpad" not in l)
        if proc.returncode == 0:
            n = out.count("✅")
            check("指标口径回归测试（AUC/AP/三种 mIoU）", True, f"{n} 项断言通过")
        else:
            tail = "\n      ".join(out.strip().splitlines()[-12:])
            check("指标口径回归测试（AUC/AP/三种 mIoU）", False,
                  f"返回码 {proc.returncode}\n      尾部日志：\n      {tail}")
    except Exception as e:  # noqa: BLE001
        check("指标口径回归测试（AUC/AP/三种 mIoU）", False, f"{type(e).__name__}: {e}")


# ==========================================================================
def test_audit_tools() -> None:
    """外部数据集审计脚本的口径自检（子进程调用）。

    为什么必须挂在自检里：`audit_flat_realfake.py` 的结论直接决定
    「同伴给来的这份数据能不能用」。它自己的 `auc`、`blockiness`、`pixel_feats`
    若算错，**审计会照常打印结论**，只是把干净数据判成有捷径（或反之）——
    不抛异常、没人会发现。这里把它钉成一项检查。
    """
    script = os.path.join(ROOT, "scripts", "audit_flat_realfake.py")
    if not os.path.exists(script):
        check("外部数据集审计自检", False, "找不到 scripts/audit_flat_realfake.py")
        return
    try:
        proc = subprocess.run(
            [sys.executable, script, "--self-test"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        out = "\n".join(l for l in out.splitlines() if "crashpad" not in l)
        if proc.returncode == 0:
            n = out.count("✅")
            check("外部数据集审计自检（AUC/块效应/像素级特征）", True, f"{n} 项断言通过")
        else:
            tail = "\n      ".join(out.strip().splitlines()[-12:])
            check("外部数据集审计自检（AUC/块效应/像素级特征）", False,
                  f"返回码 {proc.returncode}\n      尾部日志：\n      {tail}")
    except Exception as e:  # noqa: BLE001
        check("外部数据集审计自检（AUC/块效应/像素级特征）", False,
              f"{type(e).__name__}: {e}")


# ==========================================================================
def test_leak_audit() -> None:
    """训练/评测集重叠审计脚本的口径自检（子进程调用）。

    为什么必须挂在自检里：`audit_train_eval_overlap.py` 的结论直接决定
    「configs 里『与训练集图像无交集』这句话能不能写进论文」。它的重叠判定
    若算错，审计会照常打印结论 —— 两个方向的错都很贵：
      * 漏报（同图不同名没比出来）→ 带着泄漏发论文；
      * 误报（同名不同图被当成泄漏，实测 car 有 22 个）→ 白改实验。
    所以自检里**专门钉住"同名不同图必须不计入"**这一条（见 `docs/08` D33）。
    """
    script = os.path.join(ROOT, "scripts", "audit_train_eval_overlap.py")
    if not os.path.exists(script):
        check("训练/评测重叠审计自检", False,
              "找不到 scripts/audit_train_eval_overlap.py")
        return
    try:
        proc = subprocess.run(
            [sys.executable, script, "--self-test"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        out = "\n".join(l for l in out.splitlines() if "crashpad" not in l)
        if proc.returncode == 0:
            check("训练/评测重叠审计自检（内容哈希比对）", True,
                  f"{out.count('✅')} 项断言通过")
        else:
            tail = "\n      ".join(out.strip().splitlines()[-12:])
            check("训练/评测重叠审计自检（内容哈希比对）", False,
                  f"返回码 {proc.returncode}\n      尾部日志：\n      {tail}")
    except Exception as e:  # noqa: BLE001
        check("训练/评测重叠审计自检（内容哈希比对）", False,
              f"{type(e).__name__}: {e}")


# ==========================================================================
def test_fetch_extract() -> None:
    """数据解压路径的口径自检（子进程调用）。

    为什么必须挂在自检里：新增的「分卷直读」一旦偏移算错，`zipfile` **不会报错**，
    只会解出一堆错位数据；而 `--classes` 的筛选逻辑一旦选错（历史上选进过目录条目），
    会"成功解压 0 个文件"并据此**删掉源压缩包** —— 实测丢失过 74.9 GB 训练集分卷
    （见 `docs/08` D32）。这类失败全程不抛异常，只能靠自检钉住。

    ⚠ 用字节读取再手工解码：中文 Windows 下 `text=True` 会按 GBK 解码，
    输出里含非 UTF-8 字节时读取子线程会静默失败、`stdout` 变成 `None`。
    """
    script = os.path.join(ROOT, "scripts", "fetch_datasets.py")
    if not os.path.exists(script):
        check("数据解压路径自检", False, "找不到 scripts/fetch_datasets.py")
        return
    try:
        proc = subprocess.run(
            [sys.executable, script, "--self-test"],
            cwd=ROOT, capture_output=True, timeout=180,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="replace") + \
              (proc.stderr or b"").decode("utf-8", errors="replace")
        out = "\n".join(l for l in out.splitlines() if "crashpad" not in l)
        if proc.returncode == 0:
            n = out.count("✅")
            check("数据解压路径自检（分卷直读/空间预检/类别筛选/ZIP64）",
                  True, f"{n} 项断言通过")
        else:
            tail = "\n      ".join(out.strip().splitlines()[-12:])
            check("数据解压路径自检（分卷直读/空间预检/类别筛选/ZIP64）", False,
                  f"返回码 {proc.returncode}\n      尾部日志：\n      {tail}")
    except Exception as e:  # noqa: BLE001
        check("数据解压路径自检（分卷直读/空间预检/类别筛选/ZIP64）", False,
              f"{type(e).__name__}: {e}")


# ==========================================================================
def test_amp_policy() -> None:
    """混合精度策略的真值表回归测试（纯逻辑，秒级）。

    为什么必须挂在自检里：AMP 是典型的"上云才第一次启用"的路径 —— 本机没有
    CUDA，于是它成了唯一一次测试都轮不到的代码。一旦配错（对 bf16 开 GradScaler、
    或在 CPU 上开了 bf16 autocast），训练**不会报错**，只会变慢几十倍或静默失效，
    而云上按小时计费，账单照付。所以把这张表钉死。
    """
    try:
        from src.engine.trainer import resolve_amp
    except Exception as e:  # noqa: BLE001
        check("混合精度策略", False, f"导入失败：{type(e).__name__}: {e}")
        return

    cases = [
        # 配置                                设备      enabled  dtype           scaler
        ({"amp": False},                      "cuda",  False,   torch.bfloat16, False),
        ({"amp": True},                       "cuda",  True,    torch.bfloat16, False),
        ({"amp": True, "amp_dtype": "float16"}, "cuda", True,   torch.float16,  True),
        ({"amp": True},                       "cpu",   False,   torch.bfloat16, False),
        ({"amp": True, "amp_dtype": "float16"}, "cpu", False,   torch.float16,  False),
    ]
    bad = []
    for cfg, dev, en, dt, sc in cases:
        got = resolve_amp(cfg, dev)
        if got != (en, dt, sc):
            bad.append(f"{cfg}@{dev} -> {got}，期望 {(en, dt, sc)}")
    # 额外单独钉一条最容易犯的错：bf16 绝不能配 GradScaler
    if resolve_amp({"amp": True, "amp_dtype": "bfloat16"}, "cuda")[2]:
        bad.append("bf16 竟然要 GradScaler —— 这正是 docs/08 C19 记录的假安全感")

    check("混合精度策略（bf16 不配 GradScaler / CPU 上必关）", not bad,
          "; ".join(bad) if bad else f"{len(cases) + 1} 条真值表断言通过")


def test_amp_upsample_fp32() -> None:
    """D43 回归：双线性上采样必须绕开 autocast 的 bf16 慢路径。

    为什么必须挂在自检里：`aten::upsample_bilinear2d_backward` 在 bf16 上单次
    实测 2.9 s、fp32 只要 ~7 ms（400 倍），而它**只在 CUDA + autocast 下出现**
    —— 本机没有 CUDA，天然跑不到这条路径；一旦回归，训练**不会报错**，只会把
    stage2 拖慢 14 倍，云上账单照付。所以这里用两件事把契约钉住：
      ① **行为**：给 bf16 输入（以及 autocast 打开时），函数必须返回 fp32；
      ② **源码**：两处上采样都必须经由 helper，不允许再出现裸的 `F.interpolate`
         （裸调用在 autocast 下会跟随输入 dtype 走 bf16 慢路径）。
    """
    try:
        from src.models.mobile_unetv2 import _bilinear_upsample_fp32
    except Exception as e:  # noqa: BLE001
        check("上采样强制 fp32（D43）", False, f"导入失败：{type(e).__name__}: {e}")
        return

    bad = []
    # ① 行为断言：bf16 输入 -> fp32 输出，且形状正确
    y = _bilinear_upsample_fp32(torch.randn(2, 3, 7, 7, dtype=torch.bfloat16), (14, 14))
    if y.dtype != torch.float32:
        bad.append(f"bf16 输入得到 {y.dtype}，期望 float32")
    if tuple(y.shape) != (2, 3, 14, 14):
        bad.append(f"输出形状 {tuple(y.shape)}，期望 (2, 3, 14, 14)")
    # autocast 打开时也必须返回 fp32（helper 内部强制关闭 autocast）
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y2 = _bilinear_upsample_fp32(torch.randn(2, 3, 7, 7), (14, 14))
        if y2.dtype != torch.float32:
            bad.append(f"autocast 下得到 {y2.dtype}，期望 float32")
    except Exception as e:  # noqa: BLE001
        bad.append(f"autocast 路径异常：{type(e).__name__}: {e}")

    # ② 源码断言：不允许裸的 F.interpolate（helper 内部那次 x.float() 除外）
    #    注意：**必须用 AST，不能用正则** —— 该模块的文档字符串里就写着
    #    「`F.interpolate(mode="bilinear")` 不在 autocast 的 fp32 提升名单里」，
    #    正则会把这句说明当成一处调用（这正是"用 grep/正则判断源码"的经典坑）。
    import ast

    src_path = os.path.join(ROOT, "src", "models", "mobile_unetv2.py")
    src = open(src_path, encoding="utf-8").read()
    tree = ast.parse(src)
    bare, helper_calls = 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "interpolate" \
                and isinstance(fn.value, ast.Name) and fn.value.id == "F":
            # 唯一允许的形式：F.interpolate(x.float(), ...)
            arg0 = node.args[0] if node.args else None
            is_fp32 = (isinstance(arg0, ast.Call)
                       and isinstance(arg0.func, ast.Attribute)
                       and arg0.func.attr == "float")
            if not is_fp32:
                bare += 1
        elif isinstance(fn, ast.Name) and fn.id == "_bilinear_upsample_fp32":
            helper_calls += 1
    if bare:
        bad.append(f"mobile_unetv2.py 里有 {bare} 处裸的 F.interpolate（未强制 fp32）")
    if helper_calls != 2:
        bad.append(f"_bilinear_upsample_fp32 有 {helper_calls} 处调用，期望 2 处")

    # ③ 优化器路径：CUDA 上必须优先选 fused（每步少 ~400 次 .item() 同步）
    opt_src = open(os.path.join(ROOT, "src", "engine", "trainer.py"),
                   encoding="utf-8").read()
    if "fused=True" not in opt_src:
        bad.append("_optimizer 里没有 fused=True 分支")

    check("上采样强制 fp32（D43）", not bad,
          "; ".join(bad) if bad else "bf16/autocast 下均返回 fp32；2 处调用均经由 helper")


def test_limit_batches_keeps_full_val() -> None:
    """D43 附带：`--limit-batches` 只允许截训练集，**不能截验证集**。

    为什么必须钉住：验证集的顺序是 ForenSynths(val, 8000 张、无掩码) 在前、
    CASIAv2(val, 1261 张、带掩码) 在后，且 `shuffle=False` —— 一旦截断，
    前几批**必然**一张带掩码样本都没有，`validate()` 于是不产出 loc_* 指标，
    stage2 的 `monitor=val_loc_miou` 会直接抛 KeyError；而且截断验证集本身
    会让 best.pt 由 3% 的数据选出。所以这条检查"验证集没有被包进限流器"。
    """
    src = open(os.path.join(ROOT, "scripts", "train.py"), encoding="utf-8").read()
    bad = []
    if re.search(r'loaders\["val"\]\s*=\s*_Limited', src):
        bad.append('loaders["val"] 被 _Limited 截断了（会导致 val_loc_miou 缺失）')
    # 训练集仍应被限流（否则 --limit-batches 就是个摆设）
    if not re.search(r'loaders\["train"\]\s*=\s*_Limited', src):
        bad.append('loaders["train"] 没有被限流，--limit-batches 失效')
    check("--limit-batches 不截验证集（D43）", not bad, "; ".join(bad))


def test_check_backbone_tool() -> None:
    """骨干真伪检查脚本的可用性回归（纯子进程，秒级）。

    为什么挂在自检里：这个脚本是**唯一**能在烧 GPU 前拦住"静默兜底骨干"的东西。
    它一旦坏了（导入路径错、退出码语义变、模型属性改名），拦不住的那次训练
    会照常跑完并产出指标 —— 而且是错的。
    """
    script = os.path.join(ROOT, "scripts", "check_backbone.py")
    if not os.path.exists(script):
        check("骨干检查脚本存在", False, f"缺文件：{script}")
        return

    # 注意：不要用 subprocess.run(..., text=True) —— 中文 Windows 下按 GBK 解码，
    # 只要输出里有非 UTF-8 字节，读取线程就会抛 UnicodeDecodeError 而主线程拿到 None。
    p = subprocess.run([sys.executable, script, "--require-clip"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
    out = (p.stdout or b"").decode("utf-8", errors="replace")

    bad = []
    # 1) 退出码语义：0=真 CLIP / 2=路线 B / 3=非 OK 时被闸门拦下
    if p.returncode not in (0, 2, 3):
        bad.append(f"退出码 {p.returncode} 不在 {{0,2,3}} 内")
    # 2) 必须给出明确的三态结论，而不是含糊其辞
    if "[3] 结论" not in out:
        bad.append("没有打印结论段")
    verdicts = [k for k in ("[OK] 骨干 = 真 CLIP",
                            "[!] 骨干 = 离线兜底架构",
                            "[X] 骨干 = 随机初始化") if k in out]
    if len(verdicts) != 1:
        bad.append(f"三态结论不唯一（命中 {len(verdicts)} 条）")
    # 3) 闸门语义：只要不是真 CLIP，--require-clip 就必须返回 3
    if "[OK] 骨干 = 真 CLIP" not in out and p.returncode != 3:
        bad.append("非 CLIP 状态却未返回 3（闸门失效）")

    check("骨干真伪检查脚本（三态结论 + 闸门退出码）", not bad,
          "; ".join(bad) if bad else f"退出码 {p.returncode}，结论：{verdicts[0]}")


# ==========================================================================
def test_cloud_autodl_cli() -> None:
    """上云一键脚本的 CLI 回归（静态检查 + 一次轻量 fixture，秒级）。

    为什么挂在自检里：`scripts/cloud_autodl.sh` 是上云阶段唯一的操作入口，
    而它每次改动都很容易"看起来没问题"地坏掉 —— 更糟的是坏掉之后**报错在别处**
    （见 docs/08 D35~D39）。这里钉住三件最容易退化的东西：

      1) 九个 stage 都还在 `case` 分发里。少一个的话，那条命令不会报错，
         而是**静默地打印用法并以退出码 2 结束**（用户只会以为"没反应"）。
      2) `doctor` 在"项目正确 + 上级散装旧副本"时打印的归档命令，必须是
         **一整行可以复制粘贴**的。$dup 本身是多行，忘了 `tr '\\n' ' '`
         就会打出断行的废命令 —— 这正是实测抓到的 bug，所以值得钉住。
      3) `bash -n` 语法必须通过（这个脚本 700 行，改坏一处语法整条链全废）。
    """
    script = os.path.join(ROOT, "scripts", "cloud_autodl.sh")
    if not os.path.exists(script):
        check("上云一键脚本存在", False, f"缺文件：{script}")
        return

    bad = []

    # ---- 1) 语法 ----
    p = subprocess.run(["bash", "-n", script], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, cwd=ROOT)
    if p.returncode != 0:
        detail = (p.stdout or b"").decode("utf-8", errors="replace")[:200]
        bad.append(f"bash -n 未通过：{detail}")

    # ---- 2) stage 分发完整 ----
    src = open(script, encoding="utf-8").read()
    for st in ("doctor", "status", "install", "clip", "data",
               "verify", "probe", "train", "pack"):
        if not re.search(rf"^\s*{st}\)\s+stage_{st}\s*;;", src, re.M):
            bad.append(f"case 分发里没有 {st}")
        if f"stage_{st}()" not in src:
            bad.append(f"缺函数 stage_{st}()")
    if "stage_status" in src and "只读" not in src:
        bad.append("status 没有标明自己是只读的")

    # ---- 3) 归档命令必须可整行复制 ----
    with tempfile.TemporaryDirectory() as tmp:
        # 关键：夹具路径必须走 POSIX 风格正斜杠。Windows 的 tempfile 返回
        # `C:\Users\...\tmpXXX`，bash 里的 `[ -d ]` 不认反斜杠，夹具会静默失效
        # （第一次写这个测试就是这么假通过的）。
        tmp = tmp.replace("\\", "/")
        proj = f"{tmp}/vibnet-forgery-detector"
        os.makedirs(os.path.join(proj, "scripts"))
        # 上级目录里故意放散装的旧副本（模拟旧包被摊平解包）
        os.makedirs(os.path.join(tmp, "src"))
        os.makedirs(os.path.join(tmp, "scripts"))
        open(os.path.join(tmp, "scripts", "cloud_autodl.sh"), "w").close()

        env = dict(os.environ)
        env.update({"PROJ": proj, "DATA_ROOT": f"{tmp}/data",
                    "PY": sys.executable})
        try:
            q = subprocess.run(["bash", script, "doctor"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=ROOT, timeout=300)
            out = (q.stdout or b"").decode("utf-8", errors="replace")
            # 脚本用 ANSI 上色，`[OK]\033[0m 项目目录存在` 这样中间夹着转义序列，
            # 不剥掉就没法按 `[OK] 项目目录存在` 匹配（第一次就是这么假失败的）。
            out = re.sub(r"\x1b\[[0-9;]*m", "", out)
        except Exception as e:                      # noqa: BLE001
            out = ""
            bad.append(f"doctor 跑不起来：{type(e).__name__}")

    if "[OK] 项目目录存在" not in out:
        bad.append("夹具没被识别成「项目已就位」，后面的判据全部不可信")
    if "还散落着同名的旧副本" not in out:
        bad.append("没检测到散装旧副本（D39 的检测失效）")
    mv_lines = [l for l in out.split("\n") if "mkdir -p" in l and "_stale_code" in l]
    if not mv_lines:
        bad.append("没打印可用的归档命令")
    elif sum("mv " in l for l in mv_lines) != 1:
        bad.append("归档命令里 mv 缺失")

    check("上云一键脚本（stage 分发 / 语法 / 归档命令可整行复制）", not bad,
          "; ".join(bad) if bad else "9 个 stage 齐全，语法通过，归档命令为单行")


# ==========================================================================
def test_robustness_ops() -> None:
    """鲁棒性后处理算子的可调用性与形状自检。

    ★ 回归背景：`src/evaluation/robustness.py` 曾同时存在两个缺陷，导致该脚本
    **在任何输入下都跑不通**，而它不在任何自检路径上，静默坏了很久：
      ① 首个算子误写成 `op_identity()`（把函数当值调用）→ 构造列表即 TypeError；
      ② 椒盐噪声把形状 (n,) 的噪声值赋给彩色图的 `out[ys, xs]`（形状 (n,3)）→ ValueError。
    本测试对每个算子跑一张合成图，把这两类缺陷一次性钉住。
    """
    import numpy as np

    try:
        from src.evaluation import robustness as rb
    except Exception as e:  # pragma: no cover
        check("robustness 模块可导入", False, f"{type(e).__name__}: {e}")
        return

    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    try:
        ops = rb.build_ops(cfg)
    except Exception as e:
        check("鲁棒性算子可构造", False, f"{type(e).__name__}: {e}")
        return

    bad = []
    # 1 + jpeg6 + scale5 + crop5 + rotate4 + gauss3 + saltpepper3 = 27
    if len(ops) != 27:
        bad.append(f"算子数 {len(ops)} != 27")

    rng = np.random.RandomState(3407)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
    for name, op in ops:
        if not callable(op):
            bad.append(f"{name}: 不是可调用对象")
            continue
        try:
            out = op(img.copy())
        except Exception as e:
            bad.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if out.shape != img.shape or out.dtype != np.uint8:
            bad.append(f"{name}: 形状/类型被改变 {out.shape}/{out.dtype}")
    check("鲁棒性 27 个后处理算子全部可用（含椒盐噪声形状）", not bad,
          "；".join(bad[:5]))


def test_empty_batch_guard(cfg: dict) -> None:
    """空监督批次（整批不带掩码）不得把训练崩在 loss.backward()。

    ★ 回归背景（2026-09-25 云上 stage2 第 1 个 epoch 实测，训练 103 分钟后死掉）
      stage2 只启用 loc/edge，这两个任务的监督**全部来自像素掩码**；而训练集是
      ForenSynths（整图真伪，无掩码，40000 张）+ CASIAv2/COVERAGE（带掩码，6120 张）
      的混合集 —— 掩码样本占比约 13%，于是 batch=16 时约 (1-0.133)^16 ≈ 10% 的
      批次**一张带掩码的样本都没有**。此时 MultiTaskLoss 的 raw 为空，
      而 UncertaintyWeighting 的 `total` 停在 Python 浮点 0.0（不是张量），
      训练循环走到 `loss.backward()` 抛
          AttributeError: 'float' object has no attribute 'backward'
      —— 修好"早停掀翻后续阶段"之后**第一次真正跑进 stage2**，就撞上它。

    本用例把这个"数据分布 + 损失的耦合"一次性钉住：空批次必须返回**可 backward
    的零张量**并标记 empty=True，正常批次必须完全不受影响。
    """
    print("\n【8】空监督批次守卫（整批无掩码不得崩在 loss.backward()）")
    cfg["model"]["spatial"]["backbone"] = "tiny_vit"
    cfg["train"]["amp"] = False
    crit = MultiTaskLoss(cfg)

    bad = []
    # ---- ① 整批无掩码 + 只启用定位任务：三种任务组合都必须给可 backward 的张量
    for tasks in (["loc", "edge"], ["loc"], ["edge"]):
        model = build_model(cfg).train()
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 0, 1])
        out = model(x, sample_vib=True, beta=0.0)          # 不包 no_grad：要留计算图
        lo = crit(out, y, None, beta=0.0, active_tasks=tasks, update_norm=True,
                  mask_valid=torch.zeros(4, dtype=torch.bool))
        why = []
        if not isinstance(lo["total"], torch.Tensor):
            why.append(f"total 不是张量而是 {type(lo['total']).__name__}")
        if lo["empty"] is not True:
            why.append(f"empty={lo['empty']}（应为 True）")
        try:
            lo["total"].backward()
        except Exception as e:                                  # noqa: BLE001
            why.append(f"backward() 抛 {type(e).__name__}: {e}")
        print(f"      任务={str(tasks):22s} raw={lo['raw']} empty={lo['empty']} "
              f"total={type(lo['total']).__name__}")
        if why:
            bad.append(f"{tasks}: " + "；".join(why))

    check("整批无掩码 → 零张量 + empty=True + backward() 不抛异常", not bad,
          "；".join(bad) if bad else "三种任务组合均通过")

    # ---- ② 有掩码的批次必须完全不受影响（四项都在、能反向、梯度非全零）
    model = build_model(cfg).train()
    crit = MultiTaskLoss(cfg)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.zeros(4, 1, 224, 224)
    mask[1, :, 60:140, 60:140] = 1
    mask[3] = 1
    mvalid = torch.tensor([False, True, False, True])
    out = model(x, sample_vib=True, beta=0.05)
    lo = crit(out, y, mask, beta=0.05, active_tasks=["loc", "edge"],
              update_norm=True, mask_valid=mvalid)
    ok2 = (len(lo["raw"]) == 3 and lo["empty"] is False)
    try:
        lo["total"].backward()
        g = sum(float(p.grad.abs().sum()) for p in model.parameters()
                if p.requires_grad and p.grad is not None)
        ok2 = ok2 and g > 0
        detail = f"raw={sorted(lo['raw'])} 梯度绝对值和={g:.3e}"
    except Exception as e:                                      # noqa: BLE001
        ok2 = False
        detail = f"backward() 抛 {type(e).__name__}: {e}"
    check("有掩码批次不受影响（3 项损失 + 反向出非零梯度）", ok2, detail)

    # ---- ③ 训练循环：空批次必须被跳过（不计入平均损失、不更新参数）
    from src.engine.trainer import StageConfig

    with tempfile.TemporaryDirectory() as td:
        c = {**cfg,
             "project": {**cfg["project"], "output_dir": td,
                         "ckpt_dir": os.path.join(td, "ckpt")}}

        no_mask = {"mask": None, "mask_valid": torch.zeros(4, dtype=torch.bool)}
        has_mask = dict(no_mask)
        has_mask["mask"] = torch.zeros(4, 1, 224, 224)
        has_mask["mask_valid"] = torch.tensor([True, True, False, False])

        class _FakeLoader:
            def __init__(self, kinds):
                self.kinds = kinds

            def __iter__(self):
                for has in self.kinds:
                    b = has_mask if has else no_mask
                    yield {"image": torch.randn(4, 3, 224, 224),
                           "label": torch.tensor([0, 1, 0, 1]), **b}

            def __len__(self):
                return len(self.kinds)

        # 3 个空批次 + 1 个有效批次
        tm = build_model(c).train()
        tr = Trainer(c, tm, torch.device("cpu"), _FakeLoader([0, 0, 0, 1]),
                     val_loader=None, logger=TrainLogger(None))
        st2 = StageConfig.from_dict({"name": "s2", "epochs": 1, "lr": 1e-4,
                                     "use_tasks": ["loc", "edge"], "freeze": ["vib", "cls_head"],
                                     "train": ["localization"]})
        stats = tr._train_one_epoch(tr._optimizer(st2), st2, 0.0, 0)
        ok3 = stats["n_batches"] == 1 and stats["empty_batches"] == 3
        check("空批次被跳过且不计入平均损失", ok3,
              f"有效批次={stats['n_batches']} 空批次={stats['empty_batches']} "
              f"loss={stats['loss']:.4f}")

        # 分类阶段不该出现空批次（vib 项与掩码无关）
        tm2 = build_model(c).train()
        tr2 = Trainer(c, tm2, torch.device("cpu"), _FakeLoader([0, 0, 0, 1]),
                      val_loader=None, logger=TrainLogger(None))
        st1 = StageConfig.from_dict({"name": "s1", "epochs": 1, "lr": 1e-4,
                                     "use_tasks": ["cls"], "freeze": ["localization"],
                                     "train": ["all"]})
        stats1 = tr2._train_one_epoch(tr2._optimizer(st1), st1, 0.0, 0)
        check("只训分类时无掩码也不算空批次",
              stats1["n_batches"] == 4 and stats1["empty_batches"] == 0,
              f"有效批次={stats1['n_batches']} 空批次={stats1['empty_batches']}")

        # ---- ④ 整个 epoch 全空 → 必须硬失败，且报错里要点出"掩码数据集没加载"
        tm3 = build_model(c).train()
        tr3 = Trainer(c, tm3, torch.device("cpu"), _FakeLoader([0, 0, 0, 0]),
                      val_loader=None, logger=TrainLogger(None))
        # 只改副本，别污染 cfg 里共享的 stages（后面的用例还要用）
        c["train"]["stages"] = [dict(s, epochs=1) for s in cfg["train"]["stages"]]
        tr3.stages = [StageConfig.from_dict(s) for s in c["train"]["stages"]]
        msg = ""
        try:
            tr3.fit()
            ok4 = False
        except RuntimeError as e:
            msg = str(e)
            ok4 = ("没有任何一个批次含有效监督" in msg and "掩码" in msg)
        except Exception as e:                                  # noqa: BLE001
            ok4 = False
            msg = f"{type(e).__name__}: {e}"
        check("整轮全空 → 硬失败并指出掩码数据集未加载", ok4, msg[:120])


# ==========================================================================
def test_start_stage_resume(cfg: dict) -> None:
    """`--start-stage` / `fit(start_stage=N)` 必须跳过已完成的阶段。

    ★ 为什么值得单独立用例：stage1 分类预训练在 4090 上实测约 100 分钟（≈¥3）。
      修完早停 bug 后重跑时，stage1 已经跑完并选出了 best.pt，却因为 `--resume`
      只加载权重、不推进阶段与轮次，会被整个重跑一遍。
      跳过阶段时必须同步把 global_epoch 顶到前面阶段的总轮数上，否则
      beta 退火（公式 24：t<20 时 β=0，t≥40 才到 0.1）会按 t=0 重新起算，
      阶段三的 β 永远升不起来 —— 这种错不报错、只是"结果悄悄变差"。
    """
    print("\n【9】跳过已完成阶段续训（--start-stage）")
    cfg["model"]["spatial"]["backbone"] = "tiny_vit"
    cfg["train"]["amp"] = False

    # 阶段轮数压到最小：stage1=3 轮（跳过时要累加的就是它）
    epochs = [3, 1, 1]
    with tempfile.TemporaryDirectory() as td:
        c = {**cfg,
             "project": {**cfg["project"], "output_dir": td,
                         "ckpt_dir": os.path.join(td, "ckpt")},
             "train": {**cfg["train"], "amp": False}}
        c["train"]["stages"] = [dict(s) for s in cfg["train"]["stages"]]   # 改副本
        for s, e in zip(c["train"]["stages"], epochs):
            s["epochs"] = e
        c["train"]["stages"][0]["freeze"] = ["localization"]
        c["train"]["stages"][1]["freeze"] = ["vib", "cls_head"]
        c["train"]["stages"][2]["freeze"] = []

        class _FakeLoader:
            """混合批次：一个带掩码、一个不带（模拟 ForenSynths + CASIA 混采）"""

            def __iter__(self):
                yield {"image": torch.randn(4, 3, 224, 224),
                       "label": torch.tensor([0, 1, 0, 1]),
                       "mask": torch.zeros(4, 1, 224, 224),
                       "mask_valid": torch.tensor([True, True, False, False])}
                yield {"image": torch.randn(4, 3, 224, 224),
                       "label": torch.tensor([0, 1, 0, 1]),
                       "mask": None,
                       "mask_valid": torch.zeros(4, dtype=torch.bool)}

            def __len__(self):
                return 2

        m = build_model(c).train()
        tr = Trainer(c, m, torch.device("cpu"), _FakeLoader(),
                     val_loader=None, logger=TrainLogger(None))
        hist = tr.fit(start_stage=1)
        stages_ran = {r["stage"] for r in hist}
        first_epoch = hist[0]["epoch"] if hist else -1
        ok = (stages_ran == {"stage2_loc_pretrain", "stage3_joint_finetune"}
              and first_epoch == epochs[0])
        check("start_stage=1 只跑 stage2/stage3，且 global_epoch 承接 stage1 的轮数",
              ok, f"跑过的阶段={sorted(stages_ran)}，stage2 首个 epoch={first_epoch}"
                  f"（应为 {epochs[0]}）")
        ck_last = os.path.join(c["project"]["ckpt_dir"], "vibnet_last.pt")
        check("跳阶段后仍正常落 last.pt", os.path.exists(ck_last), ck_last)

        # 越界与非法名字必须被拦住，而不是静默从头训
        m2 = build_model(c).train()
        tr2 = Trainer(c, m2, torch.device("cpu"), _FakeLoader(),
                      val_loader=None, logger=TrainLogger(None))
        hist2 = tr2.fit(start_stage=99)          # 越界 → 收敛到最后一个阶段
        check("start_stage 越界时收敛到最后一个阶段（不静默从头训）",
              {r["stage"] for r in hist2} == {"stage3_joint_finetune"},
              f"实际跑了 {sorted({r['stage'] for r in hist2})}")

    # CLI 侧：名字解析与错误提示（不跑训练，只验证参数校验分支）
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                        "--start-stage", "bogus_stage"],
                       capture_output=True, cwd=ROOT)
    blob = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    check("--start-stage 写错名字时报错并列出可选值",
          r.returncode != 0 and "既不是序号也不是阶段名" in blob
          and "stage2_loc_pretrain" in blob,
          f"returncode={r.returncode}")

    script = open(os.path.join(ROOT, "scripts", "cloud_autodl.sh"),
                  encoding="utf-8").read()
    check("上云脚本透传额外参数（train --resume ... --start-stage ...）",
          "TRAIN_EXTRA" in script and "--start-stage" in script
          and "$PY -u scripts/train.py" in script,
          "cloud_autodl.sh 已支持把附加参数原样传给 train.py")


def test_backbone_freeze_policy(cfg: dict) -> None:
    """`apply_stage()` 之后必须重放骨干冻结策略；`--cudnn` 开关必须真的可用。

    ★ 2026-09-26 发现：`CLIPViTBackbone.__init__` 按申报书 1.1(1) 冻好的前 6 层，
      会被 `VIBNet.apply_stage()` 的 `set_requires_grad(["all"], True)` **全部解冻**，
      于是 `spatial.freeze_first_n_layers: 6` 在整个训练里形同虚设，而且**不报错**：
      启动日志是"冻结前 6/12 层，可训练参数 42.53M / 85.80M"（骨干口径、构造那一刻），
      进 stage1 后却变成"可训练参数 86.84M / 总计 90.53M"——相隔几十行的两个数字
      自相矛盾，只有把它们对起来看才会发现。连带后果：docs/00、docs/04、docs/08-C5
      里"分类 ACC 卡 50% 就先检查 freeze_first_n_layers"这条排查建议是**死的**。
    """
    print("\n【10】骨干冻结策略在 apply_stage 之后仍生效（申报书 1.1(1)：冻结前 6 层）")
    cfg["model"]["spatial"]["backbone"] = "tiny_vit"
    net = build_model(cfg)
    bb = net.spatial.backbone

    # ① 每个阶段都必须重放一次（阶段数 == 调用次数）
    calls = {"n": 0}
    orig = bb._apply_freeze_policy

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    bb._apply_freeze_policy = _spy
    stages = list(cfg["train"]["stages"])
    for st in stages:
        net.apply_stage(list(st.get("freeze", [])), list(st.get("train", ["all"])))
    check("每个阶段都重放骨干冻结策略（3 个阶段 → 3 次）",
          calls["n"] == len(stages), f"重放 {calls['n']} 次 / 阶段 {len(stages)} 个")

    # ② 兜底骨干走的是"全解冻"分支，测不出策略本身是否生效 —— 装一个最小 CLIP 形状
    #    替身（只保留 vision_model.encoder.layers + 两个 layernorm），
    #    让 _apply_freeze_policy 走真实分支，且不需要真的下载 CLIP 权重。
    class _Blk(nn.Module):
        def __init__(self, d: int = 8):
            super().__init__()
            self.lin = nn.Linear(d, d)

    class _VM(nn.Module):
        def __init__(self, n: int = 12, d: int = 8):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([_Blk(d) for _ in range(n)])
            self.post_layernorm = nn.LayerNorm(d)
            self.pre_layrnorm = nn.LayerNorm(d)

    class _FakeCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = _VM()

    bb.is_fallback = False
    bb.freeze_first_n = 6
    bb.unfreeze_last_n = 6
    bb.clip = _FakeCLIP()
    net.apply_stage([], ["all"])          # stage3：train=[all]，最容易被"全解冻"带走
    layers = bb.clip.vision_model.encoder.layers
    low_frozen = all(not p.requires_grad for p in layers[:6].parameters())
    high_open = all(p.requires_grad for p in layers[6:].parameters())
    check("train=[all] 的 stage3 也不会解冻骨干前 6 层", low_frozen)
    check("同时骨干后 6 层仍可训练（没把整支冻死）", high_open)

    # ③ 卷积后端开关：--cudnn / --amp-dtype 必须**真的落到运行期状态**上。
    #    只 add_argument 不消费 = 探针对照实验给出的结论是反向的（docs/08 D42）。
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                        "--cudnn", "bogus"], capture_output=True, cwd=ROOT)
    out = ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "replace")
    check("--cudnn 非法取值被拒绝（choices 生效）",
          r.returncode != 0 and "cudnn" in out.lower())

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_train_entry", os.path.join(ROOT, "scripts", "train.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)        # 有 __main__ 守卫，import 不会触发训练
    keep_enabled, keep_bench = torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark
    try:
        c2 = {"train": {"amp": True, "amp_dtype": "bfloat16"}}
        _mod.apply_runtime_flags(c2, "off", "float16")
        check("--cudnn off 真的把 cuDNN 关了（不只是被 argparse 收下）",
              torch.backends.cudnn.enabled is False)
        check("--amp-dtype float16 真的改写了 cfg.train.amp_dtype",
              c2["train"]["amp_dtype"] == "float16")
        _mod.apply_runtime_flags(c2, "bench", None)
        check("--cudnn bench 真的打开了 benchmark 自动调优",
              torch.backends.cudnn.benchmark is True)
        # 复位成"默认开启"再验 "on"：on 是默认态，正确行为是**什么都不做**
        torch.backends.cudnn.enabled = True
        _mod.apply_runtime_flags(c2, "on", None)
        check("--cudnn on 不误改全局后端状态（保持默认开启）",
              torch.backends.cudnn.enabled is True)
    finally:
        torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark = keep_enabled, keep_bench

    sh_path = os.path.join(ROOT, "scripts", "cloud_autodl.sh")
    sh = open(sh_path, encoding="utf-8").read() if os.path.exists(sh_path) else ""
    check("上云脚本把 --cudnn $CUDNN 透传给 train.py", "--cudnn $CUDNN" in sh)
    # runner 用 <<EOF 在生成时展开，CUDNN 未设置就变空串 → 行尾只剩裸的
    # `--cudnn` → argparse "expected one argument"，启动即死。
    check("上云脚本给 CUDNN 兜了默认值（否则 --cudnn 会收到空参数）",
          re.search(r'CUDNN="\$\{CUDNN:-on\}"', sh) is not None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="跑完整流水线（含训练与导出）")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/default.yaml"))
    args = ap.parse_args()

    print("=" * 74)
    print("空频双分支轻量 VIB-Net —— 端到端冒烟测试")
    print("=" * 74)
    torch.manual_seed(3407)

    cfg = load_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["train"]["batch_size"] = 4

    test_modules(cfg)
    test_end_to_end(cfg)
    test_metric_definitions()
    test_audit_tools()
    test_leak_audit()
    test_fetch_extract()
    test_amp_policy()
    test_amp_upsample_fp32()
    test_limit_batches_keeps_full_val()
    test_check_backbone_tool()
    test_cloud_autodl_cli()
    test_robustness_ops()
    test_empty_batch_guard(cfg)
    test_start_stage_resume(cfg)
    test_backbone_freeze_policy(cfg)
    if args.full:
        test_full_pipeline(cfg)

    n_ok = sum(1 for _, c in results if c)
    n_all = len(results)
    print("\n" + "=" * 74)
    print(f"测试结果：{n_ok}/{n_all} 通过")
    fails = [n for n, c in results if not c]
    if fails:
        print("未通过项：")
        for n in fails:
            print(f"  - {n}")
    print("=" * 74)
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
