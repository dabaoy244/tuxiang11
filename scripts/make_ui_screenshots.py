#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离屏渲染桌面工具界面截图（用于软件著作权说明书、答辩 PPT、技术报告）。

为什么要拆成两个阶段
--------------------
PyQt5 的 offscreen 平台插件与 PyTorch 在同一进程里加载模型时会 **段错误**
（本机实测 exit 139）。因此本脚本把流程拆成两个**独立进程**：

    阶段 1 `--stage detect`  纯 PyTorch/ONNX，无 Qt —— 跑推理，结果 pickle 存盘
    阶段 2 `--stage shots `  纯 Qt，无 torch     —— 构建界面、灌入结果、截图

另外 Qt 报 `Cannot find font directory` 时中文会渲染成方块，
本脚本会把 `QT_QPA_FONTDIR` 指向系统字体目录，并补一份字体到 Qt 期望的位置。

用法：
    python scripts/make_ui_screenshots.py                      # 两阶段自动串起来
    python scripts/make_ui_screenshots.py --backend onnx       # 用 ONNX 后端（更快）
    python scripts/make_ui_screenshots.py --n-real 4 --n-fake 4
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = sys.executable


# ----------------------------------------------------------------------
def setup_qt_fonts(verbose: bool = True) -> None:
    """让 offscreen 平台插件能找到中文字体，否则中文会渲染成空白方块。"""
    win_fonts = r"C:\Windows\Fonts"
    os.environ.setdefault("QT_QPA_FONTDIR", win_fonts)

    # Qt 还会去 <PyQt5>/Qt5/lib/fonts 找，补一份过去
    try:
        import PyQt5
        qt_font_dir = os.path.join(os.path.dirname(PyQt5.__file__),
                                   "Qt5", "lib", "fonts")
        if not os.path.isdir(qt_font_dir) or not os.listdir(qt_font_dir):
            os.makedirs(qt_font_dir, exist_ok=True)
            copied = []
            for name in ("msyh.ttc", "msyhbd.ttc", "simfang.ttf", "Deng.ttf",
                         "simsun.ttc", "simhei.ttf", "consola.ttf"):
                src = os.path.join(win_fonts, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(qt_font_dir, name))
                    copied.append(name)
            if verbose and copied:
                print(f"  [font] 已补充 {len(copied)} 个字体到 {qt_font_dir}")
    except Exception as e:                                             # noqa: BLE001
        if verbose:
            print(f"  [font] 补充字体失败（可忽略）：{type(e).__name__}: {e}")


def pick_samples(root: str, n_real: int, n_fake: int) -> list:
    """从 ForenSynths val 里挑真实/伪造样本各若干张。

    注意：ForenSynths 每个类别目录下的文件名是**一样的**（都是 00091.png 这种），
    所以不能简单地「排序后等间隔取样」，否则会取到一堆同名文件。
    这里改成按类别轮流取，并且每个类别取的下标不同，保证文件名与类别都分散。
    """
    from collections import OrderedDict
    from glob import glob

    base = os.path.join(root, "ForenSynths", "val")
    if not os.path.isdir(base):
        base = os.path.join(root, "ForenSynths")

    def spread(pattern: str, n: int) -> list:
        groups = OrderedDict()
        for p in sorted(glob(pattern, recursive=True)):
            groups.setdefault(os.path.dirname(p), []).append(p)
        if not groups:
            return []
        out, picked = [], set()
        keys = list(groups)
        i = 0
        while len(out) < n and i < 300:
            # ★ 每个类别用不同的下标偏移，否则各类别第一个文件同名（都是 00091.png）
            for gi, k in enumerate(keys):
                if len(out) >= n:
                    break
                g = groups[k]
                idx = (i * 13 + gi * 7) % len(g)
                cand = g[idx]
                if cand in picked:
                    continue
                picked.add(cand)
                out.append(cand)
            i += 1
        return out

    return spread(os.path.join(base, "**", "0_real", "*.png"), n_real) + \
        spread(os.path.join(base, "**", "1_fake", "*.png"), n_fake)


# ======================================================================
#  阶段 1：纯推理（无 Qt）
# ======================================================================
def stage_detect(args) -> int:
    samples = pick_samples(args.data, args.n_real, args.n_fake)
    if len(samples) < 2:
        print("[detect] 样本不足，请先运行 "
              "scripts/fetch_datasets.py --split val")
        return 2
    print(f"[detect] 样本 {len(samples)} 张，后端={args.backend}")

    from src.deploy.inference import Detector
    kwargs = dict(backend=args.backend, image_size=args.image_size,
                  threshold=args.threshold, config_path=args.config)
    if args.backend == "onnx":
        kwargs["onnx_path"] = args.onnx
    det = Detector(**kwargs)
    note = getattr(det, "_fallback_note", None)
    if note:
        print(f"[detect] 提示：{note}")

    results = []
    for i, p in enumerate(samples, 1):
        try:
            r = det.detect(p)
            results.append(r)
            print(f"  [{i}/{len(samples)}] {os.path.basename(p):16s} "
                  f"{r.label}  p={r.prob_fake:.4f}  {r.time_ms:.0f} ms")
        except Exception as e:                                         # noqa: BLE001
            print(f"  [{i}/{len(samples)}] {os.path.basename(p)} 失败："
                  f"{type(e).__name__}: {e}")

    if not results:
        print("[detect] 全部失败")
        return 3

    # ★ 只存纯 Python 字典 + numpy 数组，不要 pickle 自定义类。
    #   否则界面进程 unpickle 时会连带 import src.deploy → import torch，
    #   而 torch 与 Qt offscreen 插件在本环境不能共存（c10.dll 初始化失败 / 段错误）。
    payload = [{
        "path": r.path, "name": r.name,
        "width": int(r.width), "height": int(r.height),
        "is_fake": bool(r.is_fake), "prob_fake": float(r.prob_fake),
        "mask": r.mask, "overlay": r.overlay,
        "time_ms": float(r.time_ms), "backend": str(r.backend),
        "threshold": float(r.threshold), "warnings": list(r.warnings),
    } for r in results]
    with open(args.results, "wb") as f:
        pickle.dump({"results": payload, "backend": det.backend,
                     "threshold": args.threshold}, f)
    print(f"[detect] 结果已存 {args.results}（{len(payload)} 条）")
    return 0


# ======================================================================
#  阶段 2：纯界面（无 torch 加载）
# ======================================================================
def shot(widget, path: str, note: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm = widget.grab()
    ok = pm.save(path)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    flag = "OK" if ok and size > 3000 else "检查"
    print(f"  [shot] {os.path.basename(path):32s} {pm.width()}x{pm.height()}  "
          f"{size/1024:6.0f} KB  {flag}  {note}")


def stage_shots(args) -> int:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    setup_qt_fonts()

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    # ★ 关键：屏蔽主窗口的自动加载模型（会在 Qt 进程里加载 torch → 段错误）
    import app.main as app_main
    app_main.MainWindow.reload_detector = lambda self: None

    samples = pick_samples(args.data, args.n_real, args.n_fake)
    with open(args.results, "rb") as f:
        blob = pickle.load(f)

    # 界面进程里不能 import src.deploy（会拉起 torch），
    # 所以这里用本地轻量类复刻 DetectionResult 的接口。
    class _Res:
        def __init__(self, d: dict):
            self.__dict__.update(d)

        @property
        def label(self) -> str:
            return "疑似伪造" if self.is_fake else "判定为真实"

        def summary(self) -> dict:
            return {
                "文件名": self.name,
                "图像尺寸": f"{self.width}x{self.height}",
                "检测结论": self.label,
                "伪造置信度": round(self.prob_fake, 4),
                "判定阈值": self.threshold,
                "高置信度区域占比": round(float((self.mask > 0.5).mean()), 4),
                "推理耗时(ms)": round(self.time_ms, 2),
                "推理后端": self.backend,
            }

    results = [_Res(d) for d in blob["results"]]
    print(f"[shots] 载入 {len(results)} 条检测结果（后端={blob['backend']}）")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    win = app_main.MainWindow()
    win.resize(1360, 860)
    win.show()
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    # 装作模型已加载
    class _FakeDet:
        backend = blob["backend"]
        threshold = blob["threshold"]
    win.detector = _FakeDet()
    win.settings_panel.set_status(f"已加载 · {blob['backend']}", "#1e8e3e")
    win.log(f"模型已加载，后端={blob['backend']}，阈值={blob['threshold']}", "#1e8e3e")
    app.processEvents()

    # ---- 01 初始界面
    shot(win, os.path.join(args.out, "01_主界面_初始状态.png"), "五大模块布局")

    # ---- 02 导入图像
    if samples:
        win.input_panel.add_paths(samples)
        app.processEvents()
    shot(win, os.path.join(args.out, "02_图像输入.png"),
         f"{len(samples)} 张（真实/伪造混合）")
    shot(win.input_panel, os.path.join(args.out, "02b_输入面板_特写.png"),
         "缩略图列表")

    # ---- 03 检测结果（热力图 + 报告）
    win.result_panel.clear()
    win.progress.setMaximum(len(results))
    for i, r in enumerate(results, 1):
        win.result_panel.add_row(r)
    win.progress.setValue(len(results))
    win.progress.setFormat(f"%v/%m  完成")
    c = win.result_panel.summary_counts
    win.log(f"检测完成：共 {c['total']} 张，疑似伪造 {c['fake']} 张，"
            f"真实 {c['real']} 张", "#1e8e3e")
    win.result_panel.tabs.setCurrentIndex(0)
    if results:
        win.result_panel.show_result(results[0])
    app.processEvents()
    time.sleep(0.3)
    app.processEvents()

    shot(win, os.path.join(args.out, "03_检测结果_热力图与报告.png"),
         "原图/热力图对照 + 判定报告")
    shot(win.result_panel, os.path.join(args.out, "03b_结果面板_特写.png"),
         "结果面板")

    # ---- 04 批量结果表格
    win.result_panel.tabs.setCurrentIndex(1)
    app.processEvents()
    time.sleep(0.3)
    app.processEvents()
    shot(win, os.path.join(args.out, "04_批量结果表格.png"), "批量检测汇总表")

    # ---- 05 PDF 报告示例（转成 PNG）
    try:
        from app.modules.exporter import export_pdf_report
        pdf = args.pdf_demo or os.path.join(args.out, "_demo_report.pdf")
        export_pdf_report(results, pdf)
        print(f"  [pdf ] 示例报告 {pdf}")
        import pymupdf
        with pymupdf.open(pdf) as doc:
            for i in range(min(2, doc.page_count)):
                pix = doc[i].get_pixmap(dpi=110)
                p = os.path.join(args.out, f"05_检测报告_P{i+1}.png")
                pix.save(p)
                print(f"  [shot] {os.path.basename(p):32s} "
                      f"{pix.width}x{pix.height}  {os.path.getsize(p)/1024:6.0f} KB")
    except Exception as e:                                             # noqa: BLE001
        print(f"  [warn] 报告导出/渲染失败：{type(e).__name__}: {e}")

    # ---- 06 系统设置特写
    win.settings_panel.setMaximumHeight(16777215)
    app.processEvents()
    shot(win.settings_panel, os.path.join(args.out, "06_系统设置.png"),
         "后端/权重路径/阈值")

    # ---- 07 检测完成后的全窗口
    shot(win.centralWidget(), os.path.join(args.out, "07_全窗口_检测完成后.png"),
         "完整界面")

    print(f"\n[shots] 输出目录 {args.out}")
    return 0


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="离屏渲染桌面工具界面截图",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "Datasets"))
    ap.add_argument("--config", default="configs/lite.yaml")
    ap.add_argument("--backend", default="onnx", choices=["torch", "onnx", "openvino"])
    ap.add_argument("--onnx", default="deploy/lite_simplified.onnx")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "copyright", "ui_shots"))
    ap.add_argument("--results", default=os.path.join(ROOT, "outputs", "ui_shot_results.pkl"))
    ap.add_argument("--pdf-demo", default="")
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--n-fake", type=int, default=4)
    ap.add_argument("--stage", default="auto",
                    choices=["auto", "detect", "shots"],
                    help="auto=两阶段自动串联（推荐）")
    args = ap.parse_args()

    if args.stage == "detect":
        return stage_detect(args)
    if args.stage == "shots":
        return stage_shots(args)

    # ---- auto：用子进程分两阶段跑，避免 Qt 与 torch 同进程崩溃
    print("[1/2] 阶段一：推理（纯 torch/onnx，无 Qt）")
    r = subprocess.run([PY, "-u", os.path.abspath(__file__),
                        "--stage", "detect",
                        "--data", args.data, "--config", args.config,
                        "--backend", args.backend, "--onnx", args.onnx,
                        "--image-size", str(args.image_size),
                        "--threshold", str(args.threshold),
                        "--results", args.results,
                        "--n-real", str(args.n_real), "--n-fake", str(args.n_fake)])
    if r.returncode != 0:
        print(f"[abort] 推理阶段失败（exit {r.returncode}）")
        return r.returncode

    print("\n[2/2] 阶段二：构建界面并截图（纯 Qt，不加载 torch）")
    return subprocess.run([PY, "-u", os.path.abspath(__file__),
                           "--stage", "shots",
                           "--data", args.data,
                           "--backend", args.backend,
                           "--threshold", str(args.threshold),
                           "--out", args.out, "--results", args.results,
                           "--pdf-demo", args.pdf_demo,
                           "--n-real", str(args.n_real),
                           "--n-fake", str(args.n_fake)]).returncode


if __name__ == "__main__":
    sys.exit(main())
