#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练历史画成论文/报告用的曲线图（SVG + PNG）。

为什么要自己画而不用 matplotlib
--------------------------------
本项目的运行环境（无 GPU 的 Windows + 隔离 venv）**装不上 matplotlib**
（pip 取不到包）。所以这里走「纯 Python 拼 SVG → 用 Qt 光栅化成 PNG」的路线，
零第三方依赖，且 SVG 是矢量图，插进 Word / LaTeX 里放大不糊。

用法：
    python scripts/plot_training_curves.py                       # 默认读 outputs/run_realval/history.json
    python scripts/plot_training_curves.py --history outputs/run_lite/history.json --out docs/figs
    python scripts/plot_training_curves.py --png-only            # 只出 PNG

输出（默认写到 --out 目录）：
    train_curves.svg    矢量图，直接插论文
    train_curves.png    位图，插 PPT / 结题报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- 主题（浅色）
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BG = "#ffffff"
COLORS = {
    "train_loss": "#d9534f",
    "val_loss": "#f0ad4e",
    "val_cls_acc": "#2f7ed8",
    "val_cls_auc": "#3a9e57",
    "base": "#9aa0a6",
}
FONT = "Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"


# ================================================================== 工具
def nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    """给出 n+1 个好看的刻度值（1/2/5 × 10^k 步长）。"""
    if not (hi > lo):
        hi = lo + 1.0
    raw = (hi - lo) / max(1, n)
    mag = 10.0 ** int(__import__("math").floor(__import__("math").log10(raw)))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = __import__("math").floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks


def fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.2f}"


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ================================================================== SVG 画布
class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts: list = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill=None, stroke=None, sw=1, rx=0) -> None:
        a = f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"'
        if rx:
            a += f' rx="{rx}"'
        a += f' fill="{fill or "none"}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{sw}"'
        self.add(a + "/>")

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1, dash=None) -> None:
        a = (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
             f'stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        self.add(a + "/>")

    def poly(self, pts, stroke, sw=2, dash=None) -> None:
        if not pts:
            return
        d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        a = (f'<polyline points="{d}" fill="none" stroke="{stroke}" '
             f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        self.add(a + "/>")

    def circle(self, cx, cy, r, fill, stroke=None) -> None:
        a = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="1.5"'
        self.add(a + "/>")

    def text(self, x, y, s, size=13, fill=INK, anchor="start", weight="normal") -> None:
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}" '
                 f'font-family="{FONT}">{esc(s)}</text>')

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="{self.w}" height="{self.h}" fill="{BG}"/>'
                + "".join(self.parts) + "</svg>")


# ================================================================== 单面板
def _legend_width(series: list) -> float:
    """估算图例总宽度（中文字符按整字宽算）。"""
    w = 0.0
    for s in series:
        n = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in str(s["label"]))
        w += 26 + n * 12 + 22
    return w


def legend(svg: Svg, x: int, y: int, series: list) -> None:
    """在给定位置画图例；返回时按估算宽度横向排布。"""
    cx = x
    for s in series:
        svg.line(cx, y - 4, cx + 22, y - 4, stroke=s["color"], sw=2.4,
                 dash=s.get("dash"))
        svg.text(cx + 28, y, s["label"], size=12, fill=INK)
        n = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in str(s["label"]))
        cx += 26 + n * 12 + 22


# ================================================================== 单面板
def panel(svg: Svg, x0: int, y0: int, W: int, H: int, title: str,
          series: list, y_min=None, y_max=None, baseline=None) -> None:
    """series: [{"key","label","color","pts":[(epoch,value)],"dash"}]"""
    svg.text(x0 + W / 2, y0 - 14, title, size=15, fill=INK, anchor="middle",
             weight="600")

    allv = [v for s in series for _, v in s["pts"]]
    if not allv:
        svg.text(x0 + W / 2, y0 + H / 2, "（无数据）", size=13, fill=MUTED,
                 anchor="middle")
        return
    lo = y_min if y_min is not None else min(allv)
    hi = y_max if y_max is not None else max(allv)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    # 顶部多留白：面板左上角放了图例，等距留白会让曲线起点被图例压住
    rng = hi - lo
    hi += rng * 0.22
    lo -= rng * 0.08

    eps = [ep for s in series for ep, _ in s["pts"]]
    emin, emax = min(eps), max(eps)
    if emax == emin:
        emax = emin + 1

    def sx(e):
        return x0 + (e - emin) / (emax - emin) * W

    def sy(v):
        return y0 + H - (v - lo) / (hi - lo) * H

    # 网格 + y 刻度
    for t in nice_ticks(lo, hi, 5):
        if not (lo <= t <= hi):
            continue
        svg.line(x0, sy(t), x0 + W, sy(t), stroke=GRID, sw=1)
        svg.text(x0 - 8, sy(t) + 4, fmt(t), size=11, fill=MUTED, anchor="end")

    # x 刻度（epoch 整数）
    step = max(1, int(round((emax - emin) / 8)) or 1)
    for e in range(int(emin), int(emax) + 1, step):
        svg.line(sx(e), y0 + H, sx(e), y0 + H + 4, stroke=MUTED, sw=1)
        svg.text(sx(e), y0 + H + 18, str(e), size=11, fill=MUTED, anchor="middle")

    # 轴
    svg.line(x0, y0, x0, y0 + H, stroke=MUTED, sw=1.2)
    svg.line(x0, y0 + H, x0 + W, y0 + H, stroke=MUTED, sw=1.2)

    # 随机基线（如 0.5）
    if baseline is not None and lo <= baseline <= hi:
        svg.line(x0, sy(baseline), x0 + W, sy(baseline),
                 stroke=COLORS["base"], sw=1.4, dash="5 4")

    # 曲线 + 末点数值
    for s in series:
        pts = [(sx(e), sy(v)) for e, v in s["pts"]]
        svg.poly(pts, s["color"], sw=2.2, dash=s.get("dash"))
        for (px, py) in pts:
            svg.circle(px, py, 2.6, "#ffffff", stroke=s["color"])
        if pts:
            last = s["pts"][-1][1]
            svg.circle(pts[-1][0], pts[-1][1], 3.6, s["color"], stroke="#ffffff")
            svg.text(pts[-1][0] + 6, pts[-1][1] - 7, fmt(last), size=11,
                     fill=s["color"], weight="600")

    # 图例放在面板内部左上角：放外面会撞标题和相邻面板
    lw = _legend_width(series) + 12
    svg.rect(x0 + 10, y0 + 8, min(lw, W - 20), 24, fill="#ffffff",
             stroke=GRID, sw=1, rx=4)
    legend(svg, x0 + 22, y0 + 25, series)

    svg.text(x0 + W, y0 + H + 34, "epoch", size=11, fill=MUTED, anchor="end")


def legend(svg: Svg, x: int, y: int, series: list) -> None:
    cx = x
    for s in series:
        svg.line(cx, y - 4, cx + 22, y - 4, stroke=s["color"], sw=2.4,
                 dash=s.get("dash"))
        svg.text(cx + 28, y, s["label"], size=12, fill=INK)
        cx += 34 + len(s["label"]) * 12


# ================================================================== 主流程
def build_svg(records: list, note: str = "") -> str:
    recs = [r for r in records if isinstance(r, dict)]
    if not recs:
        raise SystemExit("[plot] history 里没有记录")

    def series_from(key, label, color, dash=None):
        pts = [(r.get("epoch", i), float(r[key]))
               for i, r in enumerate(recs) if isinstance(r.get(key), (int, float))]
        return {"key": key, "label": label, "color": color, "pts": pts, "dash": dash}

    have = [k for k in ("val_cls_acc", "val_cls_auc", "val_cls_f1")
            if any(isinstance(r.get(k), (int, float)) for r in recs)]

    W, H, GAP = 820, 210, 62
    LEFT, TOP = 92, 104
    n_panels = 2 + (1 if have else 0)
    svg = Svg(LEFT + W + 60, TOP + n_panels * (H + GAP) + 40)

    svg.text(LEFT, 36, "VIB-Net 训练曲线", size=19, fill=INK, weight="600")
    if note:
        svg.text(LEFT, 58, note, size=12, fill=MUTED)

    y = TOP
    loss_s = [series_from("train_loss", "训练损失", COLORS["train_loss"]),
              series_from("val_loss", "验证损失", COLORS["val_loss"])]
    panel(svg, LEFT, y, W, H, "损失", loss_s)
    y += H + GAP

    acc_s = [series_from("val_cls_acc", "验证准确率", COLORS["val_cls_acc"])]
    acc_vals = [v for _, v in acc_s[0]["pts"]] or [0.5]
    panel(svg, LEFT, y, W, H, "验证分类准确率", acc_s,
          y_min=min(0.4, min(acc_vals)), y_max=max(1.0, max(acc_vals)),
          baseline=0.5)
    y += H + GAP

    if have:
        auc_s = [series_from("val_cls_auc", "验证 AUC", COLORS["val_cls_auc"])]
        auc_vals = [v for _, v in auc_s[0]["pts"]] or [0.5]
        panel(svg, LEFT, y, W, H, "验证 AUC", auc_s,
              y_min=min(0.45, min(auc_vals)), y_max=max(1.0, max(auc_vals)),
              baseline=0.5)

    return svg.render()


def svg_to_png(svg_text: str, out_png: str, scale: float = 2.0) -> bool:
    """用 Qt 把 SVG 光栅化成 PNG（Qt 已解决中文字体，这里只需保证字体可见）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5.QtCore import QByteArray, Qt
        from PyQt5.QtGui import QImage, QPainter
        from PyQt5.QtSvg import QSvgRenderer
        from PyQt5.QtWidgets import QApplication

        # 复用截图脚本的字体补齐逻辑：Qt 找不到字体目录时中文会变方块
        try:
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            from make_ui_screenshots import setup_qt_fonts
            setup_qt_fonts(verbose=False)
        except Exception:                                              # noqa: BLE001
            pass

        app = QApplication.instance() or QApplication(sys.argv)
        r = QSvgRenderer(QByteArray(svg_text.encode("utf-8")))
        if not r.isValid():
            print("  [warn] SVG 解析失败，跳过 PNG")
            return False
        w = int(r.defaultSize().width() * scale)
        h = int(r.defaultSize().height() * scale)
        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(Qt.white)
        p = QPainter(img)
        r.render(p)
        p.end()
        ok = img.save(out_png)
        print(f"  [png ] {out_png}  {w}x{h}  {os.path.getsize(out_png)/1024:.0f} KB"
              if ok else "  [warn] PNG 保存失败")
        return bool(ok)
    except Exception as e:                                             # noqa: BLE001
        print(f"  [warn] 光栅化失败（SVG 仍可用）：{type(e).__name__}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="训练曲线绘图（SVG + PNG，无 matplotlib 依赖）")
    ap.add_argument("--history", default=os.path.join(ROOT, "outputs", "run_realval",
                                                     "history.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figs"))
    ap.add_argument("--name", default="train_curves")
    ap.add_argument("--note", default="")
    ap.add_argument("--png-only", action="store_true")
    ap.add_argument("--svg-only", action="store_true")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    if not os.path.exists(args.history):
        print(f"[plot] 找不到 {args.history}")
        return 2
    with open(args.history, encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = records.get("history", [])

    os.makedirs(args.out, exist_ok=True)
    svg_text = build_svg(records, args.note)

    if not args.png_only:
        p = os.path.join(args.out, args.name + ".svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg_text)
        print(f"  [svg ] {p}  {os.path.getsize(p)/1024:.0f} KB")
    if not args.svg_only:
        svg_to_png(svg_text, os.path.join(args.out, args.name + ".png"), args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
