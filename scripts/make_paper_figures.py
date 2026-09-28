"""生成论文/结题材料所需的图表（纯 PIL，无 matplotlib 依赖）。

本机 pip 取不到 PyPI（装不上 matplotlib），所以这里用 Pillow 手绘坐标轴。
所有图都是**从已落盘的实测数据重画**，不做任何估计：

  输入（全部只读）
    outputs/ablation_{pretrained,random}/history.json   逐 epoch 训练/验证曲线
    outputs/cross_gen_pretrained_fixed/cross_generator.json   逐生成器指标（预训练臂）
    outputs/cross_gen_random/cross_generator_random.json      逐生成器指标（随机臂）
    outputs/cross_gen_scores/pretrained.npz                   逐样本分数（画 ROC / 混淆矩阵）
    outputs/backbone_budget.json                              骨干预算（参数量/体积/延迟）
    outputs/cpu_benchmark.json / cpu_benchmark_b16.json        ONNX 实测吞吐
    outputs/robustness.json                                   鲁棒性（若已生成）
    configs/default.yaml                                      β 退火参数
    data/Datasets/ForenSynths/val/**                          数据集样例

  输出
    deliverables/figures/*.png

用法：
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = os.path.join(ROOT, "deliverables", "figures")

# ---------------------------------------------------------------- 主题（打印友好，浅色）
BG = (255, 255, 255)
INK = (28, 32, 38)
SUB = (110, 118, 128)
GRID = (226, 230, 236)
ACCENT = (198, 40, 40)      # 红：本项目方法 / 预训练臂
MUTED = (120, 132, 152)     # 灰蓝：对照 / 随机臂
OK = (30, 132, 73)
WARN = (196, 120, 20)

FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for p in ([FONT_PATHS[1]] if bold else []) + FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


class Chart:
    """极简绘图板：坐标轴 + 折线 + 柱状 + 散点。"""

    def __init__(self, w: int, h: int, title: str = "", sub: str = ""):
        self.w, self.h = w, h
        self.im = Image.new("RGB", (w, h), BG)
        self.d = ImageDraw.Draw(self.im)
        self.ml, self.mr, self.mt, self.mb = 72, 28, 62, 66
        if title:
            self.d.text((self.ml, 18), title, font=font(22, True), fill=INK)
        if sub:
            self.d.text((self.ml, 44), sub, font=font(13), fill=SUB)

    # -------- 坐标轴
    def axes(self, xlab: str = "", ylab: str = "", ymin: float = 0.0, ymax: float = 1.0,
             yticks: int = 5, xlabels: list | None = None, yfmt: str = "{:.2f}"):
        x0, y0 = self.ml, self.h - self.mb
        x1, y1 = self.w - self.mr, self.mt
        self._box = (x0, y0, x1, y1)
        self._ymin, self._ymax = ymin, ymax
        for i in range(yticks + 1):
            v = ymin + (ymax - ymin) * i / yticks
            y = y0 - (y0 - y1) * i / yticks
            self.d.line([(x0, y), (x1, y)], fill=GRID, width=1)
            self.d.text((x0 - 8, y - 7), yfmt.format(v), font=font(12), fill=SUB, anchor="ra")
        self.d.line([(x0, y1), (x0, y0)], fill=INK, width=2)
        self.d.line([(x0, y0), (x1, y0)], fill=INK, width=2)
        if xlab:
            self.d.text(((x0 + x1) / 2, y0 + 40), xlab, font=font(14), fill=INK, anchor="mm")
        if ylab:
            self.d.text((16, (y0 + y1) / 2), ylab, font=font(14), fill=INK, anchor="lm")
        if xlabels:
            n = len(xlabels)
            for i, t in enumerate(xlabels):
                x = x0 + (x1 - x0) * (i + 0.5) / n
                self.d.text((x, y0 + 8), str(t), font=font(11), fill=SUB, anchor="ma")
        return self

    def pxy(self, x: float, y: float):
        x0, y0, x1, y1 = self._box
        fx = x0 + (x1 - x0) * x
        fy = y0 - (y0 - y1) * (y - self._ymin) / (self._ymax - self._ymin)
        return fx, fy

    def series(self, xs: list, ys: list, color=ACCENT, width: int = 3, dots: bool = True,
               label: str | None = None):
        pts = [self.pxy(x, y) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            self.d.line(pts, fill=color, width=width, joint="curve")
        if dots:
            for (px, py) in pts:
                self.d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color,
                               outline=BG, width=2)
        if label:
            lx, ly = pts[-1]
            self.d.text((lx + 8, ly - 6), label, font=font(12, True), fill=color)
        return self

    def bars(self, values: list, colors: list | None = None, labels: list | None = None,
             width_ratio: float = 0.66, gap_ratio: float = 0.0, fmt: str = "{:.3f}",
             show_val: bool = True, val_font: int = 10, hatch_alt: bool = False):
        x0, y0, x1, y1 = self._box
        n = len(values)
        slot = (x1 - x0) / n
        bw = slot * width_ratio
        for i, v in enumerate(values):
            cx = x0 + slot * (i + 0.5)
            top = self.pxy(0.5, v)[1]
            c = (colors[i] if colors else (ACCENT if i % 2 == 0 else MUTED))
            self.d.rectangle([cx - bw / 2, top, cx + bw / 2, y0], fill=c)
            if show_val:
                self.d.text((cx, top - 8), fmt.format(v), font=font(val_font, True),
                            fill=INK, anchor="mb")
            if labels:
                self.d.text((cx, y0 + 6), labels[i], font=font(11), fill=SUB, anchor="ma")
        return self

    def legend(self, items: list, pos: str = "ul"):
        x0, y0, x1, y1 = self._box
        bx = x0 + 14 if "l" in pos else x1 - 200
        by = y1 + 12 if "u" in pos else y0 - 140
        for i, (txt, c) in enumerate(items):
            yy = by + i * 20
            self.d.rectangle([bx, yy + 3, bx + 22, yy + 12], fill=c)
            self.d.text((bx + 30, yy), txt, font=font(12), fill=INK)

    def note(self, text: str, y: int | None = None):
        yy = y if y is not None else self.h - 12
        self.d.text((self.ml, yy), text, font=font(11), fill=SUB)

    def save(self, name: str):
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, name)
        self.im.save(p)
        print(f"[fig] {os.path.relpath(p, ROOT)}")
        return p


# ---------------------------------------------------------------- 工具
def load_json(*rel):
    for r in rel:
        p = os.path.join(ROOT, r)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def roc_points(y, p, n: int = 400):
    """按分数降序扫描出 ROC 点（返回 (fpr, tpr, auc)）。"""
    pairs = sorted(zip(p, y), key=lambda t: -t[0])
    P = sum(1 for _, t in pairs if t == 1)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return [0, 1], [0, 1], 0.5
    tp = fp = 0
    pts = [(0.0, 0.0)]
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:   # 并列分数共用一个阈值
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        pts.append((fp / N, tp / P))
        i = j
    pts.append((1.0, 1.0))
    auc = 0.0
    for k in range(1, len(pts)):
        auc += (pts[k][0] - pts[k - 1][0]) * (pts[k][1] + pts[k - 1][1]) / 2
    if len(pts) > n:
        step = len(pts) // n
        pts = pts[::step] + [pts[-1]]
    return [a for a, _ in pts], [b for _, b in pts], auc


# ==========================================================================
def fig07_training_curves():
    hp = load_json("outputs/ablation_pretrained/history.json")
    hr = load_json("outputs/ablation_random/history.json")
    if not hp or not hr:
        print("[skip] fig07：缺 history.json")
        return
    c = Chart(880, 460, "图 7 · 骨干权重受控对照实验：验证集收敛曲线",
              "唯一自变量=骨干权重（ImageNet 预训练 vs 随机初始化）；均为 stage1 分类预训练 3 轮，CPU")
    c.axes("训练轮次 (epoch)", "验证集 AUC", 0.0, 1.0, 5,
           [str(e["epoch"]) for e in hp])
    xs = [e["epoch"] for e in hp]
    c.series(xs, [e["val_cls_auc"] for e in hp], ACCENT, 3, True, "预训练骨干")
    c.series([e["epoch"] for e in hr], [e["val_cls_auc"] for e in hr], MUTED, 3, True, "随机初始化")
    c.note("注：随机初始化臂 tn=0（全部判为伪造），判决边界未建立；两臂最佳 AUC 仅差 +0.038，"
           "故只能称预训练权重为「必要的非充分」条件。")
    c.save("fig07_training_curves_val_auc.png")

    # 第二张：val ACC + tn/tp
    c2 = Chart(880, 460, "图 7b · 判决是否建立：混淆矩阵关键量 tn / tp（验证集 1300 张）",
               "判据是 tn 与 tp 是否同时为正，而不是 ACC —— 退化臂的 ACC 恒为 0.5")
    c2.axes("训练轮次 (epoch)", "样本数（上限 650）", 0.0, 700.0, 7,
            [str(e["epoch"]) for e in hp], yfmt="{:.0f}")
    c2.series(xs, [e["val_cls_tn"] for e in hp], ACCENT, 3, True, "预训练 tn")
    c2.series(xs, [e["val_cls_tp"] for e in hp], OK, 3, True, "预训练 tp")
    c2.series([e["epoch"] for e in hr], [e["val_cls_tn"] for e in hr], MUTED, 3, True, "随机 tn")
    c2.series([e["epoch"] for e in hr], [e["val_cls_tp"] for e in hr], WARN, 3, True, "随机 tp")
    c2.note("随机初始化臂 tn 全程为 0：全部样本被判为伪造，模型只学到「恒判假」这一个解。")
    c2.save("fig07b_tn_tp_boundary.png")


def fig13_cross_generator():
    dp = load_json("outputs/cross_gen_pretrained_fixed/cross_generator.json")
    dr = load_json("outputs/cross_gen_random/cross_generator_random.json")
    if not dp:
        print("[skip] fig13")
        return
    gens = list(dp["per_generator"].keys())
    acc_p = [dp["per_generator"][g]["acc"] for g in gens]
    acc_r = [dr["per_generator"][g]["acc"] for g in gens] if dr else None
    c = Chart(1180, 500, "图 13 · 跨生成器泛化：各生成器检测准确率（每生成器真/假各 100 张）",
              "训练只用 ProGAN；其余 12 个生成器均为零样本。虚线为随机水平 0.50")
    c.axes("生成器", "ACC", 0.0, 1.0, 5, gens)
    x0, y0, x1, y1 = c._box
    yy = c.pxy(0.5, 0.5)[1]
    c.d.line([(x0, yy), (x1, yy)], fill=SUB, width=1)
    if acc_r:
        n = len(gens)
        slot = (x1 - x0) / n
        bw = slot * 0.34
        for i, g in enumerate(gens):
            cx = x0 + slot * (i + 0.5)
            for off, v, col in ((-bw / 2, acc_r[i], MUTED), (bw / 2, acc_p[i], ACCENT)):
                top = c.pxy(0.5, v)[1]
                c.d.rectangle([cx + off - bw / 2, top, cx + off + bw / 2, y0], fill=col)
        c.legend([("预训练骨干", ACCENT), ("随机初始化", MUTED)], "ul")
        c.note("宏平均：预训练 ACC 0.5412 / AUC 0.6076；随机臂 ACC 0.5000（真图 ACC=0，全判伪造）。"
               "各生成器差异极大（AUC 0.4665~0.8297），说明当前训练量下泛化能力很弱且极不均衡。")
    else:
        c.bars(acc_p, labels=gens)
    c.save("fig13_cross_generator_acc.png")

    # AUC 版本
    aucs = [dp["per_generator"][g]["auc"] for g in gens]
    c2 = Chart(1180, 470, "图 13b · 跨生成器泛化：各生成器 AUC（零样本）",
               "合并宏平均 AUC = 0.6076；微平均（合并全部样本）= 0.5840")
    c2.axes("生成器", "AUC", 0.0, 1.0, 5, gens)
    x0, y0, x1, y1 = c2._box
    yy = c2.pxy(0.5, 0.5)[1]
    c2.d.line([(x0, yy), (x1, yy)], fill=SUB, width=1)
    colors = [OK if a >= 0.7 else (WARN if a >= 0.55 else ACCENT) for a in aucs]
    c2.bars(aucs, colors=colors, labels=gens, fmt="{:.3f}")
    c2.note("绿：AUC≥0.70（4 个）；黄：0.55–0.70（2 个）；红：<0.55（7 个，其中 whichfaceisreal 0.4665 低于随机）。")
    c2.save("fig13b_cross_generator_auc.png")


def fig_roc_confusion():
    import numpy as np
    p = os.path.join(ROOT, "outputs/cross_gen_scores/pretrained.npz")
    if not os.path.exists(p):
        print("[skip] roc")
        return
    d = np.load(p, allow_pickle=True)
    y, probs = d["y"], d["probs"]
    fpr, tpr, auc = roc_points(list(y), list(probs))

    c = Chart(700, 620, "图 8 · ROC 曲线（ForenSynths test 跨生成器，2600 张）",
              f"并列分数按秩平均处理；AUC = {auc:.4f}（宏平均口径 0.6076）")
    c.axes("假正率 FPR", "真正率 TPR", 0.0, 1.0, 5)
    x0, y0, x1, y1 = c._box
    c.d.line([(x0, y0), (x1, y1)], fill=GRID, width=2)
    c.series(fpr, tpr, ACCENT, 3, False)
    c.note(f"AUC = {auc:.4f}（微平均，合并 2600 样本）。灰色对角线为随机水平 0.5。"
           "该值说明当前权重在跨生成器任务上仅略高于随机。")
    c.save("fig08_roc.png")

    thr = 0.5
    tp = sum(1 for a, b in zip(y, probs) if a == 1 and b >= thr)
    fn = sum(1 for a, b in zip(y, probs) if a == 1 and b < thr)
    fp = sum(1 for a, b in zip(y, probs) if a == 0 and b >= thr)
    tn = sum(1 for a, b in zip(y, probs) if a == 0 and b < thr)
    cm = [[tn, fp], [fn, tp]]
    c2 = Chart(760, 520, "图 9 · 混淆矩阵（阈值 0.5）",
               f"AUC={auc:.4f}　ACC={(tn+tp)/len(y):.4f}　真图 ACC={tn/(tn+fp):.4f}　假图 ACC={tp/(tp+fn):.4f}")
    x0, y0, x1, y1 = c2.ml, c2.h - c2.mb, c2.w - c2.mr, c2.mt
    c2._box, c2._ymin, c2._ymax = (x0, y0, x1, y1), 0.0, 1.0
    cidx = x0 + 96
    ctop = y1 + 24
    cell = 168
    for r in range(2):
        for col in range(2):
            v = cm[r][col]
            mx = max(max(row) for row in cm) or 1
            inten = 0.16 + 0.72 * (v / mx)
            base = (198, 40, 40) if (r == col) else (196, 120, 20)
            fill = tuple(int(255 - (255 - ch) * inten) for ch in base)
            X0 = cidx + col * cell
            Y0 = ctop + r * cell
            c2.d.rectangle([X0, Y0, X0 + cell - 6, Y0 + cell - 6], fill=fill)
            c2.d.text((X0 + cell / 2 - 3, Y0 + cell / 2 - 3), f"{v}", font=font(30, True),
                      fill=INK, anchor="mm")
    c2.d.text((cidx + cell - 3, ctop - 20), "预测：真实", font=font(13), fill=INK, anchor="ma")
    c2.d.text((cidx + cell * 2 - 3, ctop - 20), "预测：伪造", font=font(13), fill=INK, anchor="ma")
    c2.d.text((cidx - 12, ctop + cell / 2 - 3), "实际：真实", font=font(13), fill=INK, anchor="rm")
    c2.d.text((cidx - 12, ctop + cell * 1.5 - 3), "实际：伪造", font=font(13), fill=INK, anchor="rm")
    c2.note("注意：假图 ACC 明显高于真图 ACC —— 模型有「倾向判伪造」的偏置（真图召回偏低）。", ctop + cell * 2 + 20)
    c2.save("fig09_confusion_matrix.png")


def fig12_efficiency():
    bb = load_json("outputs/backbone_budget.json")
    cb = load_json("outputs/cpu_benchmark.json")
    cb16 = load_json("outputs/cpu_benchmark_b16.json")
    if not bb:
        print("[skip] fig12")
        return
    pts = []
    for r in bb["rows"]:
        pts.append((f'{r["config"].split(".")[0]}-{r["variant"]}', r["fp32_MB"], r["torch_cpu_fps"]))
    c = Chart(920, 520, "图 12 · 体积—速度权衡（PyTorch CPU 前向，224×224，batch=1）",
              "横轴 FP32 体积(MB)，纵轴 CPU 吞吐(张/秒)；阴影带为申报书两条硬指标约束")
    c.axes("模型体积 (MB)", "CPU 吞吐 (张/秒)", 0.0, 5.0, 5,
           yfmt="{:.1f}")
    x0, y0, x1, y1 = c._box
    # 120MB 竖线
    xr = x0 + (x1 - x0) * min(1.0, 120.0 / 420.0)
    c.d.line([(xr, y1), (xr, y0)], fill=OK, width=2)
    c.d.text((xr + 6, y1 + 6), "120 MB 上限", font=font(11, True), fill=OK)
    ry = c.pxy(0, 8.0)[1]
    if 0 <= ry <= y0:
        c.d.line([(x0, ry), (x1, ry)], fill=OK, width=2)
        c.d.text((x0 + 6, ry - 18), "8 张/秒 下限", font=font(11, True), fill=OK)
    for name, mb, fps in pts:
        px = x0 + (x1 - x0) * min(1.0, mb / 420.0)
        py = c.pxy(0, min(4.9, fps))[1]
        col = OK if (mb <= 120) else ACCENT
        c.d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=col, outline=BG, width=2)
        c.d.text((px + 10, py - 6), f"{name}  {mb:.0f}MB/{fps:.2f}", font=font(11), fill=INK)
    c.note("全部档位的 PyTorch eager 吞吐都 < 8 张/秒；达标靠 ONNX Runtime + 算子简化（见下表）。")
    c.save("fig12a_size_vs_speed_pytorch.png")

    # ONNX 实测
    rows = []
    if cb:
        for r in cb["rows"]:
            rows.append((r["model"], r["size_MB"], r["median_fps"]))
    if cb16:
        for r in cb16["rows"]:
            rows.append((r["model"], r["size_MB"], r["median_fps"]))
    seen, uniq = set(), []
    for n, s, f in rows:
        if n in seen:
            continue
        seen.add(n)
        uniq.append((n, s, f))
    c2 = Chart(980, 520, "图 12b · ONNX Runtime 实测：体积 vs 吞吐（本机 Intel 7 逻辑核，无 VNNI）",
               "同一模型不同框架相差 3.7 倍（PyTorch 392ms vs ONNX 106ms）")
    c2.axes("模型", "中位吞吐 (张/秒)", 0.0, 13.0, 5, yfmt="{:.1f}")
    vals = [f for _, _, f in uniq]
    c2.bars(vals, colors=[OK if f >= 8 else ACCENT for f in vals],
            labels=[n.replace(".onnx", "") for n, _, _ in uniq], fmt="{:.2f}")
    ry = c2.pxy(0, 8.0)[1]
    x0, y0, x1, y1 = c2._box
    c2.d.line([(x0, ry), (x1, ry)], fill=OK, width=2)
    c2.d.text((x1 - 6, ry - 18), "申报书下限 8 张/秒", font=font(11, True), fill=OK, anchor="ra")
    c2.note("只有 lite_simplified（s16 + ONNX 算子简化，98.7MB）同时满足体积与速度两项硬指标；"
            "INT8 在本机反而变慢 3.3 倍（无 VNNI，量化算子退化为逐元素模拟）。")
    c2.save("fig12b_onnx_size_speed.png")


def fig04_beta():
    import yaml
    p = os.path.join(ROOT, "configs/default.yaml")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vib = cfg["model"]["vib"]
    bmax = float(vib.get("beta_max", 0.1))
    t0 = int(vib.get("beta_warmup_start", 20))
    t1 = int(vib.get("beta_warmup_end", 40))
    tot = t1 + int(cfg["train"]["stages"][-1]["epochs"])
    xs = list(range(0, tot + 1))
    ys = [0.0 if t < t0 else (bmax if t >= t1 else bmax * (t - t0) / (t1 - t0)) for t in xs]
    c = Chart(860, 430, "图 4 · β 退火与三阶段训练划分",
              f"β(t)=0 (t<{t0})；线性升温 ({t0}≤t<{t1})；β={bmax} (t≥{t1})　——　参数取自 configs/default.yaml")
    c.axes("训练轮次 t (epoch)", "β(t)", 0.0, bmax * 1.25, 5, yfmt="{:.3f}")
    x0, y0, x1, y1 = c._box
    for xe, txt in ((t0, "分类预训练结束"), (t1, "定位预训练结束")):
        px = x0 + (x1 - x0) * xe / tot
        c.d.line([(px, y1), (px, y0)], fill=GRID, width=2)
        c.d.text((px + 5, y1 + 6), txt, font=font(11), fill=SUB)
    c.series([x / tot for x in xs], ys, ACCENT, 3, False)
    c.note("KL 散度另设 [0,10] 硬裁剪（前向裁剪 + 反向恒等），梯度 L2 裁剪上限 5.0。")
    c.save("fig04_beta_annealing.png")


def fig05_samples(n: int = 6):
    """数据集样例：ForenSynths val 的真/假对照。"""
    base = os.path.join(ROOT, "data/Datasets/ForenSynths/val")
    if not os.path.isdir(base):
        print("[skip] fig05")
        return
    pat = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.bmp", "*.webp")
    real, fake = [], []
    for p in pat:
        real += glob.glob(os.path.join(base, "**", "0_real", p), recursive=True)
        fake += glob.glob(os.path.join(base, "**", "1_fake", p), recursive=True)
    real, fake = sorted(real)[:n], sorted(fake)[:n]
    if not real or not fake:
        print("[skip] fig05：未找到样例图")
        return
    T = 200
    W = T * n + 20
    H = T * 2 + 108
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((12, 12), "图 5 · 数据集样例：ForenSynths/val 真图（上）与 ProGAN 生成图（下）",
           font=font(19, True), fill=INK)
    d.text((12, 40), "同尺寸同格式，肉眼几乎不可分；这正是「仅靠空域特征难以奏效」的直接证据。",
           font=font(12), fill=SUB)
    for row, paths in ((0, real), (1, fake)):
        for i, p in enumerate(paths[:n]):
            try:
                t = Image.open(p).convert("RGB").resize((T - 8, T - 8))
            except Exception:
                continue
            im.paste(t, (12 + i * T, 70 + row * T))
    im.save(os.path.join(OUT, "fig05_dataset_samples.png"))
    print("[fig] deliverables/figures/fig05_dataset_samples.png")


def fig11_robustness():
    p = os.path.join(ROOT, "outputs/robustness.json")
    if not os.path.exists(p):
        print("[skip] fig11：robustness.json 尚未生成")
        return
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"]
    groups = {"JPEG 压缩": "JPEG", "缩放": "缩放", "裁剪": "裁剪",
              "旋转": "旋转", "高斯噪声": "高斯", "椒盐噪声": "椒盐"}
    c = Chart(1080, 500, "图 11 · 鲁棒性：常见后处理下的检测 ACC 与 AUC",
              f"样本 {data['n_samples']} 张，阈值 {data['threshold']}；虚线为无处理基线")
    c.axes("后处理操作", "指标值", 0.0, 1.0, 5)
    x0, y0, x1, y1 = c._box
    xs = [i / max(1, len(res) - 1) for i in range(len(res))]
    base_acc = res[0]["acc"]
    base_auc = res[0]["auc"]
    c.series(xs, [r["acc"] or 0 for r in res], ACCENT, 3, True, "ACC")
    c.series(xs, [r["auc"] or 0 for r in res], MUTED, 3, True, "AUC")
    ba = c.pxy(0, base_acc)[1]
    c.d.line([(x0, ba), (x1, ba)], fill=GRID, width=2)
    step = max(1, len(res) // 14)
    for i in range(0, len(res), step):
        px = x0 + (x1 - x0) * xs[i]
        c.d.text((px, y0 + 6), res[i]["op"][:9], font=font(10), fill=SUB, anchor="ma")
    c.note(f"基线（原图）ACC={base_acc:.4f}、AUC={base_auc:.4f}。所有后处理下指标均在随机线附近波动，"
           "说明当前权重的判别信号本身很弱，鲁棒性结论需在正式训练后重测。")
    c.save("fig11_robustness.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[fig] 输出目录：{os.path.relpath(OUT, ROOT)}")
    fig04_beta()
    fig05_samples()
    fig07_training_curves()
    fig_roc_confusion()
    fig11_robustness()
    fig12_efficiency()
    fig13_cross_generator()
    print("[fig] 完成。")


if __name__ == "__main__":
    main()
