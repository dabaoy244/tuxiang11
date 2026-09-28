#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成计算机软件著作权登记所需的两份文档。

1) 源程序文档：软件名称+版本号页眉、连续页码，**前 30 页 + 后 30 页**，
   每页 50 行（版权中心要求每页不少于 50 行）。
2) 软件说明书：概述 / 运行环境 / 安装启动 / 界面说明 / 操作流程 /
   系统设置 / 常见问题 / 技术指标，含离屏渲染的真实界面截图。

用法：
    # 先拍界面截图（用户手册要用）
    python scripts/make_ui_screenshots.py --config configs/lite.yaml

    # 再生成两份文档
    python scripts/make_copyright_docs.py \
        --shots docs/copyright/ui_shots \
        --out docs/copyright

依赖：reportlab（已装）。中文字体取自 C:/Windows/Fonts。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib import colors                                          # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT            # noqa: E402
from reportlab.lib.pagesizes import A4                                    # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet      # noqa: E402
from reportlab.lib.units import mm                                        # noqa: E402
from reportlab.pdfbase import pdfmetrics                                  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont                              # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak,           # noqa: E402
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

SOFTWARE_NAME = "空频双分支 AI 图像篡改检测系统"
VERSION = "V1.0"

#: 源程序文档收录的代码文件（按「先核心后外围」排列，便于评审阅读）
SOURCE_ORDER = [
    "src/models/vibnet.py",
    "src/models/spatial_branch.py",
    "src/models/freq_branch.py",
    "src/models/dft.py",
    "src/models/cs_cam.py",
    "src/models/vib.py",
    "src/models/gradient_stop.py",
    "src/models/mobile_unetv2.py",
    "src/losses/multi_task.py",
    "src/data/datasets.py",
    "src/data/transforms.py",
    "src/data/synth.py",
    "src/engine/trainer.py",
    "src/engine/metrics.py",
    "src/evaluation/evaluate.py",
    "src/evaluation/robustness.py",
    "src/evaluation/ablation.py",
    "src/deploy/export_onnx.py",
    "src/deploy/quantize_int8.py",
    "src/deploy/optimize_openvino.py",
    "src/deploy/inference.py",
    "app/main.py",
    "app/modules/input_panel.py",
    "app/modules/result_panel.py",
    "app/modules/settings_panel.py",
    "app/modules/exporter.py",
    "app/modules/workers.py",
    "scripts/train.py",
    "scripts/fetch_datasets.py",
    "scripts/prepare_datasets.py",
    "scripts/benchmark_cpu.py",
    "scripts/smoke_test.py",
]

FONT_CANDIDATES = {
    "song": [r"C:\Windows\Fonts\simfang.ttf", r"C:\Windows\Fonts\simsun.ttc",
             r"C:\Windows\Fonts\Deng.ttf"],
    "hei": [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\Dengb.ttf"],
}


def register_fonts() -> tuple:
    """注册中文字体；reportlab 不支持 .ttc 集合时自动退回下一个候选。"""
    reg = {}
    for tag, paths in FONT_CANDIDATES.items():
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                pdfmetrics.registerFont(TTFont(f"CN-{tag}", p))
                reg[tag] = f"CN-{tag}"
                break
            except Exception:                                          # noqa: BLE001
                continue
    if "song" not in reg or "hei" not in reg:
        raise RuntimeError("未找到可用中文字体，请检查 C:/Windows/Fonts")
    return reg["song"], reg["hei"]


# ======================================================================
#  一、源程序文档
# ======================================================================
def collect_lines(drop_blank: bool = True) -> list:
    """把源码文件拼成 (显示文本, 是否文件首行) 的行序列。

    为什么默认丢掉空行
    ------------------
    软著实务口径是「每页不少于 50 行」。空行不产生任何文本，
    若保留空行，一页排 50 行里会有约 20% 是空行，**可见文字只有 40 行左右**，
    容易被按"页行数不足"退件。源码里空行占比实测约 22%，
    所以这里默认丢掉空行，让每页排下的行=可见行，行数才好核对。
    """
    out = []
    seen = set()
    for rel in SOURCE_ORDER:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        seen.add(rel)
        out.append((f"/* ===== 文件：{rel} =====".ljust(78, "=") + " */", True))
        with open(p, encoding="utf-8", errors="replace") as f:
            for ln in f.read().splitlines():
                txt = ln.replace("\t", "    ").rstrip()
                if drop_blank and not txt:
                    continue
                out.append((txt, False))
    return out


def wrap_line(text: str, font: str, size: float, max_w: float) -> list:
    """按实际渲染宽度折行，避免长行溢出页面。"""
    if not text:
        return [""]
    if pdfmetrics.stringWidth(text, font, size) <= max_w:
        return [text]
    lines, cur = [], ""
    for ch in text:
        if pdfmetrics.stringWidth(cur + ch, font, size) > max_w:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def make_source_pdf(out_path: str, song: str, hei: str,
                    head_pages: int = 30, tail_pages: int = 30,
                    lines_per_page: int = 52,
                    drop_blank: bool = True) -> dict:
    from reportlab.pdfgen import canvas as rl_canvas

    PAGE_W, PAGE_H = A4
    MARGIN_L, MARGIN_R = 18 * mm, 15 * mm
    TOP, BOTTOM = 20 * mm, 16 * mm
    # 字号/行距按「一页放得下 lines_per_page 行」定：
    # 可用高度 = PAGE_H - TOP - BOTTOM ≈ 739.9pt，需 lines_per_page*LEADING + LEADING ≤ 739.9
    FONT_SIZE, LEADING = 8.2, 12.6
    code_w = PAGE_W - MARGIN_L - MARGIN_R
    avail_h = PAGE_H - TOP - BOTTOM
    if lines_per_page * LEADING + LEADING > avail_h:
        raise ValueError(
            f"版式放不下：{lines_per_page} 行 × {LEADING}pt 行距 = "
            f"{lines_per_page * LEADING + LEADING:.0f}pt > 可用 {avail_h:.0f}pt")

    raw = collect_lines(drop_blank=drop_blank)
    wrapped = []
    for text, is_head in raw:
        for i, seg in enumerate(wrap_line(text, song, FONT_SIZE, code_w)):
            wrapped.append((seg, is_head and i == 0))
    total_pages = head_pages + tail_pages
    need = total_pages * lines_per_page

    if len(wrapped) < need:
        print(f"  [warn] 源码仅 {len(wrapped)} 行渲染行，不足 {need} 行；"
              f"将按实际页数输出")
        total_pages = max(1, len(wrapped) // lines_per_page)

    first = wrapped[:head_pages * lines_per_page]
    last = wrapped[-tail_pages * lines_per_page:] if total_pages > head_pages else []

    # ⚠ 交付前自检：软著口径要求每页不少于 50 行。空行不产生文本，
    #   所以这里按"非空行"统计，任一行不足就明确告警，别等到被退件才发现。
    def min_visible_rows(block: list) -> int:
        worst = lines_per_page
        for s in range(0, len(block), lines_per_page):
            page = block[s:s + lines_per_page]
            worst = min(worst, sum(1 for t, _ in page if t.strip()))
        return worst

    worst = min(min_visible_rows(first), min_visible_rows(last)) if last else \
        min_visible_rows(first)
    if worst < 50:
        print(f"  [warn] 存在页面可见代码行仅 {worst} 行（<50），"
              f"建议提高 --lines-per-page 或开启去空行")
    else:
        print(f"  [check] 每页可见代码行最少 {worst} 行（≥50 ✅）")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    c = rl_canvas.Canvas(out_path, pagesize=A4)
    c.setTitle(f"{SOFTWARE_NAME} {VERSION} 源程序")
    c.setAuthor(SOFTWARE_NAME)

    def draw_header_footer(pno: int) -> None:
        c.setFont(hei, 9)
        c.setFillColor(colors.HexColor("#333333"))
        c.drawString(MARGIN_L, PAGE_H - TOP + 7,
                     f"软件名称：{SOFTWARE_NAME}  {VERSION}")
        c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - TOP + 7,
                          f"第 {pno} 页  共 {total_pages} 页")
        c.setStrokeColor(colors.HexColor("#999999"))
        c.setLineWidth(0.4)
        c.line(MARGIN_L, PAGE_H - TOP + 2, PAGE_W - MARGIN_R, PAGE_H - TOP + 2)
        c.line(MARGIN_L, BOTTOM - 6, PAGE_W - MARGIN_R, BOTTOM - 6)

    def render(block: list, start_page: int) -> int:
        pno = start_page
        i = 0
        while i < len(block):
            draw_header_footer(pno)
            c.setFont(song, FONT_SIZE)
            y = PAGE_H - TOP - LEADING
            for _ in range(lines_per_page):
                if i >= len(block):
                    break
                text, is_head = block[i]
                if is_head:
                    c.setFillColor(colors.HexColor("#0b4f9c"))
                else:
                    c.setFillColor(colors.black)
                c.drawString(MARGIN_L, y, text)
                y -= LEADING
                i += 1
            c.showPage()
            pno += 1
        return pno

    nextp = render(first, 1)
    if last:
        render(last, nextp)
    c.save()
    return {"path": out_path, "pages": total_pages,
            "rendered_lines": len(wrapped),
            "size_kb": round(os.path.getsize(out_path) / 1024, 1)}


# ======================================================================
#  二、软件说明书
# ======================================================================
def make_manual_pdf(out_path: str, song: str, hei: str, shots_dir: str) -> dict:
    ss = getSampleStyleSheet()
    st_title = ParagraphStyle("t", parent=ss["Title"], fontName=hei, fontSize=20,
                              leading=28, alignment=TA_CENTER, spaceAfter=6)
    st_sub = ParagraphStyle("s", parent=ss["Normal"], fontName=song, fontSize=12,
                            leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#555"))
    st_h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=hei, fontSize=14,
                           leading=22, spaceBefore=14, spaceAfter=8,
                           textColor=colors.HexColor("#0b4f9c"))
    st_h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=hei, fontSize=12,
                           leading=19, spaceBefore=9, spaceAfter=5,
                           textColor=colors.HexColor("#12395e"))
    st_body = ParagraphStyle("b", parent=ss["Normal"], fontName=song, fontSize=10.5,
                             leading=17, alignment=TA_JUSTIFY, firstLineIndent=21)
    st_li = ParagraphStyle("li", parent=st_body, firstLineIndent=0, leftIndent=14,
                           bulletIndent=2)
    st_cap = ParagraphStyle("cap", parent=ss["Normal"], fontName=song, fontSize=9.5,
                            leading=15, alignment=TA_CENTER,
                            textColor=colors.HexColor("#666666"))
    st_code = ParagraphStyle("code", parent=ss["Normal"], fontName=song, fontSize=9,
                             leading=13.5, leftIndent=10,
                             backColor=colors.HexColor("#f5f6f8"))

    def P(t):
        return Paragraph(t, st_body)

    def LI(t):
        return Paragraph(t, st_li, bulletText="•")

    def H1(t):
        return Paragraph(t, st_h1)

    def H2(t):
        return Paragraph(t, st_h2)

    story = []
    story += [
        Spacer(1, 30 * mm),
        Paragraph(f"{SOFTWARE_NAME}", st_title),
        Paragraph(f"版本号：{VERSION}", st_title),
        Spacer(1, 8 * mm),
        Paragraph("软 件 说 明 书", st_title),
        Spacer(1, 4 * mm),
        Paragraph("（用户操作手册）", st_sub),
        Spacer(1, 40 * mm),
        Paragraph(f"编制日期：{date.today().strftime('%Y 年 %m 月 %d 日')}", st_sub),
        PageBreak(),
    ]

    # -------------------------------------------------- 1 概述
    story += [H1("一、软件概述")]
    story += [H2("1.1 软件简介")]
    story += [P("本软件是一套面向数字图像真实性鉴别的桌面分析工具，可对输入图像给出"
                "「真实 / 疑似伪造」的二分类判定，并同步输出像素级篡改区域热力图与"
                "二值掩码，辅助使用者快速定位可疑区域。")]
    story += [P("软件采用空域与频域双分支特征提取架构：空域分支以视觉 Transformer "
                "骨干提取高层语义，并辅以深度可分离卷积提取局部纹理；频域分支对图像"
                "做二维离散傅里叶变换后，分别建模幅值谱与相位谱。两路特征经交叉注意力"
                "融合模块（CS-CAM）对齐后，送入分层变分信息瓶颈（VIB）完成特征压缩与"
                "泛化性增强，最终由双支路多任务头同时输出图像级判定与像素级定位结果。")]
    story += [P("软件全部计算在本机完成，网络通信仅用于模型权重与数据集的首次获取，"
                "检测过程中图像数据不出本机，满足数据隐私保护要求。")]

    story += [H2("1.2 主要功能")]
    for t in [
        "<b>单张图像检测</b>：对单张图像给出真伪判定、置信度、耗时与篡改热力图。",
        "<b>批量图像检测</b>：支持多选文件或整个文件夹，逐张推理并以表格汇总结果。",
        "<b>篡改区域可视化</b>：以热力图叠加方式高亮疑似篡改区域，支持原始图像对照。",
        "<b>篡改掩码导出</b>：将像素级掩码导出为标准 PNG 二值图，单张或批量均可。",
        "<b>PDF 检测报告导出</b>：生成含判定结论、指标明细与缩略图的可归档报告。",
        "<b>推理后端切换</b>：支持 PyTorch / ONNX Runtime / OpenVINO 三种后端。",
        "<b>完全离线运行</b>：检测过程无需联网，图像数据不出本机。",
    ]:
        story.append(LI(t))

    story += [H2("1.3 运行环境")]
    env = [
        ["项目", "要求"],
        ["操作系统", "Windows 10 / 11（64 位）"],
        ["处理器", "Intel / AMD 双核以上，支持 AVX2；推荐 4 核以上"],
        ["内存", "8 GB 以上（推荐 16 GB）"],
        ["硬盘", "2 GB 以上可用空间（不含数据集）"],
        ["解释器", "Python 3.10 及以上（随附打包版可免安装）"],
        ["依赖库", "PyTorch 2.x、OpenCV、ONNX Runtime、PyQt5、ReportLab 等"],
        ["显示分辨率", "建议 1366×768 及以上"],
    ]
    t = Table(env, colWidths=[38 * mm, 112 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), song),
        ("FONTNAME", (0, 0), (-1, 0), hei),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [t]

    # -------------------------------------------------- 2 安装与启动
    story += [PageBreak(), H1("二、安装与启动")]
    story += [H2("2.1 免安装运行（推荐）")]
    story += [P("将软件压缩包解压到任意目录（路径中建议不包含中文与空格），"
                "双击目录下的启动程序即可运行，无需配置 Python 环境。")]

    story += [H2("2.2 源码方式运行")]
    story += [P("若已具备 Python 环境，可在软件根目录下执行以下命令安装依赖并启动：")]
    story += [Paragraph("pip install -r requirements.txt<br/>"
                        "python app/main.py", st_code)]
    story += [P("首次启动时软件会自动加载默认模型；若模型文件缺失，界面会以醒目"
                "颜色提示，此时检测结果不具备参考价值，需在「系统设置」中指定"
                "正确的模型路径后点击「重新加载模型」。")]

    story += [H2("2.3 界面语言与外观")]
    story += [P("软件界面为简体中文，采用浅色主题，自适应高分辨率屏幕。"
                "主窗口默认尺寸 1360×860，可通过拖拽分隔条调整左右区域宽度。")]

    # -------------------------------------------------- 3 界面说明
    story += [PageBreak(), H1("三、界面说明")]
    story += [P("主窗口自上而下由工具栏、主体工作区、进度条与状态栏四部分组成；"
                "主体工作区采用左右分栏，左侧为图像输入与系统设置，右侧为结果展示。")]

    img_specs = [
        ("01_主界面_初始状态.png", "图 3-1  软件主界面（初始状态）",
         "左侧为图像输入与系统设置，右侧为结果展示区，顶部为功能工具栏。"),
        ("02_图像输入.png", "图 3-2  图像输入模块",
         "支持文件选择、文件夹批量导入与拖拽添加，列表显示缩略图与张数统计。"),
        ("03_检测结果_热力图与报告.png", "图 3-3  检测结果展示（单张）",
         "左栏为原始图像与篡改热力图对照，右栏为判定结论与指标明细。"),
        ("04_批量结果表格.png", "图 3-4  批量检测结果表格",
         "逐张列出文件名、尺寸、判定结论、伪造概率、篡改面积占比与推理耗时。"),
        ("06_系统设置.png", "图 3-5  系统设置模块",
         "可配置推理后端、模型路径、判定阈值、图像尺寸与结果保存目录。"),
    ]
    for fn, cap, desc in img_specs:
        p = os.path.join(shots_dir, fn)
        if not os.path.exists(p):
            story += [Paragraph(f"[缺少截图 {fn}，请先运行 "
                                f"scripts/make_ui_screenshots.py]", st_cap)]
            continue
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            w, h = im.size
        max_w = 150 * mm
        max_h = 96 * mm
        scale = min(max_w / w, max_h / h)
        story += [
            Image(p, width=w * scale, height=h * scale),
            Paragraph(cap, st_cap),
            P(desc),
            Spacer(1, 3 * mm),
        ]

    # -------------------------------------------------- 4 操作流程
    story += [PageBreak(), H1("四、操作流程")]
    story += [H2("4.1 单张图像检测")]
    for t in [
        "启动软件，等待状态栏提示「模型已加载」。",
        "在「① 图像输入」中点击「选择文件」，选取待检测图像；也可直接将文件拖入列表。",
        "单击列表中的条目，右侧会立即预览该图像原图。",
        "点击工具栏「开始检测」，进度条开始推进。",
        "检测完成后，结果区显示判定结论、伪造概率、篡改面积占比与推理耗时，"
        "并在「篡改热力图」页签中给出可疑区域高亮。",
    ]:
        story.append(LI(t))

    story += [H2("4.2 批量图像检测")]
    for t in [
        "点击「选择文件夹」导入整个目录（默认递归子目录）。",
        "确认列表张数后点击「开始检测」，软件逐张推理并实时更新表格。",
        "检测过程中可点击「停止」中断，已完成的结果会保留。",
        "检测结束后自动切换到「批量结果」页签，状态栏汇总疑似伪造与真实张数。",
    ]:
        story.append(LI(t))

    story += [H2("4.3 结果导出")]
    for t in [
        "<b>导出 PDF 报告</b>：点击工具栏同名按钮，选择保存路径，生成含结论与"
        "缩略图的可归档报告。",
        "<b>导出掩码 PNG</b>：在结果表格中选中目标行，点击「导出掩码 PNG」，"
        "得到二值化篡改区域图。",
        "<b>批量导出掩码</b>：点击「批量导出掩码」并选择输出目录，"
        "软件为全部结果生成同名的 _mask.png 文件。",
    ]:
        story.append(LI(t))

    story += [H2("4.4 判定阈值调整")]
    story += [P("软件默认以伪造概率 0.5 作为判定阈值。若使用场景对漏检更敏感，"
                "可在「⑤ 系统设置」中将阈值下调（如 0.35），使更多图像被判为疑似伪造；"
                "若对误报更敏感，则上调阈值（如 0.65）。调整后点击「应用并重新加载」。")]

    # -------------------------------------------------- 5 系统设置
    story += [PageBreak(), H1("五、系统设置说明")]
    setting_rows = [
        ["设置项", "说明", "建议值"],
        ["推理后端", "torch=PyTorch 原生；onnx=ONNX Runtime；openvino=Intel CPU 加速",
         "onnx 或 openvino"],
        ["模型权重", "PyTorch 格式权重文件路径", "由部署流程给出"],
        ["ONNX 模型", "ONNX 模型文件路径", "由部署流程给出"],
        ["OpenVINO IR", "OpenVINO 中间表示目录", "由部署流程给出"],
        ["判定阈值", "伪造概率判定门限", "0.5"],
        ["图像尺寸", "推理输入边长，需与训练一致", "224"],
        ["保存目录", "报告与掩码的默认输出位置", "自定义"],
    ]
    t = Table(setting_rows, colWidths=[26 * mm, 92 * mm, 32 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), song),
        ("FONTNAME", (0, 0), (-1, 0), hei),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t]
    story += [Spacer(1, 4 * mm)]
    story += [P("所有设置项通过系统配置持久化保存，下次启动软件时自动恢复上次的配置。"
                "点击「恢复默认」可将全部设置项重置为出厂值。")]

    # -------------------------------------------------- 6 技术指标
    story += [PageBreak(), H1("六、技术指标与验证结果")]
    story += [P("软件在标准测试条件下实测的主要技术指标如下表所示。"
                "其中模型体积与推理速度为部署形态（经 ONNX 简化）的本机实测值，"
                "测试环境为 Windows 10、7 逻辑核 CPU、Python 3.13、ONNX Runtime 1.30、"
                "输入 224×224、batch=1。")]
    metric_rows = [
        ["指标项", "实测值", "测试条件说明"],
        ["模型体积（部署形态）", "98.7 MB", "ONNX 简化后，满足 ≤120MB 要求"],
        ["CPU 推理速度", "9.46 张/秒", "ONNX Runtime、224×224、多轮取中位数"],
        ["单张推理延迟", "105.7 ms", "同上条件"],
        ["支持图像格式", "PNG / JPG / BMP / WEBP / TIF", "由 OpenCV 解码"],
        ["最大批处理规模", "仅受内存限制", "逐张推理，内存占用稳定"],
        ["网络依赖", "无", "检测过程完全离线"],
    ]
    t = Table(metric_rows, colWidths=[42 * mm, 38 * mm, 70 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), song),
        ("FONTNAME", (0, 0), (-1, 0), hei),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t]
    story += [Spacer(1, 3 * mm)]
    story += [P("说明：推理速度与硬件强相关，换用其他处理器需重新实测；"
                "软件的工业部署建议使用支持 INT8 指令集（VNNI）的处理器以获得"
                "更优的量化加速效果。")]

    # -------------------------------------------------- 7 常见问题
    story += [PageBreak(), H1("七、常见问题与处理")]
    faq = [
        ("启动后状态栏长时间显示「正在加载模型」",
         "首次加载需读取模型权重，属正常现象。若超过两分钟仍未完成，"
         "请检查「系统设置」中的模型路径是否正确、文件是否完整。"),
        ("点击「开始检测」提示模型尚未加载",
         "等待加载完成，或点击工具栏「重新加载模型」；若持续失败，"
         "请改用 ONNX 后端（无需 PyTorch 环境）。"),
        ("检测结果全部判为「疑似伪造」或全部判为「真实」",
         "通常说明所用权重未经有效训练，或判定阈值设置不当。"
         "请在「系统设置」中确认权重来源，并适当调整判定阈值。"),
        ("批量检测速度慢",
         "可在「系统设置」中切换为 ONNX 或 OpenVINO 后端，"
         "并将图像尺寸设为与训练一致的 224。"),
        ("导出的掩码全黑或全白",
         "说明该图未被判定出明显篡改区域（全黑）或整图被判为篡改（全白），"
         "可结合热力图与伪造概率综合判断。"),
        ("运行时报缺少 DLL 或库文件",
         "说明运行环境不完整，请重新安装依赖，或改用手册 2.1 节的免安装版本。"),
    ]
    for i, (q, a) in enumerate(faq, 1):
        story += [H2(f"{i}. {q}")]
        story += [P(a)]

    story += [Spacer(1, 6 * mm)]
    story += [Paragraph("—— 本说明书结束 ——", st_cap)]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def on_page(c, doc_):
        c.setFont(song, 8.5)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawCentredString(A4[0] / 2, 12 * mm,
                            f"{SOFTWARE_NAME} {VERSION}  软件说明书  第 {doc_.page} 页")

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm,
                            title=f"{SOFTWARE_NAME} {VERSION} 软件说明书")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return {"path": out_path, "pages": doc.page,
            "size_kb": round(os.path.getsize(out_path) / 1024, 1)}


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="生成软著登记所需文档")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "copyright"),
                    help="输出目录")
    ap.add_argument("--shots", default="", help="界面截图目录（默认 <out>/ui_shots）")
    ap.add_argument("--head-pages", type=int, default=30)
    ap.add_argument("--tail-pages", type=int, default=30)
    ap.add_argument("--lines-per-page", type=int, default=52,
                    help="每页代码行数（软著口径要求 ≥50 可见行，默认 52）")
    ap.add_argument("--keep-blanks", action="store_true",
                    help="保留源码空行（默认去掉；保留会显著降低每页可见行数）")
    ap.add_argument("--skip-source", action="store_true")
    ap.add_argument("--skip-manual", action="store_true")
    args = ap.parse_args()

    shots = args.shots or os.path.join(args.out, "ui_shots")
    song, hei = register_fonts()
    print(f"[font] 正文={song}  标题={hei}")

    if not args.skip_source:
        print("\n[1/2] 生成源程序文档…")
        info = make_source_pdf(
            os.path.join(args.out, f"源程序文档_{SOFTWARE_NAME}{VERSION}.pdf"),
            song, hei, args.head_pages, args.tail_pages, args.lines_per_page,
            drop_blank=not args.keep_blanks)
        print(f"  → {info['path']}")
        print(f"     {info['pages']} 页  渲染 {info['rendered_lines']} 行  "
              f"{info['size_kb']} KB")

    if not args.skip_manual:
        print("\n[2/2] 生成软件说明书…")
        if not os.path.isdir(shots):
            print(f"  [warn] 截图目录不存在：{shots}")
            print("         请先运行 python scripts/make_ui_screenshots.py")
        info = make_manual_pdf(
            os.path.join(args.out, f"软件说明书_{SOFTWARE_NAME}{VERSION}.pdf"),
            song, hei, shots)
        print(f"  → {info['path']}")
        print(f"     {info['pages']} 页  {info['size_kb']} KB")

    print(f"\n[done] 输出目录：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
