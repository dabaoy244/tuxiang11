"""桌面工具 · 结果导出模块

导出内容（申报书 3.2）：
    ① PDF 检测报告：原始图像 + 篡改区域高亮图 + 检测结果信息
    ② PNG 篡改掩码图：白色=篡改区域，黑色=真实区域

PDF 生成方案：优先使用 PyQt5 自带的 QtPrintSupport（QTextDocument -> QPrinter），
零额外依赖；未安装 QtPrintSupport 时回退到 reportlab。
"""

from __future__ import annotations

import html
import os
import tempfile
import time
from typing import List, Optional

import numpy as np


def _save_png_cn(img: np.ndarray, path: str) -> str:
    """保存 PNG，兼容中文路径。"""
    import cv2

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"编码失败：{path}")
    buf.tofile(path)
    return path


def export_mask_png(mask: np.ndarray, path: str) -> str:
    """导出篡改掩码 PNG（白=篡改，黑=真实）。"""
    binary = (np.clip(mask, 0, 1) > 0.5).astype(np.uint8) * 255
    if not path.lower().endswith(".png"):
        path += ".png"
    return _save_png_cn(binary, path)


def export_heatmap_png(overlay_bgr: np.ndarray, path: str) -> str:
    if not path.lower().endswith(".png"):
        path += ".png"
    return _save_png_cn(overlay_bgr, path)


# ==========================================================================
def _build_html(results: List[object], title: str = "AI 图像篡改检测报告") -> str:
    rows = []
    n_fake = sum(1 for r in results if r.is_fake)
    for i, r in enumerate(results, 1):
        color = "#c0392b" if r.is_fake else "#1e8e3e"
        rows.append(
            f"<tr><td>{i}</td><td>{html.escape(r.name)}</td>"
            f"<td>{r.width}x{r.height}</td>"
            f"<td style='color:{color};font-weight:bold'>{r.label}</td>"
            f"<td>{r.prob_fake:.4f}</td>"
            f"<td>{float((r.mask > 0.5).mean()):.4f}</td>"
            f"<td>{r.time_ms:.1f}</td></tr>")

    detail = []
    for i, r in enumerate(results[:20], 1):     # 详细页最多 20 张，避免报告过长
        detail.append(f"""
        <div style="margin-top:14px;">
          <div style="font-weight:bold;color:{'#c0392b' if r.is_fake else '#1e8e3e'};">
            {i}. {html.escape(r.name)} —— {r.label}（伪造置信度 {r.prob_fake:.4f}）
          </div>
        </div>""")

    return f"""<html><head><meta charset="utf-8"></head><body
      style="font-family:'Microsoft YaHei',sans-serif;font-size:11pt;">
      <h2 style="margin-bottom:4px;">{title}</h2>
      <p style="color:#666;margin-top:0;">
        生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　|　
        受检图像 {len(results)} 张，其中疑似伪造 {n_fake} 张，判定为真实 {len(results)-n_fake} 张
      </p>
      <table border="1" cellspacing="0" cellpadding="5"
             style="border-collapse:collapse;width:100%;font-size:10pt;">
        <tr style="background:#eef1f6;">
          <th>#</th><th>文件名</th><th>尺寸</th><th>结论</th>
          <th>伪造置信度</th><th>可疑区域占比</th><th>耗时(ms)</th>
        </tr>
        {''.join(rows)}
      </table>
      <p style="color:#999;font-size:9pt;margin-top:16px;">
        说明：结论由本地运行的轻量 VIB-Net 模型自动给出，仅供辅助参考，
        不作为司法鉴定依据。本报告全部在本地生成，图像数据未离开本机。
      </p>
      </body></html>"""


def export_pdf_report(results: List[object], path: str,
                      title: str = "AI 图像篡改检测报告",
                      include_images: bool = True) -> str:
    """导出 PDF 检测报告。"""
    if not results:
        raise ValueError("没有可导出的检测结果")
    if not path.lower().endswith(".pdf"):
        path += ".pdf"

    body = _build_html(results, title)
    try:
        return _export_pdf_qt(results, path, body, include_images)
    except Exception as e:  # noqa: BLE001
        print(f"[export] Qt 导出 PDF 失败（{e}），尝试 reportlab")
        return _export_pdf_reportlab(results, path, title, include_images)


def _export_pdf_qt(results: List[object], path: str, body: str,
                   include_images: bool) -> str:
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QTextDocument
    from PyQt5.QtPrintSupport import QPrinter
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("需要先创建 QApplication")

    tmpdir = tempfile.mkdtemp(prefix="vibnet_report_")
    img_html = ""
    if include_images:
        for i, r in enumerate(results[:20], 1):     # 图录限 20 张
            from app.modules.result_panel import read_image_bgr

            orig = read_image_bgr(r.path)
            if orig is None:
                continue
            op = os.path.join(tmpdir, f"{i}_orig.png").replace("\\", "/")
            hp = os.path.join(tmpdir, f"{i}_heat.png").replace("\\", "/")
            _save_png_cn(orig, op)
            _save_png_cn(r.overlay, hp)
            img_html += f"""
            <div style="margin-top:16px;">
              <div style="font-size:10pt;font-weight:bold;">
                {i}. {html.escape(r.name)} —— {r.label}（{r.prob_fake:.4f}）
              </div>
              <table style="width:100%;margin-top:6px;"><tr>
                <td style="width:50%;"><img src="file:///{op}" width="330"></td>
                <td style="width:50%;"><img src="file:///{hp}" width="330"></td>
              </tr>
              <tr style="font-size:9pt;color:#666;">
                <td align="center">原始图像</td><td align="center">篡改区域高亮</td>
              </tr></table>
            </div>"""

    doc = QTextDocument()
    doc.setHtml(body.replace("</body>", img_html + "</body>"))

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(path)
    doc.print_(printer)
    return path


def _export_pdf_reportlab(results: List[object], path: str, title: str,
                          include_images: bool) -> str:
    """reportlab 回退实现。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font = "STSong-Light"
    except Exception:
        font = "Helvetica"

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    y = h - 20 * mm
    c.setFont(font, 15)
    c.drawString(20 * mm, y, title)
    y -= 8 * mm
    c.setFont(font, 9)
    c.drawString(20 * mm, y, f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　"
                             f"受检 {len(results)} 张")
    y -= 8 * mm
    c.setFont(font, 9)
    for i, r in enumerate(results, 1):
        if y < 30 * mm:
            c.showPage()
            c.setFont(font, 9)
            y = h - 20 * mm
        c.drawString(20 * mm, y, f"{i}. {r.name}  {r.width}x{r.height}  "
                                 f"{r.label}  p={r.prob_fake:.4f}  "
                                 f"{(r.mask > 0.5).mean():.4f}  {r.time_ms:.1f}ms")
        y -= 6 * mm
    c.save()
    return path


def export_batch_masks(results: List[object], out_dir: str) -> List[str]:
    """批量导出掩码 PNG 到指定目录。"""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for r in results:
        base = os.path.splitext(r.name)[0]
        p = os.path.join(out_dir, f"{base}_mask.png")
        out.append(export_mask_png(r.mask, p))
    return out
