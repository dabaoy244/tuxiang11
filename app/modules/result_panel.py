"""桌面工具 · 结果展示模块

分栏布局（申报书 3.2）：左侧原始图像 / 中间篡改区域热力图 / 右侧检测结果报告；
下方为批量检测结果表格，点击某一行可回到该图的详细视图。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)


def ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """numpy -> QPixmap。支持 (H,W) 灰度 / (H,W,3) BGR / (H,W,4)。"""
    if img is None:
        return QPixmap()
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        import cv2

        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        h, w, _ = arr.shape
        qimg = QImage(np.ascontiguousarray(arr).data, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        return QPixmap()
    return QPixmap.fromImage(qimg.copy())


class ImageView(QLabel):
    """自适应缩放的图像显示控件。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self.setMinimumSize(260, 260)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#f5f6f8;border:1px solid #dcdfe6;border-radius:6px;")
        self.setText(f"{title}\n（暂无结果）")
        self.setWordWrap(True)

    def set_image(self, img: np.ndarray) -> None:
        self._pix = ndarray_to_qpixmap(img)
        self._rescale()

    def clear_image(self) -> None:
        self._pix = None
        self.setText("（暂无结果）")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        self.setPixmap(self._pix.scaled(self.size() - self.size() * 0.04,
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation))


def read_image_bgr(path: str) -> Optional[np.ndarray]:
    """读取图像为 BGR ndarray（兼容中文路径）。"""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
    return img


class ResultPanel(QGroupBox):
    """③ 结果展示模块。"""

    def __init__(self, parent=None):
        super().__init__("③ 结果展示", parent)
        self._results: List[object] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()

        # ---- 详细视图
        detail = QWidget()
        dl = QVBoxLayout(detail)
        split = QSplitter(Qt.Horizontal)

        self.view_orig = ImageView("原始图像")
        self.view_heat = ImageView("篡改区域高亮（热力图，越暖越可疑）")
        for v in (self.view_orig, self.view_heat):
            split.addWidget(v)

        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setMinimumWidth(260)
        self.txt_report.setPlaceholderText("检测结果报告将显示在这里")
        split.addWidget(self.txt_report)
        split.setSizes([320, 320, 300])
        dl.addWidget(split)
        self.tabs.addTab(detail, "详细视图")

        # ---- 批量表格
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["文件名", "尺寸", "结论", "伪造置信度", "可疑区域占比", "耗时(ms)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row_changed)
        self.tabs.addTab(self.table, "批量结果")
        root.addWidget(self.tabs)

    # ------------------------------------------------------------------
    def show_result(self, res, keep_table: bool = True) -> None:
        """显示单张图像的检测结果。"""
        self.view_orig.set_image(read_image_bgr(res.path))
        self.view_heat.set_image(res.overlay)

        color = "#c0392b" if res.is_fake else "#1e8e3e"
        lines = [
            f"<h3 style='color:{color};margin:4px 0;'>{res.label}</h3>",
            "<table cellspacing='0' cellpadding='4'>",
        ]
        for k, v in res.summary().items():
            lines.append(
                f"<tr><td style='color:#666;'>{k}</td>"
                f"<td><b>{v}</b></td></tr>")
        lines.append("</table>")
        if res.warnings:
            lines.append("<p style='color:#e67e22;'>" + "<br>".join(res.warnings) + "</p>")
        lines.append(
            "<p style='color:#999;font-size:11px;margin-top:10px;'>"
            "结论由模型自动给出，仅供辅助参考，不作为司法鉴定依据。</p>"
        )
        self.txt_report.setHtml("".join(lines))

    def add_row(self, res) -> None:
        from PyQt5.QtGui import QColor

        self._results.append(res)
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [
            res.name, f"{res.width}x{res.height}", res.label,
            f"{res.prob_fake:.4f}", f"{(res.mask > 0.5).mean():.4f}",
            f"{res.time_ms:.1f}",
        ]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(str(v))
            if c == 2:
                item.setForeground(QColor("#c0392b" if res.is_fake else "#1e8e3e"))
            self.table.setItem(r, c, item)

    def _on_row_changed(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows or not self._results:
            return
        idx = min(rows)
        if idx < len(self._results):
            self.show_result(self._results[idx])

    def clear(self) -> None:
        self._results.clear()
        self.table.setRowCount(0)
        self.view_orig.clear_image()
        self.view_heat.clear_image()
        self.txt_report.clear()

    def results(self) -> List[object]:
        return list(self._results)

    @property
    def summary_counts(self) -> Dict[str, int]:
        fake = sum(1 for r in self._results if r.is_fake)
        return {"fake": fake, "real": len(self._results) - fake, "total": len(self._results)}
