"""空频双分支轻量 VIB-Net —— Windows 离线桌面检测工具

启动方式（在项目根目录下）：
    python app/main.py

界面布局（对应申报书 3.2 五大功能模块）：
    ┌──────────────── 顶部工具栏：开始检测 / 停止 / 导出 ────────────────┐
    │ ① 图像输入（左）                ③ 结果展示（右，分栏）            │
    │ ⑤ 系统设置（左）                · 原始图像 / 热力图 / 报告         │
    │                                 · 批量结果表格                    │
    ├──────────────────────── 进度条 + 状态栏 ──────────────────────────┤
    └──────────────────────────────────────────────────────────────────┘
工具完全离线运行，所有计算在本机完成，图像数据不出本机。
"""

from __future__ import annotations

import os
import sys

# ---- 让 app/ 与项目根目录都在 import 路径上（支持 pyinstaller 打包）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from app.modules.exporter import export_batch_masks, export_mask_png, export_pdf_report  # noqa: E402
from app.modules.input_panel import InputPanel  # noqa: E402
from app.modules.result_panel import ResultPanel  # noqa: E402
from app.modules.settings_panel import SettingsPanel  # noqa: E402
from app.modules.workers import BatchDetectWorker, DetectorLoader  # noqa: E402


STYLE = """
QMainWindow, QWidget { background:#fafbfc; color:#222; }
QGroupBox { font-weight:600; border:1px solid #dcdfe6; border-radius:8px;
            margin-top:10px; padding:10px 8px 8px 8px; background:#fff; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#2c3e50; }
QPushButton { background:#3b6ef0; color:#fff; border:none; border-radius:6px;
              padding:6px 14px; font-weight:500; }
QPushButton:hover { background:#2f5dd8; }
QPushButton:disabled { background:#c3cad6; }
QPushButton#ghost { background:#eef1f6; color:#2c3e50; }
QPushButton#ghost:hover { background:#e2e7f0; }
QTableWidget { gridline-color:#eef1f6; }
QHeaderView::section { background:#f2f4f8; border:none; padding:6px; font-weight:600; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("空频双分支 VIB-Net · AI 图像篡改检测系统 V1.0")
        self.resize(1360, 860)

        self.detector = None
        self.loader = None
        self.worker = None

        self._build_ui()
        self.setStyleSheet(STYLE)

        # 启动后自动加载模型，省去用户手动点击
        QTimer.singleShot(300, self.reload_detector)

    # ==================================================================
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- 顶部工具栏
        bar = QHBoxLayout()
        self.btn_start = QPushButton("开始检测")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("ghost")
        self.btn_stop.setEnabled(False)
        self.btn_pdf = QPushButton("导出 PDF 报告")
        self.btn_mask = QPushButton("导出掩码 PNG")
        self.btn_masks = QPushButton("批量导出掩码")
        self.btn_reload = QPushButton("重新加载模型")
        for b in (self.btn_pdf, self.btn_mask, self.btn_masks, self.btn_reload):
            b.setObjectName("ghost")
        for b in (self.btn_start, self.btn_stop):
            bar.addWidget(b)
        bar.addSpacing(16)
        for b in (self.btn_pdf, self.btn_mask, self.btn_masks):
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(self.btn_reload)
        root.addLayout(bar)

        # ---- 主体：左（输入 + 设置） / 右（结果）
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.input_panel = InputPanel()
        self.settings_panel = SettingsPanel(project_root=ROOT)
        lv.addWidget(self.input_panel, 3)
        lv.addWidget(self.settings_panel, 4)

        self.result_panel = ResultPanel()
        splitter.addWidget(left)
        splitter.addWidget(self.result_panel)
        splitter.setSizes([430, 900])
        root.addWidget(splitter, 1)

        # ---- 进度条 + 状态栏
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = QLabel("正在初始化…")
        sb.addWidget(self.lbl_status)

        # ---- 信号
        self.btn_start.clicked.connect(self.start_detect)
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_pdf.clicked.connect(self.save_pdf)
        self.btn_mask.clicked.connect(self.save_mask)
        self.btn_masks.clicked.connect(self.save_masks)
        self.btn_reload.clicked.connect(self.reload_detector)
        self.settings_panel.applied.connect(lambda _cfg: self.reload_detector())
        self.input_panel.list.currentItemChanged.connect(self._preview_current)

    # ==================================================================
    def log(self, text: str, color: str = "#666") -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color:{color};")

    # ------------------------------------------------------------------
    def reload_detector(self) -> None:
        cfg = self.settings_panel.get_config()
        self.log(f"正在加载模型（后端={cfg['backend']}）…", "#e67e22")
        self.settings_panel.set_status("加载中…", "#e67e22")
        self.btn_start.setEnabled(False)
        self.detector = None

        self.loader = DetectorLoader({
            "backend": cfg["backend"],
            "image_size": cfg["image_size"],
            "threshold": cfg["threshold"],
            "ckpt": cfg["ckpt"],
            "onnx_path": cfg["onnx_path"],
            "openvino_ir": cfg["openvino_ir"],
            "config_path": cfg["config_path"],
            "device": cfg["device"],
        })
        self.loader.loaded.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()

    def _on_loaded(self, det) -> None:
        self.detector = det
        note = getattr(det, "_fallback_note", None)
        if note:
            self.log(f"模型已加载（{det.backend}）⚠ {note}", "#e67e22")
            self.settings_panel.set_status("已加载（未找到权重，随机初始化）", "#e67e22")
        else:
            self.log(f"模型已加载，后端={det.backend}，阈值={det.threshold}", "#1e8e3e")
            self.settings_panel.set_status(f"已加载 · {det.backend}", "#1e8e3e")
        self.btn_start.setEnabled(True)

    def _on_load_failed(self, err: str) -> None:
        self.log("模型加载失败，请检查设置中的模型路径", "#c0392b")
        self.settings_panel.set_status("加载失败", "#c0392b")
        QMessageBox.warning(self, "模型加载失败", err)
        self.btn_start.setEnabled(True)

    # ==================================================================
    def _preview_current(self) -> None:
        """单击列表时预览原图（不推理）。"""
        p = self.input_panel.selected_path()
        if not p:
            return
        from app.modules.result_panel import read_image_bgr

        img = read_image_bgr(p)
        if img is not None:
            self.result_panel.view_orig.set_image(img)

    # ------------------------------------------------------------------
    def start_detect(self) -> None:
        paths = self.input_panel.paths()
        if not paths:
            QMessageBox.information(self, "提示", "请先添加待检测图像（支持拖拽）。")
            return
        if self.detector is None:
            QMessageBox.information(self, "提示", "模型尚未加载完成，请稍候或点击“重新加载模型”。")
            return

        self.result_panel.clear()
        self.progress.setMaximum(len(paths))
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log(f"开始检测 {len(paths)} 张图像…", "#3b6ef0")

        self.worker = BatchDetectWorker(self.detector, paths)
        self.worker.progress.connect(self._on_progress)
        self.worker.one_done.connect(self._on_one_done)
        self.worker.finished_all.connect(self._on_all_done)
        self.worker.failed.connect(lambda e: self.log(e, "#c0392b"))
        self.worker.start()

    def _on_progress(self, i: int, total: int, name: str) -> None:
        self.progress.setValue(i)
        self.progress.setFormat(f"%v/%m  {name}")

    def _on_one_done(self, res) -> None:
        self.result_panel.add_row(res)
        self.result_panel.show_result(res)      # 实时显示最新一张

    def _on_all_done(self, ok: list, bad: list) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.worker = None
        c = self.result_panel.summary_counts
        msg = f"检测完成：共 {c['total']} 张，疑似伪造 {c['fake']} 张，真实 {c['real']} 张"
        if bad:
            msg += f"，失败 {len(bad)} 张"
        self.log(msg, "#1e8e3e" if not bad else "#e67e22")
        self.result_panel.tabs.setCurrentIndex(1)

    def stop_detect(self) -> None:
        if self.worker is not None:
            self.worker.requestInterruption()
            self.log("已请求停止，等待当前图像完成…", "#e67e22")

    # ==================================================================
    def save_pdf(self) -> None:
        res = self.result_panel.results()
        if not res:
            QMessageBox.information(self, "提示", "还没有检测结果。")
            return
        cfg = self.settings_panel.get_config()
        default = os.path.join(cfg["save_dir"], "检测报告.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "导出 PDF 报告", default, "PDF (*.pdf)")
        if not path:
            return
        try:
            export_pdf_report(res, path)
            self.log(f"PDF 报告已导出：{path}", "#1e8e3e")
            QMessageBox.information(self, "导出成功", f"报告已保存到：\n{path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(e))

    def save_mask(self) -> None:
        res = self.result_panel.results()
        if not res:
            QMessageBox.information(self, "提示", "还没有检测结果。")
            return
        cur = self.result_panel.table.currentRow()
        r = res[cur] if 0 <= cur < len(res) else res[-1]
        cfg = self.settings_panel.get_config()
        default = os.path.join(cfg["save_dir"], os.path.splitext(r.name)[0] + "_mask.png")
        path, _ = QFileDialog.getSaveFileName(self, "导出篡改掩码", default, "PNG (*.png)")
        if not path:
            return
        try:
            export_mask_png(r.mask, path)
            self.log(f"掩码已导出：{path}", "#1e8e3e")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(e))

    def save_masks(self) -> None:
        res = self.result_panel.results()
        if not res:
            QMessageBox.information(self, "提示", "还没有检测结果。")
            return
        cfg = self.settings_panel.get_config()
        out_dir = QFileDialog.getExistingDirectory(self, "选择掩码输出目录", cfg["save_dir"])
        if not out_dir:
            return
        try:
            files = export_batch_masks(res, out_dir)
            self.log(f"已批量导出 {len(files)} 张掩码到 {out_dir}", "#1e8e3e")
            QMessageBox.information(self, "导出成功", f"共导出 {len(files)} 张掩码图。")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(e))


# ==========================================================================
def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
