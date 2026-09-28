"""桌面工具 · 图像输入模块

支持：单张上传 / 批量上传 / 文件夹遍历 / 拖拽上传 / 列表删除。
输出：`self.paths()` 返回待检测图像路径列表。
"""

from __future__ import annotations

import os
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FILTER = "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"


class ThumbListWidget(QListWidget):
    """支持拖拽文件的列表控件。"""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setIconSize(self.iconSize())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        paths: List[str] = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                paths += collect_images(p)
            elif p.lower().endswith(SUPPORTED_EXT):
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


def collect_images(folder: str, recursive: bool = True) -> List[str]:
    out: List[str] = []
    if recursive:
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(SUPPORTED_EXT):
                    out.append(os.path.join(root, fn))
    else:
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(SUPPORTED_EXT):
                out.append(os.path.join(folder, fn))
    return sorted(out)


class InputPanel(QGroupBox):
    """① 图像输入模块。"""

    selection_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("① 图像输入", parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        btns = QHBoxLayout()
        self.btn_single = QPushButton("上传单张图片")
        self.btn_batch = QPushButton("批量上传图片")
        self.btn_folder = QPushButton("选择文件夹")
        self.btn_clear = QPushButton("清空列表")
        for b in (self.btn_single, self.btn_batch, self.btn_folder):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.btn_clear)
        root.addLayout(btns)

        self.list = ThumbListWidget()
        self.list.setMinimumHeight(180)
        self.list.setToolTip("可直接把图片或文件夹拖拽到这里")
        root.addWidget(self.list)

        self.lbl_count = QLabel("共 0 张待检测图像")
        self.lbl_count.setStyleSheet("color:#666;")
        root.addWidget(self.lbl_count)

        self.btn_single.clicked.connect(self.pick_single)
        self.btn_batch.clicked.connect(self.pick_batch)
        self.btn_folder.clicked.connect(self.pick_folder)
        self.btn_clear.clicked.connect(self.clear)
        self.list.files_dropped.connect(self.add_paths)
        self.list.itemSelectionChanged.connect(self._on_selection)

    # ------------------------------------------------------------------
    def pick_single(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图像", "", FILTER)
        self.add_paths(paths)

    def pick_batch(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "批量选择图像", "", FILTER)
        self.add_paths(paths)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.add_paths(collect_images(folder))

    def add_paths(self, paths: List[str]) -> None:
        exist = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        added = 0
        for p in paths:
            if p in exist or not os.path.exists(p):
                continue
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            item.setIcon(self._thumb(p))
            self.list.addItem(item)
            added += 1
        if added:
            self._refresh_count()

    @staticmethod
    def _thumb(path: str):
        from PyQt5.QtGui import QIcon

        pm = QPixmap(path)
        if pm.isNull():
            return QIcon()
        return QIcon(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def clear(self) -> None:
        self.list.clear()
        self._refresh_count()

    def _refresh_count(self) -> None:
        n = self.list.count()
        self.lbl_count.setText(f"共 {n} 张待检测图像")
        self.selection_changed.emit(n)

    def _on_selection(self) -> None:
        self.selection_changed.emit(self.list.count())

    def paths(self) -> List[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def selected_path(self) -> str:
        items = self.list.selectedItems()
        return items[0].data(Qt.UserRole) if items else ""

    def count(self) -> int:
        return self.list.count()
