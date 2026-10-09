"""桌面工具 · 图像输入模块

支持：单张上传 / 批量上传 / 文件夹遍历 / 拖拽上传 / 列表删除。
输出：`self.paths()` 返回待检测图像路径列表。
"""

from __future__ import annotations

import os
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from app.modules.workers import ThumbWorker

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FILTER = "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"

#: ★ 2026-10-09（A1）：单次载入的列表上限。
#: 为什么要上限：选 `data/Datasets` 这种根目录一次能命中 **132,817 张**，
#: 即使缩略图已挪到后台（见 `ThumbWorker`），逐张解码也要约 **84 分钟**才做完，
#: 而批量检测本身在 11 张/秒下要 3 小时以上 —— 一次全收没有实际意义。
#: 到上限会**明确告诉用户**（计数栏写"已载入前 N 张"），不是静默截断。
MAX_LIST_ITEMS = 2000


class ThumbListWidget(QListWidget):
    """支持拖拽文件的列表控件。"""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setIconSize(self.iconSize())
        #: 由 `InputPanel` 注入的"规模前置确认"钩子：收到"这个文件夹里有 N 张"，
        #: 返回是否继续载入。见 `InputPanel._confirm_large`。
        self.front_check = None

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
                got = collect_images(p)
                # 文件夹是"规模失控"的唯一入口（一次能带进 13 万张）⇒ 先问一句
                if self.front_check is not None and not self.front_check(len(got)):
                    continue
                paths += got
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
        #: 后台缩略图线程与其"代次号"（见 ThumbWorker 的 gen 说明）
        self._thumb_worker = None
        self._thumb_gen = 0
        #: 上一次 add_paths 的入参张数 / 是否被上限截断（供计数栏明示）
        self._requested = 0
        self._hit_cap = False
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
        self.list.files_dropped.connect(self._add_with_confirm)
        self.list.front_check = self._confirm_large
        self.list.itemSelectionChanged.connect(self._on_selection)

    # ------------------------------------------------------------------
    def pick_single(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图像", "", FILTER)
        self._add_with_confirm(paths)

    def pick_batch(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "批量选择图像", "", FILTER)
        self._add_with_confirm(paths)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self._add_with_confirm(collect_images(folder))

    # ------------------------------------------------------------------
    def _confirm_large(self, n: int) -> bool:
        """规模**前置**确认：超过单次上限就先问一句，返回是否继续。

        为什么要它：上限（`MAX_LIST_ITEMS`）只是"截断"，用户事先并不知道
        自己选中的是一个 **13 万张**的根目录 —— 载入后在计数栏才会看到。
        这里在**载入之前**把规模摊开，让人自己决定要不要继续。
        默认按钮是「否」，避免手快误点把 2000 张灌进来。
        """
        if n <= MAX_LIST_ITEMS:
            return True
        btn = QMessageBox.question(
            self, "图像数量超过单次上限",
            f"选中的范围里共 {n} 张图像，已超过单次载入上限 {MAX_LIST_ITEMS} 张。\n\n"
            f"继续的话只会载入前 {MAX_LIST_ITEMS} 张（其余需分批处理）；\n"
            f"批量检测这 {MAX_LIST_ITEMS} 张在本机 CPU 上约需 3 分钟。\n\n"
            f"是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if btn != QMessageBox.Yes:
            self._requested = n
            self._hit_cap = True
            self.lbl_count.setText(
                f"已取消载入：选中 {n} 张，超过单次上限 {MAX_LIST_ITEMS} 张")
            self.lbl_count.setStyleSheet("color:#e67e22;")
            return False
        return True

    def _add_with_confirm(self, paths: List[str]) -> None:
        """**交互入口专用**：先做规模确认，再交给 `add_paths`。

        ⚠ 刻意不在 `add_paths` 里做确认 —— 脚本/验收会直接调 `add_paths`，
        模态框会把调用方（含自动化）卡在嵌套事件循环里。
        """
        if self._confirm_large(len(paths)):
            self.add_paths(paths)

    def add_paths(self, paths: List[str]) -> None:
        """把路径加入列表。**不再在 UI 线程里解码缩略图**（A1 修复）。

        改动点：
          * 遍历 + 建 item 留在 UI 线程（实测 0.8 s/13 万张，可接受）；
          * 缩略图交给 `ThumbWorker` 后台解码，逐张回填；
          * 总数封顶 `MAX_LIST_ITEMS`，截断时在计数栏明示。
        """
        exist = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        room = max(0, MAX_LIST_ITEMS - len(exist))
        accepted: List[str] = []
        hit_cap = False
        for p in paths:
            if len(accepted) >= room:
                hit_cap = len(paths) > len(accepted)   # 还有没看过的 ⇒ 确实是上限截的
                break
            if p in exist or not os.path.exists(p):
                continue
            accepted.append(p)
            exist.add(p)                      # 同批内的重复只收一次
        self._requested = len(paths)
        self._hit_cap = hit_cap

        first_row = self.list.count()
        for p in accepted:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            self.list.addItem(item)           # ★ 先不带图标，避免 UI 线程解码
        self._refresh_count()
        if accepted:
            self._start_thumbs([(first_row + i, p) for i, p in enumerate(accepted)])

    # ------------------------------------------------------------------
    def _start_thumbs(self, jobs) -> None:
        """起一个新的缩略图 worker；旧的那条只请求中断（不 wait，避免卡 UI）。"""
        self._thumb_gen += 1
        old = self._thumb_worker
        if old is not None and old.isRunning():
            old.requestInterruption()
        self._thumb_worker = ThumbWorker(self._thumb_gen, jobs, self)
        self._thumb_worker.one_ready.connect(self._on_thumb)
        self._thumb_worker.start()

    def _on_thumb(self, gen: int, row: int, qimg) -> None:
        """在主线程把 QImage 转成图标并回填（QPixmap 只能在这里建）。"""
        if gen != self._thumb_gen:
            return                            # 旧代次的信号，行号已失效
        item = self.list.item(row)
        if item is None:
            return
        item.setIcon(QIcon(QPixmap.fromImage(qimg)))

    @staticmethod
    def _thumb(path: str):
        """同步取缩略图（保留兼容；列表已改用 `ThumbWorker` 异步解码）。"""
        pm = QPixmap(path)
        if pm.isNull():
            return QIcon()
        return QIcon(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def clear(self) -> None:
        if self._thumb_worker is not None and self._thumb_worker.isRunning():
            self._thumb_worker.requestInterruption()
        self._thumb_gen += 1                  # 让在途信号全部失效
        self._requested = 0
        self._hit_cap = False
        self.list.clear()
        self._refresh_count()

    def shutdown(self) -> None:
        """关窗前必须调用：停掉缩略图线程并等它退出。

        ★ 为什么必须：**销毁一个仍在运行的 QThread 会让 Qt 直接 fail-fast**
        （本机实测退出码 `0xC0000409`，stderr 报
        `QThread: Destroyed while thread is still running`）。
        选完大文件夹立刻关窗口就会命中 —— 2000 张缩略图要后台跑约 76 秒。
        单张解码约 38 ms，所以请求中断后几十毫秒内就会退出，3 s 超时足够宽裕。
        本方法可重复调用。
        """
        w = self._thumb_worker
        self._thumb_gen += 1                  # 先让在途信号失效，避免回填到已清空的列表
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.wait(3000)
        self._thumb_worker = None

    def _refresh_count(self) -> None:
        n = self.list.count()
        txt = f"共 {n} 张待检测图像"
        if getattr(self, "_hit_cap", False):
            txt = (f"共 {n} 张待检测图像（已达单次上限 {MAX_LIST_ITEMS}；"
                   f"本次选中 {getattr(self, '_requested', n)} 张，仅载入前 {n} 张 —— "
                   f"可分批检测）")
            self.lbl_count.setStyleSheet("color:#e67e22;")
        else:
            self.lbl_count.setStyleSheet("color:#666;")
        self.lbl_count.setText(txt)
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
