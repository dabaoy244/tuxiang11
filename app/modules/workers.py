"""桌面工具 · 检测处理模块 —— 异步推理工作线程

设计要点（申报书 3.2 "检测处理模块：加载优化后的模型，实现异步推理，避免界面卡顿"）：
  * 模型加载与推理都放在 QThread 中，主线程（UI）只负责信号槽通信；
  * 批量检测支持中途取消（`requestInterruption`）；
  * 逐张 emit 结果，界面可以边检测边刷新，长任务也有实时反馈。
"""

from __future__ import annotations

import os
import traceback
from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal


class DetectorLoader(QThread):
    """后台加载模型（首次加载 CLIP 权重较慢，不能阻塞 UI）。"""

    loaded = pyqtSignal(object)      # Detector
    failed = pyqtSignal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            from src.deploy.inference import Detector

            det = Detector(**self.kwargs)
            self.loaded.emit(det)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchDetectWorker(QThread):
    """批量检测工作线程。"""

    progress = pyqtSignal(int, int, str)      # 当前索引, 总数, 文件名
    one_done = pyqtSignal(object)             # DetectionResult
    finished_all = pyqtSignal(list, list)     # (成功列表, 失败列表[(path, err)])
    failed = pyqtSignal(str)

    def __init__(self, detector, paths: List[str], parent=None):
        super().__init__(parent)
        self.detector = detector
        self.paths = list(paths)

    def run(self) -> None:
        if self.detector is None:
            self.failed.emit("模型尚未加载完成")
            return
        ok, bad = [], []
        total = len(self.paths)
        for i, p in enumerate(self.paths):
            if self.isInterruptionRequested():
                break
            self.progress.emit(i + 1, total, os.path.basename(p))
            try:
                res = self.detector.detect(p)
                ok.append(res)
                self.one_done.emit(res)
            except Exception as e:  # noqa: BLE001
                bad.append((p, f"{type(e).__name__}: {e}"))
        self.finished_all.emit(ok, bad)
