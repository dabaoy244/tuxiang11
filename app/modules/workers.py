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
            # ★ 加载完立刻自检：本仓库有过"lite*.onnx 是随机初始化模型、对任何图都
            #   恒定输出 0.4852"的静默失效，只查文件存在性拦不住，必须喂两组不同
            #   输入比对输出。结论挂到 det 上，由主窗口读出来显示给用户。
            try:
                det.self_check_notes = det.self_check()
            except Exception as e:  # noqa: BLE001
                det.self_check_notes = [f"自检异常：{type(e).__name__}: {e}"]
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


class ThumbWorker(QThread):
    """后台解码列表缩略图。

    为什么需要（A1 缺陷）
    --------------------
    `input_panel.add_paths()` 原先在 **UI 线程**里逐张 `QPixmap(path)`：那是
    **解码整图**再缩到 64×64，实测约 **38 ms/张**。选 `data/Datasets`（132,817 张）
    ⇒ 界面冻结 **约 84 分钟**。（目录遍历本身只 0.8 s，不是瓶颈。）
    现在遍历/建 item 仍在 UI 线程（0.8 s 可接受），解码挪到这里。

    ★ Qt 硬约束
    -----------
    `QPixmap` **只能在 GUI 线程创建**。所以本线程只产 **`QImage`**，
    主线程收到后自己做 `QPixmap.fromImage()` → `QIcon`。
    （`QImage` 在非 GUI 线程构造允许，但必须 `.copy()` 拥有数据 ——
     否则它只引用 numpy 缓冲，缓冲释放后就是野指针。）

    ★ 代次号（gen）
    -------------
    每次 `add_paths` 都会起一个新 worker，旧的可能还在跑，它的 `one_ready` 带着
    **上一批的行号** —— 照行号 setIcon 会写错行。所以信号里带 `gen`，
    主线程只接受与当前代次一致的信号。
    """

    one_ready = pyqtSignal(int, int, object)   # gen, row, QImage

    def __init__(self, gen: int, jobs, parent=None):
        super().__init__(parent)
        self.gen = gen
        self.jobs = list(jobs)                 # [(row, path), ...]

    def run(self) -> None:
        try:
            import cv2
            import numpy as np
            from PyQt5.QtGui import QImage
        except Exception:  # noqa: BLE001
            return
        for row, path in self.jobs:
            if self.isInterruptionRequested():
                return
            try:
                # 兼容中文路径：cv2.imread 在 Windows 上读不到非 ASCII 路径
                buf = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                s = 64.0 / max(h, w, 1)
                if s < 1.0:
                    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                                     interpolation=cv2.INTER_AREA)
                rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3,
                              QImage.Format_RGB888).copy()      # ★ copy：拥有数据
                self.one_ready.emit(self.gen, row, qimg)
            except Exception:  # noqa: BLE001
                continue        # 单张坏图不该拖垮整批
