"""桌面工具 · 系统设置模块

可配置项（申报书 3.2）：推理后端、推理设备、检测阈值、结果保存路径、模型路径。
所有配置通过 QSettings 持久化到注册表/ini，下次启动自动恢复。
"""

from __future__ import annotations

import os

from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class SettingsPanel(QGroupBox):
    """⑤ 系统设置模块。"""

    applied = pyqtSignal(dict)

    DEFAULTS = {
        "backend": "torch",
        "device": "cpu",
        "threshold": 0.5,
        "save_dir": "./outputs",
        "ckpt": "checkpoints/vibnet_best.pt",
        "onnx": "deploy/vibnet_simplified.onnx",
        "openvino": "deploy/vibnet_ir/vibnet.xml",
        "config": "configs/default.yaml",
        "image_size": 224,
        "auto_load": True,
    }

    def __init__(self, parent=None, project_root: str = "."):
        super().__init__("⑤ 系统设置", parent)
        self.root = project_root
        self.settings = QSettings("GXU", "VIBNetDetector")
        self._build()
        self.load_config()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.cmb_backend = QComboBox()
        try:
            from src.deploy.inference import list_available_backends

            self.cmb_backend.addItems(list_available_backends())
        except Exception:
            self.cmb_backend.addItems(["torch", "onnx", "openvino"])
        self.cmb_backend.setToolTip("torch=调试用；onnx=通用；openvino=Intel CPU 加速（最快）")
        form.addRow("推理后端", self.cmb_backend)

        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["cpu", "gpu"])
        form.addRow("推理设备", self.cmb_device)

        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.05, 0.95)
        self.spin_thr.setSingleStep(0.05)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setToolTip("伪造概率大于该阈值即判定为疑似伪造，默认 0.50")
        form.addRow("检测阈值", self.spin_thr)

        self.spin_size = QSpinBox()
        self.spin_size.setRange(128, 512)
        self.spin_size.setSingleStep(32)
        form.addRow("输入尺寸", self.spin_size)

        self.edit_save = QLineEdit()
        btn_save = QPushButton("…")
        btn_save.setFixedWidth(30)
        row = QHBoxLayout()
        row.addWidget(self.edit_save)
        row.addWidget(btn_save)
        w = QWidget()
        w.setLayout(row)
        form.addRow("结果保存路径", w)
        btn_save.clicked.connect(lambda: self._pick_dir(self.edit_save))

        self.edit_ckpt = QLineEdit()
        b1 = QPushButton("…")
        b1.setFixedWidth(30)
        r1 = QHBoxLayout(); r1.addWidget(self.edit_ckpt); r1.addWidget(b1)
        w1 = QWidget(); w1.setLayout(r1)
        form.addRow("PyTorch 权重", w1)
        b1.clicked.connect(lambda: self._pick_file(self.edit_ckpt, "*.pt"))

        self.edit_onnx = QLineEdit()
        b2 = QPushButton("…")
        b2.setFixedWidth(30)
        r2 = QHBoxLayout(); r2.addWidget(self.edit_onnx); r2.addWidget(b2)
        w2 = QWidget(); w2.setLayout(r2)
        form.addRow("ONNX 模型", w2)
        b2.clicked.connect(lambda: self._pick_file(self.edit_onnx, "*.onnx"))

        self.edit_ov = QLineEdit()
        b3 = QPushButton("…")
        b3.setFixedWidth(30)
        r3 = QHBoxLayout(); r3.addWidget(self.edit_ov); r3.addWidget(b3)
        w3 = QWidget(); w3.setLayout(r3)
        form.addRow("OpenVINO IR", w3)
        b3.clicked.connect(lambda: self._pick_file(self.edit_ov, "*.xml"))

        root.addLayout(form)

        btns = QHBoxLayout()
        self.btn_apply = QPushButton("保存并加载模型")
        self.btn_reset = QPushButton("恢复默认")
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        root.addLayout(btns)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#666;")
        root.addWidget(self.lbl_status)

        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self.reset)

    # ------------------------------------------------------------------
    def _pick_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择目录", edit.text() or self.root)
        if d:
            edit.setText(d)

    def _pick_file(self, edit: QLineEdit, pattern: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择文件", self.root, f"文件 ({pattern})")
        if p:
            edit.setText(p)

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        return {
            "backend": self.cmb_backend.currentText(),
            "device": self.cmb_device.currentText(),
            "threshold": float(self.spin_thr.value()),
            "save_dir": self.edit_save.text().strip() or "./outputs",
            "ckpt": self.edit_ckpt.text().strip(),
            "onnx_path": self.edit_onnx.text().strip(),
            "openvino_ir": self.edit_ov.text().strip(),
            "image_size": int(self.spin_size.value()),
            "config_path": self._abs(self.DEFAULTS["config"]),
        }

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.root, p))

    def _apply(self) -> None:
        cfg = self.get_config()
        for k, v in cfg.items():
            self.settings.setValue(k, v)
        self.applied.emit(cfg)

    def reset(self) -> None:
        for k, v in self.DEFAULTS.items():
            self.settings.setValue(k, v)
        self.load_config()
        self.lbl_status.setText("已恢复默认设置")

    def load_config(self) -> None:
        for k, dv in self.DEFAULTS.items():
            v = self.settings.value(k, dv)
            if k == "backend":
                i = self.cmb_backend.findText(str(v))
                self.cmb_backend.setCurrentIndex(i if i >= 0 else 0)
            elif k == "device":
                i = self.cmb_device.findText(str(v))
                self.cmb_device.setCurrentIndex(i if i >= 0 else 0)
            elif k == "threshold":
                self.spin_thr.setValue(float(v))
            elif k == "image_size":
                self.spin_size.setValue(int(v))
            elif k == "save_dir":
                self.edit_save.setText(str(v))
            elif k == "ckpt":
                self.edit_ckpt.setText(str(v))
            elif k == "onnx":
                self.edit_onnx.setText(str(v))
            elif k == "openvino":
                self.edit_ov.setText(str(v))

    def set_status(self, text: str, color: str = "#666") -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color:{color};")
