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
        # ★ backend=onnx：本机实测 onnx 加载 0.3s、单张 ~90ms，而 torch 后端要 27s。
        #   torch / openvino 仍可在下拉框里选。
        "backend": "onnx",
        "device": "cpu",
        "threshold": 0.5,
        "save_dir": "./outputs",
        # ★★★ 2026-10-09（D 组 ③）：默认权重换成审计认定的交付臂 g43。
        #   依据**只有一条**：阈无关主口径 AUC 显著更高 —— ForenSynths 未见生成器
        #   （strict-9）宏平均 **AUC 0.8671 vs 出厂 0928 的 0.8434 ⇒ +2.37pp★**
        #   （同战役配对 bootstrap，CI [+0.0107, +0.0365] 不含 0；full-13 口径 +1.78pp★）。
        #   ⚠ 换权重前必查两件事（本项目踩过）：
        #     ① 不要认文件名，认 md5 —— checkpoint 内 `best.pt` 会被后续 stage 覆盖写；
        #     ② 换完必须重做一次「ONNX vs .pt 同抽样一致性」检查（本轮已做，见下）。
        "ckpt": "outputs/upload/weights_improved_20261005/improved_noaug_best.pt",
        # ★★ 2026-10-09（③）：ONNX 默认值同步换成 g43 的导出版。
        #   为什么不能再用 lite 三件套：`deploy/lite.onnx` / `lite_simplified.onnx` /
        #   `lite_int8.onnx` 是 `export_onnx.py --ckpt ""`（**空 checkpoint**）导出的
        #   **随机初始化模型**，当初只为测「体积 / 速度」两项指标（docs/09 §导出命令、
        #   docs/08 §38），从未训练过。实测：三件套对任何正常图像都恒定输出
        #   p_fake=0.4852，对全黑/全白输入出 NaN ⇒ 这类文件**存在但无效**，
        #   查"文件是否存在"拦不住，只能靠 `_KNOWN_DEAD` + `self_check()` 兜。
        #   导出来源：`src/deploy/export_onnx.py --ckpt <上面的 g43 .pt>
        #   --out deploy/b16_g43.onnx --opset 17 --config configs/improved_noaug.yaml`
        #   （纯 CPU trace，¥0；363.4 MB）。
        #   一致性：与 .pt 同抽样对比，mask_logits max|Δ| 5.5e-4（float32 舍入量级），
        #   prob_fake max|Δ| 5.3e-5 ⇒ 导出件确实是同一套权重。
        "onnx": "deploy/b16_g43.onnx",
        "openvino": "deploy/vibnet_ir/vibnet.xml",
        "config": "configs/default.yaml",
        "image_size": 224,
        "auto_load": True,
    }

    #: 这几项是**文件路径**，加载配置时要校验存在性（见 `load_config`）。
    _PATH_KEYS = ("ckpt", "onnx", "openvino")

    # ==================================================================
    # ★★ 模型卡（2026-10-09 新增）
    #
    # 为什么要有它：此前 `app/` 全目录**不含任何实测指标数字** —— 软件只会给
    # "这张图怎么样"，说不出"这个模型多少分"。软著/答辩时被问"精度多少"就答不上来。
    #
    # 为什么放在「关于」而不是结果面板/PDF 报告：
    #   结果面板与 PDF 报的是**单张推断结果**。若把测试集指标混进去，读者会把它
    #   误读成"本次检测的准确率"。性能指标属于"模型卡"，是**这个构建**的属性，
    #   不是**这一次检测**的属性。
    #
    # ★ 三条纪律（本项目已为此栽过三次）：
    #   ① 每个数都带 (权重 md5, 抽样口径, 聚合方式) —— 见 `MODEL_CARD_PROTOCOL`；
    #   ② **不跨权重误归属** —— 例如"CASIAv2 定位 mIoU 0.2668"是 **0928 出厂件**的数，
    #      换成 g43 之后**不得**再写进本卡；本卡只列 g43 自己测出来的数。
    #   ③ 数字**只作参考、不作承诺** —— 它们是特定测试集上的指标，不是现场精度保证。
    # ==================================================================
    MODEL_CARD_WEIGHT = "improved_noaug_best.pt（g43）"
    MODEL_CARD_WEIGHT_MD5 = "e1e5a7e9d8b9832523ccaa98a1c7b757"
    MODEL_CARD_ONNX = "deploy/b16_g43.onnx"
    #: 这条数来自 `outputs/_audit_20261008/step6_shipped_0928.md:13` 的同战役四行表
    #: （与出厂 0928、主表 full、aug g50 一起测的同一套抽样 ⇒ 行内可比）。
    MODEL_CARD_METRICS = [
        ("ForenSynths 未见生成器（strict-9）宏平均 AUC", "0.8671", "主口径（阈无关）"),
        ("　同上 AP", "0.8540", "必报伴随"),
        ("　同上 ACC@EER", "0.7467", "必报伴随（= 1 − EER，阈无关）"),
        ("　同上 ACC@0.5", "0.7115", "仅当前操作点参考（阈值相关）"),
        ("ForenSynths 全量 13 生成器 宏平均 AUC", "0.8954", "含与训练同源的 progan，偏高"),
    ]
    MODEL_CARD_PROTOCOL = (
        "抽样：`--per-class 150`（每个生成器 150 真 + 150 假，真图去重，n=3900）；"
        "聚合：宏平均（先逐生成器算再平均）。"
        "strict-9 = 13 个生成器里排除 `progan`（与训练同源）与 "
        "`biggan/cyclegan/stargan`（留作 val_cross）之后的 9 个（n=2700）。"
    )

    def _md5_of(self, path: str) -> str:
        import hashlib
        h = hashlib.md5()
        p = self._abs(path)
        if not os.path.exists(p):
            return "(文件不存在)"
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def model_card_html(self, weight_md5: str = None) -> str:
        """模型卡正文。`weight_md5` 传 None 时显示占位（由对话框后台补齐）。"""
        rows = "".join(
            "<tr><td style='padding:2px 10px 2px 0;'>{}</td>"
            "<td style='padding:2px 14px 2px 0;'><b>{}</b></td>"
            "<td style='color:#777;'>{}</td></tr>".format(*m) for m in self.MODEL_CARD_METRICS)
        if weight_md5 is None:
            md5_cell = "<span style='color:#999;'>计算中…</span>"
        elif weight_md5.lower() == self.MODEL_CARD_WEIGHT_MD5:
            md5_cell = "{}　<span style='color:#1e8e3e;'>✅ 与预期一致</span>".format(weight_md5)
        else:
            md5_cell = "{}　<span style='color:#c0392b;'>⚠ 与预期不一致（"
            md5_cell = md5_cell.format(weight_md5) + self.MODEL_CARD_WEIGHT_MD5 + "）</span>"
        return (
            "<h3 style='margin:2px 0 6px;'>空频双分支轻量 VIB-Net · AI 图像篡改检测系统 V1.0</h3>"
            "<p style='margin:4px 0;'>"
            "部署权重：<b>{w}</b><br>"
            "当前 ONNX：<b>{o}</b><br>"
            "权重 md5：{m}</p>"
            "<hr>"
            "<p style='margin:4px 0 2px;'><b>性能（ForenSynths）</b></p>"
            "<table style='font-size:12px;'>{r}</table>"
            "<p style='margin:6px 0 0;color:#666;font-size:11px;'>{p}</p>"
            "<hr>"
            "<p style='margin:4px 0;color:#666;font-size:11px;'>"
            "• 以上数字是<b>特定测试集</b>上的指标，<b>不是</b>本工具对任意图片的精度承诺，"
            "也<b>不是</b>司法鉴定依据。<br>"
            "• 篡改定位指标（CASIAv2 / COVERAGE，`miou_tampered_only` 口径）按臂单独报告，"
            "见项目材料 —— 本卡只列<b>本权重</b>的分类口径，避免把别的权重的数算到它头上。<br>"
            "• 结论由模型自动给出，仅供辅助参考。</p>"
        ).format(w=self.MODEL_CARD_WEIGHT, o=self.MODEL_CARD_ONNX,
                 m=md5_cell, r=rows, p=self.MODEL_CARD_PROTOCOL)

    def _show_model_card(self) -> None:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QTextBrowser, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("关于 · 模型卡")
        dlg.resize(660, 560)
        lay = QVBoxLayout(dlg)
        br = QTextBrowser()
        br.setOpenExternalLinks(False)
        br.setHtml(self.model_card_html(None))
        lay.addWidget(br)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        # md5 要读 345 MB（约 1 秒）。先让对话框画出来，再算 —— 否则点"关于"
        # 会有约 1 秒无反馈，看起来像卡死。
        QTimer.singleShot(80, lambda: br.setHtml(self.model_card_html(self._md5_of(self.DEFAULTS["ckpt"]))))
        dlg.exec_()

    #: ★ 2026-10-09（A3）：`QSettings` 持久化键名映射。

    #: ★ 2026-10-09（A3）：`QSettings` 持久化键名映射。
    #:
    #: 病根：`get_config()` 返回的键名是**给 `Detector(**kwargs)` 用的**
    #: （`onnx_path` / `openvino_ir` / `config_path`，见 `src/deploy/inference.py:148`
    #: 的形参名），而 `DEFAULTS` 与 `load_config()` 用的是 `onnx` / `openvino` /
    #: `config`。`_apply()` 此前**把 `get_config()` 的键名原样写进注册表** ⇒
    #: 那 3 个键**写了从不读**，症状 = "保存成功、重启丢失"。
    #:
    #: ⚠ **不能改 `get_config()` 的键名** —— `app/main.py:185-187` 直接读
    #: `cfg["onnx_path"]` / `["openvino_ir"]` / `["config_path"]` 去构造 Detector，
    #: 而 Detector 的形参本来就叫 `onnx_path`。所以输出名是对的、错的是持久化层。
    #: ⇒ 只在**写边界**做映射，接口名保持不动（`main.py` / `inference.py` 零改动）。
    #:
    #: 注：`reset()` 走的 `DEFAULTS` 键名本来就是规范名，无需映射。
    _PERSIST_ALIASES = {
        "onnx_path": "onnx",
        "openvino_ir": "openvino",
        "config_path": "config",
    }

    #: ★ 本仓库内**已知无效**的导出件：文件存在、能加载、推理不报错，
    #: 但对任何图像都恒定输出（随机初始化 / 训练前导出）。仅查存在性拦不住，
    #: 所以显式拉黑，命中就回退到 DEFAULTS。
    #:   证据见 docs/09「导出命令」——lite 三件套是 `--ckpt ""` 导出的。
    _KNOWN_DEAD = {
        "deploy/lite.onnx": "随机初始化（--ckpt \"\" 导出），恒输出 0.4852",
        "deploy/lite_simplified.onnx": "随机初始化（--ckpt \"\" 导出），恒输出 0.4852",
        "deploy/lite_int8.onnx": "随机初始化（--ckpt \"\" 导出），恒输出 0.4852",
        "deploy/b16.onnx": "训练前导出，恒输出 0.4982",
        "deploy/b16_int8.onnx": "训练前导出，恒输出 0.4982",
        "checkpoints/vibnet_best.pt": "09-26 旧权重，p_fake 恒在 0.54 附近（须用 _pack 那份）",
    }

    def __init__(self, parent=None, project_root: str = "."):
        super().__init__("⑤ 系统设置", parent)
        self.root = project_root
        self.settings = QSettings("GXU", "VIBNetDetector")
        self._auto_fixed: list = []       # 被自动纠正的过期项（load_config 里填）
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
        # ★ 2026-10-09（A4）：上限 0.95 → 1.00。实测 EER 最优点在 0.9988~0.9999，
        #   旧上限连 0.99 都够不到，留一个"设不到最优点"的滑块没有意义。
        #   默认值**仍为 0.50** —— 不把任何标定值提升为默认：本项目没有合法留出集
        #   （val_cross 与域内 val 都不跟踪测试性能），标定值是测试集产物。
        #   ⚠ decimals 保持 2 位：这样 0.9988 仍不可达、只到 0.99，而操作点拐点在
        #   0.95~0.99 ⇒ 0.99 够用。两位小数恰好挡住"照着测试集把阈值拧到 0.9988"。
        self.spin_thr.setRange(0.05, 1.00)
        #   ⚠ 步长必须 0.01（原 0.05 是错的）：实测（offscreen 复现，2026-10-09）
        #   step=0.05 时箭头/滚轮可达集合只有 0.05 的倍数共 20 个值
        #   （…0.90/0.95/**1.00**），从 0.95 往上**直接跳过 0.99** ⇒ 上文"只到 0.99"
        #   注释形同虚设，0.99 只能靠手输。改 0.01 后 0.99 可正常步进选中，
        #   而 0.9988 因 decimals=2 仍被吞成 1.00 ⇒ 挡 oracle 调参的原意保留。
        self.spin_thr.setSingleStep(0.01)
        self.spin_thr.setDecimals(2)
        # 判定实现是 `prob >= threshold`（src/deploy/inference.py:373），故写"大于等于"。
        #   旧上限 0.95 下这个措辞差异无后果；上限到 1.00 后才有后果：阈值 1.00 仍会命中
        #   输出**恰好 1.0000** 的图像（本模型分数饱和），而不是"全部判真"。
        self.spin_thr.setToolTip("伪造概率大于等于该阈值即判定为疑似伪造，默认 0.50")
        form.addRow("检测阈值", self.spin_thr)

        self.spin_size = QSpinBox()
        # ★ 2026-10-09（A2）：锁成 224（原 128~512 可调）。
        #   原因：ONNX 计算图的 H/W 是**静态 224**（默认件 `deploy/b16_g43.onnx`），
        #   一改尺寸每张图都会失败且只报"加载失败"。真支持多尺寸需重导 dynamic-H/W 的
        #   ONNX 并改 Detector 预处理，属独立改动，不在这里做。
        self.spin_size.setRange(224, 224)
        self.spin_size.setSingleStep(32)
        self.spin_size.setToolTip("ONNX 计算图固定 224×224；改尺寸需重新导出模型")
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
        self.btn_about = QPushButton("关于 · 模型卡")
        self.btn_about.setToolTip("查看本构建的模型版本、权重 md5 与实测指标（含口径）")
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        btns.addWidget(self.btn_about)
        root.addLayout(btns)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#666;")
        root.addWidget(self.lbl_status)

        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_about.clicked.connect(self._show_model_card)

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
        # ★ A3：按 `_PERSIST_ALIASES` 映射后再写 —— 与 `load_config()` 读的键名对齐，
        #   否则 `onnx_path` / `openvino_ir` / `config_path` 写进去后永远读不回来。
        for k, v in cfg.items():
            self.settings.setValue(self._PERSIST_ALIASES.get(k, k), v)
        self.applied.emit(cfg)

    def reset(self) -> None:
        for k, v in self.DEFAULTS.items():
            self.settings.setValue(k, v)
        self.load_config()
        self.lbl_status.setText("已恢复默认设置")

    def load_config(self) -> None:
        # ---- ① 先解析各路径项：注册表里的值若指向不存在的文件，回退到默认值 ----
        # 为什么要这一步：QSettings 会把上次的值持久化到注册表（HKCU\Software\GXU\
        # VIBNetDetector），**只改代码里的 DEFAULTS 盖不过它** —— 上一版写进去的
        # 错误路径会一直生效，表现成「明明改好了还是报同样的错」。
        self._auto_fixed = []
        vals = {}
        for k, dv in self.DEFAULTS.items():
            v = self.settings.value(k, dv)
            if k in self._PATH_KEYS:
                v = self._pick_valid(v, dv)
            vals[k] = v

        # ---- ② 后端自愈：存的后端若依赖文件不存在，自动退到可用的那个 ----
        vals["backend"] = self._resolve_backend(str(vals.get("backend", "")), vals)

        # ---- ③ 回填控件 ----
        for k, v in vals.items():
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

        if self._auto_fixed:
            self.set_status(f"已自动纠正 {len(self._auto_fixed)} 项过期路径："
                            + "、".join(self._auto_fixed), "#e67e22")

    # ------------------------------------------------------------------
    def _pick_valid(self, value, default) -> str:
        """路径项：值不可用时回退默认值，并记进 `_auto_fixed`。

        「不可用」有两种，**必须都拦**：
          ① 文件不存在            —— 直接报 FileNotFoundError，容易发现；
          ② 文件存在但已知无效    —— 本仓库的 lite*.onnx / b16.onnx / 09-26 旧权重
             属于这一类，**加载不报错、结果全错**（静默失效），只查存在性会漏掉。
        """
        v = str(value or "")
        key = v.replace("\\", "/").lstrip("./")
        if v and key in self._KNOWN_DEAD:
            self._auto_fixed.append(f"{v}（{self._KNOWN_DEAD[key]}）→ {default}")
            return str(default)
        if v and os.path.exists(self._abs(v)):
            return v
        if v and v != str(default):
            self._auto_fixed.append(f"{v} → {default}")
        return str(default)

    def _resolve_backend(self, want: str, vals: dict) -> str:
        """挑一个**依赖文件确实存在**的后端；存的后端不可用时按优先级退让。

        这里刻意不静默：退让会记进 `_auto_fixed`，状态栏与主窗口日志都会显示，
        避免出现「下拉框显示 torch、实际用 onnx」这种对不上账的情况。
        """
        cands = [self.cmb_backend.itemText(i) for i in range(self.cmb_backend.count())]
        need = {
            "onnx": vals.get("onnx"),
            "openvino": vals.get("openvino"),
        }

        def usable(b: str) -> bool:
            p = need.get(b)
            if p is None:
                return True                      # torch：缺权重也能跑（随机初始化 + 警告）
            return os.path.exists(self._abs(str(p)))

        if want in cands and usable(want):
            return want
        for b in cands:
            if usable(b):
                self._auto_fixed.append(f"后端 {want} → {b}（依赖文件缺失）")
                return b
        return cands[0] if cands else "torch"

    def set_status(self, text: str, color: str = "#666") -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color:{color};")
