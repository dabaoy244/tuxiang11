"""统一推理封装 —— 供桌面工具 / 命令行评测 / 批量测试共用。

支持三种后端：
    torch     : 直接跑 PyTorch（开发调试用，CPU 上较慢）
    onnx      : ONNXRuntime（跨平台，装机最简单）
    openvino  : Intel CPU 指令集优化（申报书指定的端侧加速方案，最快）

桌面工具通过 `Detector(backend=...)` 切换，无需改动业务代码。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

# 热力图配色（BGR，OpenCV 顺序）
HEATMAP = None  # 运行时惰性导入 cv2


def imread_unicode(path: str, flags: int = 1):
    """读图，兼容 Windows 中文 / 非 ASCII 路径。

    ★ 为什么先判 ASCII：`cv2.imread` 在 Windows 上碰到非 ASCII 路径时会
    **返回 None，并往 stderr 打一行**
    `[WARN] loadsave.cpp: findDecoder imread_('...'): can't open/read file`。
    我们随后确实会回退到 `np.fromfile + imdecode`（结果是对的），但那行警告会把
    用户日志弄脏——桌面工具拿它读中文路径时**每次都会刷**（实测 e2e 里一次三行）。
    所以：ASCII 路径直接走 imread（最快），非 ASCII 直接 imdecode（零警告）。

    读失败 / 路径不存在一律返回 None，**不抛异常**，由调用方按各自契约处理。
    """
    import cv2

    ascii_safe = True
    try:
        path.encode("ascii")
    except (UnicodeEncodeError, AttributeError):
        ascii_safe = False

    if ascii_safe:
        img = cv2.imread(path, flags)
        if img is not None:
            return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf is None or getattr(buf, "size", 0) == 0:
        return None
    return cv2.imdecode(buf, flags)


@dataclass
class DetectionResult:
    path: str
    name: str
    width: int
    height: int
    is_fake: bool
    prob_fake: float           # 伪造置信度 [0,1]
    mask: np.ndarray           # (H,W) float32 [0,1] 篡改概率
    overlay: np.ndarray        # (H,W,3) uint8 BGR 热力图叠加
    time_ms: float
    backend: str
    threshold: float
    warnings: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "疑似伪造" if self.is_fake else "判定为真实"

    def summary(self) -> Dict[str, object]:
        return {
            "文件名": self.name,
            "图像尺寸": f"{self.width}x{self.height}",
            "检测结论": self.label,
            "伪造置信度": round(self.prob_fake, 4),
            "判定阈值": self.threshold,
            "高置信度区域占比": round(float((self.mask > 0.5).mean()), 4),
            "推理耗时(ms)": round(self.time_ms, 2),
            "推理后端": self.backend,
        }


# ==========================================================================
def list_available_backends() -> List[str]:
    """探测本机可用后端（桌面工具用它来填充下拉框）。"""
    out = ["torch"]
    try:
        import onnxruntime  # noqa: F401
        out.append("onnx")
    except ImportError:
        pass
    try:
        import openvino  # noqa: F401
        out.append("openvino")
    except ImportError:
        pass
    return out


#: ONNX Runtime 的 intra-op 线程数。
#:
#: ⚠️ 不要留空让 ORT 自己决定（它默认用满所有逻辑核）。本机实测
#: （7 逻辑核，`deploy/lite_simplified.onnx`，224×224，batch=1，
#: 5 轮 × 20 次取中位数，2026-09-29）：
#:
#:     ORT 默认（全核）  147.6 ms  →  6.77 张/秒   轮间波动 46%   ❌ 不达 8 张/秒
#:     threads=2         117.4 ms  →  8.52 张/秒    2%
#:     threads=3          92.9 ms  → 10.77 张/秒    4%
#:     threads=4          85.3 ms  → 11.73 张/秒   10%   ← 最优（复测 10.98）
#:     threads=5          90.7 ms  → 11.03 张/秒   21%
#:     threads=6          99.9 ms  → 10.01 张/秒   72%
#:
#: 即「把所有核心都给 ORT」反而慢 42%，而且结果极不稳定 —— 线程超订后各线程
#: 互相抢占、空转（spin-wait）。桌面工具与验收演示都走这条路径，所以这个值
#: 直接决定现场能不能达标。实测过 4 是拐点；部署机核数差别大时用环境变量
#: `VIBNET_ORT_THREADS` 覆盖，或调用 `benchmark_cpu.py --threads N` 现场标定。
_DEFAULT_ORT_THREADS = 4


def ort_num_threads() -> int:
    """取当前生效的 ORT intra-op 线程数（环境变量优先）。"""
    env = os.environ.get("VIBNET_ORT_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return min(_DEFAULT_ORT_THREADS, os.cpu_count() or _DEFAULT_ORT_THREADS)


# ==========================================================================
class Detector:
    def __init__(
        self,
        backend: str = "torch",
        image_size: int = 224,
        threshold: float = 0.5,
        # ★★★ 2026-10-11：默认权重 = `geofix_s3_best.pt`
        #   （md5 `184fb15c1787cf5470397adbe8a1ac93`）—— 几何修复 + stage3 联合微调。
        #   它是**两个任务都优于 g43** 的第一个交付件：分类持平（strict-9 宏 AUC
        #   0.8632 vs g43 0.8665，−0.33pp）、定位 mIoU★ **+13.79pp**
        #   （0.3482 vs g43 在同命令同分母 n=474 下重测的 0.2103）。
        #   前两任：0928 出厂件 → g43(improved_noaug_best) → geofix_s2_epoch10。
        #   ⚠ 不要用同名的 `checkpoints/vibnet_best.pt`（09-26 旧权重，
        #   p_fake 恒在 0.54 附近、对真图/篡改图无区分）—— 认 md5，不认文件名。
        ckpt: str = "outputs/upload/weights_geofix_s3_20261011/geofix_s3_best.pt",
        # ★★★ 与 `app/modules/settings_panel.py` 的 `DEFAULTS["onnx"]` **必须一致**
        #   —— 两处都改才算"换了默认模型"。本件由上面那份 .pt 导出：
        #   `python -m src.deploy.export_onnx --ckpt <上面的 .pt>
        #    --out deploy/b16_s3best.onnx --opset 17
        #    --config configs/geofix_s3_only.yaml --simplify`（纯 CPU trace，¥0，363.4 MB）。
        #   一致性已验（logit 级）：cls_logits max|Δ| 2.10e-05、
        #   mask_logits max|Δ| 3.89e-04；且已核实 --simplify 真跑过（Identity 72→0）。
        #   ⚠ 不要用 deploy/lite*.onnx —— 那是 `--ckpt ""` 导出的随机初始化件。
        onnx_path: str = "deploy/b16_s3best.onnx",
        # ★★★ 2026-10-11：与上面的 onnx_path 同步换成同一权重的 IR。
        #   若这里不改，OpenVINO 后端会**静默跑另一个权重**。
        #   生成：`openvino.convert_model('deploy/b16_s3best.onnx')` +
        #     `save_model(..., compress_to_fp16=False)` → `deploy/vibnet_ir_s3best/`
        #     （本地 CPU，1.2 s，¥0；已回读验证：输入 [?,3,224,224]、3 个输出）。
        #   旧件（0928 的 `vibnet_ir/`、g43 的 `vibnet_ir_g43/`、e10 的）都保留留档。
        openvino_ir: str = "deploy/vibnet_ir_s3best/b16_s3best.xml",
        config_path: str = "configs/default.yaml",
        device: str = "cpu",
        num_threads: int = 0,
    ):
        self.backend = backend
        self.image_size = image_size
        self.threshold = threshold
        self.device = device
        self.num_threads = num_threads          # 0 = 自动（见 ort_num_threads）
        self._torch_model = None
        self._ort = None
        self._ov = None
        self._fallback_note: Optional[str] = None

        if backend == "onnx":
            self._init_onnx(onnx_path)
        elif backend == "openvino":
            self._init_openvino(openvino_ir)
        else:
            self._init_torch(ckpt, config_path)

    # ------------------------------------------------------------------
    def _init_torch(self, ckpt: str, config_path: str) -> None:
        import torch

        from ..models.spatial_branch import align_cfg_to_checkpoint
        from ..models.vibnet import build_model, load_config

        cfg = load_config(config_path) if os.path.exists(config_path) else None
        if cfg is None:
            raise FileNotFoundError(f"找不到配置 {config_path}")
        cfg["data"]["image_size"] = self.image_size
        state = None
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            # 按 checkpoint 反推骨干档位并对齐，避免旧权重配新配置时报 size mismatch
            align_cfg_to_checkpoint(cfg, state.get("model", state))
        model = build_model(cfg)
        if state is not None:
            model.load_state_dict(state.get("model", state), strict=False)
        else:
            self._fallback_note = f"未找到权重 {ckpt}，当前使用随机初始化模型（仅用于界面联调）"
        model.eval()
        # ★ 2026-10-09（③）：把 DFT 实现切到 matmul，与导出的 ONNX 件保持一致。
        #   病根：`freq_branch.py:117` 的 `DFT2D(..., mode=f.get("dft_mode","torch"))`
        #   默认走 `torch.fft`，而 `export_onnx.py:79` 在导出前会 `set_dft_mode("matmul")`
        #   （ONNX 不支持 aten::fft_fft2）。两种实现数学等价但**数值有差**，
        #   实测在本机同一批样本上：mask_logits max|Δ| 达 1.6e-2（换 g43 后）、
        #   出厂件 0928 也有 4.1e-3 ⇒ 是**后端间既有差异**，不是某次导出走样。
        #   切到 matmul 后 torch 通路与 ONNX 件走同一个算子，残差降到 float32 量级。
        #   代价：matmul DFT 比 fft 慢；但 torch 后端只是调试通路（默认是 onnx），
        #   而"两条通路给出同一个数"比这点速度重要 —— 否则 demo 看到的掩码
        #   与 ONNX 件并不是同一套数值。
        #   ⚠ 作用范围仅限**部署/演示通路**：评测链路（`src/evaluation/`）直接
        #     `build_model`，不经过这里 ⇒ 已归档的审计数字（AUC 0.8671 等）不受影响、
        #     也不需要重跑。
        model.set_dft_mode("matmul")
        self._torch_model = model
        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._torch_model.to(self._device)

    def _init_onnx(self, path: str) -> None:
        import onnxruntime as ort

        if not os.path.exists(path):
            alt = path.replace("_simplified", "")
            path = alt if os.path.exists(alt) else path
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 ONNX 模型：{path}，请先运行 src/deploy/export_onnx.py")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        # ★ 显式设线程数：不设 = 用满全核 = 本机实测慢 42%（见 ort_num_threads 注释）。
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._onnx_threads = self.num_threads or ort_num_threads()
        so.intra_op_num_threads = self._onnx_threads
        self._ort = ort.InferenceSession(path, so, providers=providers)
        self._onnx_path = path

    def _init_openvino(self, path: str) -> None:
        import openvino as ov

        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 OpenVINO IR：{path}，请先运行 src/deploy/optimize_openvino.py")
        core = ov.Core()
        model = core.read_model(path)
        self._ov = core.compile_model(model, self.device.upper())
        self._ov_req = self._ov.create_infer_request()

    # ==================================================================
    @staticmethod
    def _crop_geom(h: int, w: int, size: int):
        """返回「等比缩放到短边=size 再中心裁剪」这套几何的参数 (nh, nw, top, left)。

        ★★★ 2026-10-10：**必须与 `preprocess` 用同一套几何**，否则热力图会画错位置。

        为什么要有这个函数：`preprocess` 走的是「等比缩放 + 中心裁剪」，
        于是模型**输出的 224×224 掩码只对应原图中心的 min(h,w)×min(h,w) 方形**。
        旧代码在 `predict` 里写的是 `cv2.resize(mask, (w, h))` —— 那是把
        "中心裁剪块的预测"**整幅拉伸**回原图，等于把热力图铺到了模型根本没看过的区域上，
        而且**位置整体偏移**（256×384 图会偏 31 像素，见 deliverables/14）。
        正确做法：先把掩码**贴回它实际对应的裁剪框**、框外置 0，再缩放回原图尺寸。

        实测影响（256×384）：旧写法与真值框错位 31px；新写法对齐。
        """
        if h == w:
            return size, size, 0, 0
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        top = max(0, (nh - size) // 2)
        left = max(0, (nw - size) // 2)
        return nh, nw, top, left

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """缩放 + 归一化 + HWC->CHW，返回 (1,3,S,S) float32。"""
        import cv2

        h, w = img_bgr.shape[:2]
        S = self.image_size
        if h == w:
            resized = cv2.resize(img_bgr, (S, S),
                                 interpolation=cv2.INTER_AREA if h > S else cv2.INTER_LINEAR)
        else:
            scale = S / min(h, w)
            nh, nw, top, left = self._crop_geom(h, w, S)
            resized = cv2.resize(img_bgr, (nw, nh),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            resized = resized[top:top + S, left:left + S]

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return rgb.transpose(2, 0, 1)[None].astype(np.float32)

    def mask_to_full(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """把模型输出的 (S,S) 掩码**正确铺回** (h,w) 原图。

        旧做法 `cv2.resize(mask, (w,h))` 只在**方形图**上等价（那时裁剪框就是全图）；
        非方形图上它把中心块的预测拉满全图 ⇒ 位置错 + 尺寸错。
        """
        import cv2

        S = self.image_size
        if h == w:
            return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        nh, nw, top, left = self._crop_geom(h, w, S)
        canvas = np.zeros((nh, nw), dtype=np.float32)
        # 模型只对裁剪框内负责 ⇒ 框内贴预测、框外留 0（=不声称那里有篡改）
        m = cv2.resize(np.asarray(mask, dtype=np.float32), (S, S),
                       interpolation=cv2.INTER_LINEAR)
        canvas[top:top + S, left:left + S] = m
        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    def _forward_raw(self, x: np.ndarray):
        """跑一次前向，返回**未过激活**的 (cls_logits (2,), mask_logits (S,S))。

        ★ 存在的意义：三个后端的原始输出语义必须统一。
        torch 分支拿到的是 `cls_logits` / `mask_logits`；onnx / openvino 拿到的
        `out[0]` / `out[1]` 也是 logits。以前 `infer()` 只在 torch 分支做了
        sigmoid，onnx 分支直接把 logits 当概率返回，**违反了
        `DetectionResult.mask` 声明的「(H,W) float32 [0,1] 篡改概率」契约**，
        后果是 onnx 后端的热力图全冷（logits 均值 −22.8 被 clip 成 0）、
        「高置信度区域占比」统计的是 logits>0.5 而不是概率>0.5。
        """
        if self.backend == "onnx":
            out = self._ort.run(None, {self._ort.get_inputs()[0].name: x})
        elif self.backend == "openvino":
            res = self._ov_req.infer({0: x})
            out = [res[i] for i in range(len(res))]
        else:
            with self._torch.no_grad():
                t = self._torch.from_numpy(x).to(self._device)
                o = self._torch_model(t, sample_vib=False, beta=0.0)
                return (o["cls_logits"][0].cpu().numpy().astype(np.float32),
                        o["mask_logits"][0, 0].cpu().numpy().astype(np.float32))

        cls_logits = np.asarray(out[0])[0].astype(np.float32)
        mask_logits = np.asarray(out[1])[0, 0].astype(np.float32)
        return cls_logits, mask_logits

    # ------------------------------------------------------------------
    def infer(self, x: np.ndarray):
        """返回 (cls_prob_fake, mask_prob (S,S))，**两者都在 [0,1]**。"""
        cls_logits, mask_logits = self._forward_raw(x)
        e = np.exp(cls_logits - cls_logits.max())
        cls_prob = float((e / e.sum())[1])
        return cls_prob, (1.0 / (1.0 + np.exp(-mask_logits))).astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _self_check_probes(size: int):
        """构造一对"应该引发不同反应"的探针，供 self_check 使用。

        base = 低频结构图（模拟自然图像）
        x1   = base
        x2   = base 的中央 80×80 换成强噪声（模拟局部篡改）
        可用的检测器应当对 x2 产生明显不同的 mask / cls 输出；死模型两者几乎相同。
        """
        import cv2

        patch = min(80, size // 2)
        rng = np.random.default_rng(20261002)
        gy, gx = np.mgrid[0:size, 0:size].astype("float32") / size
        gray = 0.5 + 0.20 * np.sin(6 * np.pi * gx) * np.cos(4 * np.pi * gy)
        gray = gray + cv2.GaussianBlur(rng.normal(0, 0.05, (size, size)).astype("float32"), (0, 0), 8)
        rgb = np.repeat(gray[None], 3, 0)
        x1 = ((rgb - 0.5) / 0.25).astype("float32")[None]

        x2 = x1.copy()
        c, h = size // 2, patch // 2
        x2[0, :, c - h:c + h, c - h:c + h] = rng.normal(0, 3.0, (3, patch, patch)).astype("float32")
        return x1, x2

    def self_check(self) -> List[str]:
        """自检：确认模型对输入**真的有反应**，返回警告列表（空 = 通过）。

        为什么需要：本仓库踩过一次"静默失效"——`deploy/lite*.onnx` 是
        `export_onnx.py --ckpt ""` 导出的**随机初始化模型**，文件存在、能加载、
        推理不报错，但对任何图像都恒定输出 p_fake=0.4852。只查"文件是否存在"
        拦不住，必须喂**一对应当在输出上产生差异**的输入比对原始 logits。

        判据（阈值由 4 个真实样本标定，见 outputs/_ui_test/probe_calib.log）：
            Δcls  = max|logits2 - logits1|     死模型 0.000 / 坏模型 0.011 / 好模型 0.21
            Δmask = max|mask_logits2 - mask_logits1|   死模型 0.000
          * Δcls < 0.05 且 Δmask < 0.05 → 输出与输入无关，判定不可用；
          * 输出含 NaN/Inf                 → 数值崩溃（零方差输入会触发）。

        成本：2 次前向（onnx 后端本机约 0.2s），只在加载模型时跑一次。
        """
        warns: List[str] = []
        try:
            x1, x2 = self._self_check_probes(self.image_size)
            c1, m1 = self._forward_raw(x1)
            c2, m2 = self._forward_raw(x2)
        except Exception as e:  # noqa: BLE001
            return [f"自检无法执行：{type(e).__name__}: {e}"]

        if not (np.isfinite(c1).all() and np.isfinite(c2).all()
                and np.isfinite(m1).all() and np.isfinite(m2).all()):
            warns.append("模型输出含 NaN/Inf，数值不稳定（零方差输入会直接崩溃）")
        else:
            d_cls = float(np.abs(c2 - c1).max())
            d_mask = float(np.abs(m2 - m1).max())
            if d_cls < 0.05 and d_mask < 0.05:
                warns.append(
                    f"模型输出与输入**无关**（Δcls={d_cls:.2e}, Δmask={d_mask:.2e}）——"
                    "该权重疑似随机初始化 / 导出时权重没加载上，检测结果不可用"
                )
            elif d_cls < 0.05:
                warns.append(f"分类输出几乎不随输入变化（Δcls={d_cls:.2e}），结论可信度存疑")
        return warns

    # ------------------------------------------------------------------
    def detect(self, image_path: str) -> DetectionResult:
        import cv2

        img = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if img is None:
            # 契约：统一抛 ValueError（调用方按 ValueError 处理，UI 只显示
            # "无法读取图像：<路径>"）。缺图时 np.fromfile 抛的 FileNotFoundError
            # 已在 imread_unicode 内部吞掉，这里不会漏出 Errno 串。
            raise ValueError(f"无法读取图像：{image_path}")

        x = self.preprocess(img)
        t0 = time.perf_counter()
        prob, mask = self.infer(x)
        dt = (time.perf_counter() - t0) * 1000

        h, w = img.shape[:2]
        # ★★★ 2026-10-10：改用 `mask_to_full`（贴回中心裁剪框、框外置 0），
        #   不再是 `cv2.resize(mask,(w,h))`（整幅拉伸）。必须与「新几何训练出的权重」
        #   一起上：旧权重学的是"错位的掩码"，对它们这句拉伸反而是自洽的。
        mask_full = self.mask_to_full(mask, h, w)
        overlay = self.build_overlay(img, mask_full)

        warns: List[str] = []
        if self._fallback_note:
            warns.append(self._fallback_note)

        return DetectionResult(
            path=image_path, name=os.path.basename(image_path), width=w, height=h,
            is_fake=bool(prob >= self.threshold), prob_fake=prob,
            mask=mask_full, overlay=overlay, time_ms=dt,
            backend=self.backend, threshold=self.threshold, warnings=warns,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def build_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                      alpha: float = 0.45) -> np.ndarray:
        """篡改区域热力图叠加：颜色越暖表示篡改概率越高。"""
        import cv2

        heat = cv2.applyColorMap((np.clip(mask, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=3, sigmaY=3)
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
        # 高置信度区域加轮廓，便于人工核对
        binary = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        return overlay

    @staticmethod
    def save_mask_png(mask: np.ndarray, path: str) -> str:
        """导出篡改掩码 PNG：白=篡改，黑=真实（申报书 3.2 结果导出模块要求）。

        注意 OpenCV 的 imwrite 不支持中文路径，这里统一用 imencode + tofile。
        """
        import cv2

        binary = (np.clip(mask, 0, 1) > 0.5).astype(np.uint8) * 255
        ok, buf = cv2.imencode(".png", binary)
        if not ok:
            raise RuntimeError("掩码编码失败")
        buf.tofile(path if path.lower().endswith(".png") else path + ".png")
        return path
