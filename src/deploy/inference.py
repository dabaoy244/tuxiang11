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
        ckpt: str = "checkpoints/vibnet_best.pt",
        onnx_path: str = "deploy/vibnet_simplified.onnx",
        openvino_ir: str = "deploy/vibnet_ir/vibnet.xml",
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
    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """缩放 + 归一化 + HWC->CHW，返回 (1,3,S,S) float32。"""
        import cv2

        h, w = img_bgr.shape[:2]
        if h == w:
            resized = cv2.resize(img_bgr, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_AREA if h > self.image_size else cv2.INTER_LINEAR)
        else:
            scale = self.image_size / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            resized = cv2.resize(img_bgr, (nw, nh),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            top, left = max(0, (nh - self.image_size) // 2), max(0, (nw - self.image_size) // 2)
            resized = resized[top:top + self.image_size, left:left + self.image_size]

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return rgb.transpose(2, 0, 1)[None].astype(np.float32)

    # ------------------------------------------------------------------
    def infer(self, x: np.ndarray):
        """返回 (cls_prob_fake, mask_prob (S,S))。"""
        if self.backend == "onnx":
            out = self._ort.run(None, {self._ort.get_inputs()[0].name: x})
        elif self.backend == "openvino":
            res = self._ov_req.infer({0: x})
            out = [res[i] for i in range(len(res))]
        else:
            with self._torch.no_grad():
                t = self._torch.from_numpy(x).to(self._device)
                o = self._torch_model(t, sample_vib=False, beta=0.0)
                cls = self._torch.softmax(o["cls_logits"], dim=-1)[0, 1].item()
                mask = o["mask_prob"][0, 0].cpu().numpy()
                return float(cls), mask

        cls_logits = np.asarray(out[0])[0]
        e = np.exp(cls_logits - cls_logits.max())
        cls_prob = float((e / e.sum())[1])
        mask = np.asarray(out[1])[0, 0]
        return cls_prob, mask.astype(np.float32)

    # ------------------------------------------------------------------
    def detect(self, image_path: str) -> DetectionResult:
        import cv2

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        x = self.preprocess(img)
        t0 = time.perf_counter()
        prob, mask = self.infer(x)
        dt = (time.perf_counter() - t0) * 1000

        h, w = img.shape[:2]
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
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
