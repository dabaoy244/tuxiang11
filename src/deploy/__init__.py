"""部署相关：ONNX 导出、OpenVINO 加速、统一推理封装。"""

from .inference import Detector, DetectionResult, list_available_backends
from .export_onnx import export_onnx, simplify_onnx

__all__ = [
    "Detector",
    "DetectionResult",
    "list_available_backends",
    "export_onnx",
    "simplify_onnx",
]
