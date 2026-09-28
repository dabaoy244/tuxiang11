"""空频双分支轻量 VIB-Net —— 模型组件。

模块与申报书章节对应关系：
    spatial_branch.py  -> 1.1 (1) 空域特征提取分支
    freq_branch.py     -> 1.1 (2) 频域特征提取分支（幅相联合 + 可学习掩码）
    cs_cam.py          -> 1.1 (3) 通道-空间交叉注意力自适应融合
    vib.py             -> 1.2 分层 VIB 信息瓶颈特征提纯
    gradient_stop.py   -> 1.3 (1) 梯度隔离机制
    mobile_unetv2.py   -> 1.3 (3) 定位支路 + 边缘监督分支
    vibnet.py          -> 整体模型装配（图表2 技术路线）
"""

from .spatial_branch import SpatialBranch
from .freq_branch import FrequencyBranch
from .cs_cam import CSCAM
from .vib import HierarchicalVIB
from .gradient_stop import GradientStopLayer, grad_stop
from .mobile_unetv2 import MobileUNetV2, EdgeHead
from .vibnet import VIBNet, build_model

__all__ = [
    "SpatialBranch",
    "FrequencyBranch",
    "CSCAM",
    "HierarchicalVIB",
    "GradientStopLayer",
    "grad_stop",
    "MobileUNetV2",
    "EdgeHead",
    "VIBNet",
    "build_model",
]
