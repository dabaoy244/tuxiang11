"""1.3 (1) 梯度隔离机制 —— 梯度停止层 (Gradient Stop)

对应申报书 1.3(1)：在前向传播时恒等传递，在反向传播时阻断梯度回传。
在 F_fusion 与分类支路之间放 GS1，在 F_fusion 与定位支路之间放 GS2。

----------------------------------------------------------------------
⚠ 实现说明：这里必须讲清楚，否则论文里会被审稿人追问
----------------------------------------------------------------------
申报书列出的三条性质：
    ① 分类支路的梯度仅回传至 VIB 模块和 F_fusion
    ② 定位支路的梯度仅回传至定位头和 F_fusion
    ③ F_fusion 同时接收两条支路的梯度

在"共享 F_fusion + 两个兄弟头"这种朴素结构下，① ② ③ **是自动成立的**：
分类头与定位头之间本来就没有计算图连线，梯度不可能跨支路串扰。
也就是说，若把 GS 实现为简单恒等映射，它在数学上是"空操作"，
不会给模型带来任何额外收益——这点在写论文/答辩时必须主动说明，
否则"加了梯度停止层"这句话站不住脚。

因此本模块提供两种可配置语义，把"梯度隔离"变成**真实有效**的机制：

  A) `stop=False`（默认，对应申报书描述的前向恒等行为）
     恒等映射，用于显式标注计算图中的解耦点，工程上便于可视化与调试。

  B) `stop=True`（严格解耦 / 真正的梯度隔离）
     阻断该支路梯度回传到共享空频融合特征 F_fusion。语义为：
       - 打开 cls 侧 GS  → 定位任务的梯度不再塑造共享特征，共享特征
         完全由分类任务驱动（分类优先，泛化性优先）；
       - 打开 loc 侧 GS  → 分类任务的梯度不再塑造共享特征，共享特征
         完全由像素级定位任务驱动（定位细粒度优先）。
     这等价于"任务独立的特征子空间优化"，能显著缓解多任务梯度冲突
     （与 PCGrad / GradNorm 属于同一类思路，但零额外计算开销），
     是消融实验中可以真正观察到指标差异的开关。

推荐实验配置：
    * 三阶段训练中，阶段 1/2 由于另一支路被冻结，GS 无影响；
    * 阶段 3 联合微调时，`gs_stop_cls=True` 常用于"定位精度优先"的场景，
      而 `gs_stop_loc=True` 常用于"跨模型泛化优先"的场景。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
class _GradientStopFunction(torch.autograd.Function):
    """前向恒等，反向按需清零。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, stop: bool):
        ctx.stop = bool(stop)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.stop:
            return torch.zeros_like(grad_output), None
        return grad_output, None


def grad_stop(x: torch.Tensor, stop: bool = True) -> torch.Tensor:
    """函数式梯度停止层。stop=True 时阻断梯度回传。"""
    if x.requires_grad:
        return _GradientStopFunction.apply(x, stop)
    return x


# --------------------------------------------------------------------------
class GradientStopLayer(nn.Module):
    """梯度停止层（nn.Module 形式，便于在 `nn.ModuleDict` 中按阶段开关）。

    Args:
        stop: True  → 反向阻断（严格解耦）；False → 恒等（默认，符合申报书描述）
        name: 用于日志/调试图标注（如 "GS1_cls" / "GS2_loc"）

    使用示例::

        f_fusion = fusion(f_spa, f_freq)          # 共享特征
        z = vib(grad_stop(f_fusion, gs1.stop))    # 分类支路
        mask = loc_head(grad_stop(f_fusion, gs2.stop))   # 定位支路
    """

    def __init__(self, stop: bool = False, name: str = "GS"):
        super().__init__()
        self.stop = stop
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_stop(x, self.stop)

    def extra_repr(self) -> str:
        return f"name={self.name}, stop={self.stop}"

    # ------------------------------------------------------------------
    @staticmethod
    def verify(stop: bool = True) -> dict:
        """自检：验证梯度阻断是否生效。"""
        x = torch.ones(3, 4, requires_grad=True)
        y = grad_stop(x, stop).sum()
        y.backward()
        blocked = bool(torch.allclose(x.grad, torch.zeros_like(x.grad)))
        return {"stop": stop, "上游梯度全零": blocked,
                "grad_shape": tuple(x.grad.shape)}
