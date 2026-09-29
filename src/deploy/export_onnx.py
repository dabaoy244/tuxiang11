"""3.1 模型封装与 CPU 加速优化 —— ONNX 导出与简化

申报书 3.1 要求：
    ① 导出 ONNX 1.14，opset_version=17, do_constant_folding=True, export_params=True
    ② 用 ONNX Simplifier 去除冗余算子：python -m onnxsim model.onnx model_simplified.onnx

用法：
    python -m src.deploy.export_onnx --ckpt checkpoints/vibnet_best.pt --out deploy/vibnet.onnx

设计说明：
    模型的 forward 返回 dict，ONNX 不支持 dict 输出，因此这里用 `_ExportWrapper`
    包一层，只输出 3 个张量：cls_logits / mask_logits / edge_logits。
    VIB 在导出时走 `sample_vib=False`（取 mu），保证推理**确定性**
    —— 否则每次推理结果抖动，桌面工具会出现"同一张图两次结果不同"的 bug。
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """ONNX 导出包装：dict 输出 -> tuple 输出，且 VIB 走确定性路径。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x, sample_vib=False, beta=0.0)
        cls_logits = out["cls_logits"]
        mask_logits = out["mask_logits"]
        edge_logits = out.get("edge_logits")
        if edge_logits is None:
            edge_logits = mask_logits
        return cls_logits, mask_logits, edge_logits


# --------------------------------------------------------------------------
def export_onnx(
    ckpt_path: str,
    out_path: str = "deploy/vibnet.onnx",
    opset: int = 17,
    image_size: int = 224,
    config_path: str = "configs/default.yaml",
    dynamic_batch: bool = True,
    verify: bool = True,
) -> str:
    """把 PyTorch 权重导出为 ONNX。"""
    from ..models.spatial_branch import align_cfg_to_checkpoint
    from ..models.vibnet import build_model, load_config

    cfg = load_config(config_path)
    cfg["data"]["image_size"] = image_size
    state = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        # 按 checkpoint 反推骨干档位并对齐，避免旧权重配新配置时导出成随机模型
        align_cfg_to_checkpoint(cfg, state)
    model = build_model(cfg)
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[export] 缺失权重 {len(missing)} 项（若使用兜底骨干属正常）")
        model.eval()
    else:
        print(f"[export] ⚠ 未找到 {ckpt_path}，导出随机初始化模型（仅用于流程验证）")

    wrapper = ExportWrapper(model).eval()
    # ★ 关键一步：把频域分支的 DFT 从 torch.fft 切到 matmul 实现。
    #   PyTorch 的 aten::fft_fft2 无法导出到 ONNX opset 17，
    #   不切就会报 UnsupportedOperatorError。两者数值等价（误差 ~1e-5）。
    model.set_dft_mode("matmul")
    dummy = torch.randn(1, 3, image_size, image_size)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "cls_logits": {0: "batch"},
            "mask_logits": {0: "batch"},
            "edge_logits": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, out_path,
            input_names=["input"],
            output_names=["cls_logits", "mask_logits", "edge_logits"],
            opset_version=opset,
            do_constant_folding=True,       # 申报书要求
            export_params=True,             # 申报书要求
            dynamic_axes=dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
    print(f"[export] ONNX 已导出：{out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, opset={opset})")

    if verify:
        _verify_onnx(out_path, dummy)
    return out_path


def _verify_onnx(path: str, dummy: torch.Tensor) -> None:
    try:
        import onnx

        m = onnx.load(path)
        onnx.checker.check_model(m)
        print("[export] onnx.checker 通过")
    except ImportError:
        print("[export] 未安装 onnx，跳过结构校验（pip install onnx）")
        return
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})
        print(f"[export] onnxruntime 推理通过，输出："
              f"{[o.shape for o in out]}")
    except ImportError:
        print("[export] 未安装 onnxruntime，跳过推理校验（pip install onnxruntime）")


# --------------------------------------------------------------------------
def simplify_onnx(src: str, dst: Optional[str] = None) -> str:
    """调用 onnx-simplifier 去冗余算子。"""
    dst = dst or src.replace(".onnx", "_simplified.onnx")
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("[simplify] 未安装 onnx-simplifier，跳过。"
              "安装：pip install onnx-simplifier  或直接运行："
              f"python -m onnxsim {src} {dst}")
        return src

    model = onnx.load(src)
    model_sim, ok = simplify(model)
    if ok:
        onnx.save(model_sim, dst)
        print(f"[simplify] 简化成功：{src} -> {dst} "
              f"({os.path.getsize(src)/1e6:.1f}MB -> {os.path.getsize(dst)/1e6:.1f}MB)")
        return dst
    print("[simplify] 简化失败，保留原始模型")
    return src


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vibnet_best.pt")
    ap.add_argument("--out", default="deploy/vibnet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--simplify", action="store_true")
    args = ap.parse_args()

    out = export_onnx(args.ckpt, args.out, args.opset, args.image_size, args.config)
    if args.simplify:
        simplify_onnx(out)


if __name__ == "__main__":
    main()
