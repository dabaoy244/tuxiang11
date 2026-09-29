"""2.3 模型轻量化 —— ONNX Runtime 动态 INT8 量化

为什么需要这一步（**关键结论，见 docs/09**）
----------------------------------------------------------------------
申报书 6.3 写的是「模型体积 ≤ 120MB」，而 1.1(1) 指定骨干为 CLIP-ViT-B/16。
这两条**在 FP32 下直接冲突**——实测：

    CLIP-ViT-B/16 结构骨干 85.81M + 其余模块 4.72M = 90.53M 参数
    FP32 体积 = 362.1 MB        ❌ 超标 3 倍

要让 b16 档骨干满足 ≤120MB，只有两条路：
    A. 换轻量档骨干（configs/lite.yaml，s16 → 104.8MB / ti16 → 39.8MB）；
    B. 对 b16 做 INT8 量化 —— 权重 4 字节 → 1 字节，90.53M × 1B ≈ 90.5MB ✅

本脚本实现 B。动态量化（dynamic quantization）只需一次前向采样确定激活的
量化范围，**不需要校准数据集**，对 CNN/Transformer 混合结构最省事。

用法
----
    python -m src.deploy.quantize_int8 \
        --onnx deploy/vibnet_simplified.onnx \
        --out  deploy/vibnet_int8.onnx \
        --bench

----------------------------------------------------------------------
兼容性注意
----------------------------------------------------------------------
* 量化会插入 `DynamicQuantizeLinear` / `MatMulInteger` 等算子，
  **OpenVINO 2023 之前对 MatMulInteger 支持不完整**；若后续要转 OpenVINO IR，
  建议用 OpenVINO 自己的 NNCF PTQ（`src/deploy/optimize_openvino.py --int8`），
  而不是先把 ONNX 量化再转换。两条路线在本文档 2.3 节都有说明。
* `Conv` 算子默认不参与动态量化（ORT 只量化 MatMul/Gemm/Attention 类），
  所以 Mobile-UNetv2 定位头基本保持 FP32，骨干的注意力/MLP 会被压到 INT8。
  这正好符合「骨干是大头」的参数分布。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------
def _size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def quantize_int8(
    onnx_path: str,
    out_path: str,
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
    extra_exclude: Optional[List[str]] = None,
) -> str:
    """动态 INT8 量化。

    Args:
        onnx_path: 输入 ONNX（建议先用 simplify 过的版本）。
        out_path: 输出 ONNX 路径。
        weight_type: "QInt8"（有符号，推荐）或 "QUInt8"（无符号，部分 CPU 更快）。
        per_channel: 逐通道量化，精度更好但部分推理后端不支持。
        reduce_range: 把范围压到 7bit，兼容老款无 VNNI 的 CPU。
        extra_exclude: 额外排除的算子名列表。

    Returns:
        输出文件路径。
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"找不到 ONNX 模型：{onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    qtype = QuantType.QInt8 if weight_type.upper() == "QINT8" else QuantType.QUInt8

    # 定位头与分类头对数值敏感，排除掉不量化
    nodes_to_exclude = list(extra_exclude or [])
    nodes_to_exclude += _collect_head_nodes(onnx_path)

    before = _size_mb(onnx_path)
    print(f"[int8] 量化前 {before:.1f} MB -> 目标 {weight_type}"
          f"{' / per-channel' if per_channel else ''}")
    if nodes_to_exclude:
        print(f"[int8] 排除 {len(nodes_to_exclude)} 个头部算子（定位头/分类头）")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_path,
        weight_type=qtype,
        per_channel=per_channel,
        reduce_range=reduce_range,
        nodes_to_exclude=nodes_to_exclude or None,
        extra_options={"EnableSubgraph": True, "ForceQuantizeNoInputCheck": False},
    )

    after = _size_mb(out_path)
    print(f"[int8] 量化后 {after:.1f} MB  （压缩 {before / max(after, 1e-9):.2f}x，"
          f"省下 {before - after:.1f} MB）")
    return out_path


def _collect_head_nodes(onnx_path: str, keywords=("cls_head", "vib", "out_conv", "refine")) -> List[str]:
    """把分类头 / VIB / 定位输出层的算子排除出量化范围。

    这些层参数量只有零点几 M，量化带来的体积收益可忽略，
    但一旦掉精度就直接影响 ACC / mIoU，性价比极低。
    """
    import onnx

    model = onnx.load(onnx_path)
    keep = []
    for node in model.graph.node:
        blob = " ".join([node.name or ""] + list(node.input) + list(node.output))
        if any(k in blob for k in keywords):
            keep.append(node.name)
    return keep


# --------------------------------------------------------------------------
def verify_quantized(fp32_path: str, int8_path: str, shape=(1, 3, 224, 224),
                     atol: float = 0.15) -> Dict[str, float]:
    """对比 FP32 与 INT8 的输出偏差——量化后的必要验收步骤。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s32 = ort.InferenceSession(fp32_path, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8_path, so, providers=["CPUExecutionProvider"])

    name = s32.get_inputs()[0].name
    x = np.random.randn(*shape).astype(np.float32)
    o32 = s32.run(None, {name: x})
    o8 = s8.run(None, {name: x})

    diffs: Dict[str, float] = {}
    for i, (a, b) in enumerate(zip(o32, o8)):
        a, b = np.asarray(a), np.asarray(b)
        scale = float(np.abs(a).max()) + 1e-8
        diffs[f"out{i}_max_abs_diff"] = float(np.abs(a - b).max())
        diffs[f"out{i}_rel_to_peak"] = float(np.abs(a - b).max() / scale)
    worst = max(diffs[f"out{i}_rel_to_peak"] for i in range(len(o32)))
    diffs["worst_rel_to_peak"] = worst
    diffs["pass"] = float(worst < atol)
    return diffs


# --------------------------------------------------------------------------
def benchmark(paths: List[str], shape=(1, 3, 224, 224), n_runs: int = 30) -> List[Dict]:
    """对若干 ONNX 模型做 CPU 推理延迟对比。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ★ 不要用满全核：本机实测 7 核跑满反而慢 42%、且波动大到不可比（见
    #   src/deploy/inference.py 的 ort_num_threads 注释与 docs/21）。
    try:
        from .inference import ort_num_threads
        so.intra_op_num_threads = ort_num_threads()
    except Exception:                                        # noqa: BLE001
        so.intra_op_num_threads = min(4, os.cpu_count() or 4)

    results = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[bench] 跳过不存在的 {p}")
            continue
        sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = np.random.randn(*shape).astype(np.float32)
        for _ in range(3):                                   # warmup
            sess.run(None, {name: x})
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {name: x})
        dt = (time.perf_counter() - t0) / n_runs
        rec = {
            "model": os.path.basename(p),
            "size_MB": round(_size_mb(p), 1),
            "latency_ms": round(dt * 1000, 1),
            "fps": round(1.0 / dt, 2),
        }
        results.append(rec)
        print(f"[bench] {rec['model']:<28s} {rec['size_MB']:>7.1f} MB  "
              f"{rec['latency_ms']:>7.1f} ms  {rec['fps']:>6.2f} img/s")
    return results


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX 动态 INT8 量化")
    ap.add_argument("--onnx", default="deploy/vibnet_simplified.onnx")
    ap.add_argument("--out", default="deploy/vibnet_int8.onnx")
    ap.add_argument("--weight-type", default="QInt8", choices=["QInt8", "QUInt8"])
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--reduce-range", action="store_true")
    ap.add_argument("--bench", action="store_true", help="量化前后延迟对比")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--report", default="", help="把结果写成 JSON 的路径")
    args = ap.parse_args()

    quantize_int8(args.onnx, args.out, args.weight_type,
                  args.per_channel, args.reduce_range)

    report: Dict[str, object] = {
        "source": args.onnx,
        "quantized": args.out,
        "size_fp32_MB": round(_size_mb(args.onnx), 1),
        "size_int8_MB": round(_size_mb(args.out), 1),
        "shape": [1, 3, 224, 224],
    }
    try:
        report["numeric_check"] = verify_quantized(args.onnx, args.out)
        ok = report["numeric_check"]["pass"] == 1.0
        print(f"[int8] 数值一致性检查 {'通过 ✓' if ok else '未通过 ✗'}  "
              f"最大相对偏差 {report['numeric_check']['worst_rel_to_peak']:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"[int8] 数值检查跳过：{type(e).__name__}: {e}")

    if args.bench:
        report["benchmark"] = benchmark([args.onnx, args.out], n_runs=args.runs)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[int8] 报告已写入 {args.report}")


if __name__ == "__main__":
    main()
