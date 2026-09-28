"""部署路径的 CPU 吞吐基准（ONNX Runtime）——申报书「CPU ≥ 8 张/秒」的验收脚本。

为什么要单独测 ONNX Runtime 而不是 PyTorch：
    申报书 3.1 的指标口径是「经模型封装与 CPU 加速优化后」的吞吐。
    实测同一台机器上，PyTorch 的 eager 推理比 ONNX Runtime 慢 5~8 倍
    （图优化 + 算子融合 + 线程池策略差异），所以两个数字都要报，
    但验收应以本项目实际交付的推理路径（ONNX / OpenVINO）为准。

用法：
    python scripts/benchmark_cpu.py                       # 默认测 deploy/lite*.onnx
    python scripts/benchmark_cpu.py --models deploy/lite_int8.onnx --reps 5
    python scripts/benchmark_cpu.py --threads 4           # 固定线程数，减少波动

输出：
    outputs/cpu_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env_info() -> dict:
    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
    except Exception:  # noqa: BLE001
        ort_ver = "not installed"
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "onnxruntime": ort_ver,
        "torch": _torch_ver(),
    }


def _torch_ver() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:  # noqa: BLE001
        return "not installed"


def bench_onnx(path: str, shape=(1, 3, 224, 224), reps: int = 3, runs: int = 20,
               threads: int = 0, optimize: bool = True) -> dict:
    """多轮测量，返回中位数与各轮明细。"""
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    if threads > 0:
        so.intra_op_num_threads = threads

    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    x = np.random.randn(*shape).astype(np.float32)

    for _ in range(5):                                  # warmup
        sess.run(None, {iname: x})

    per_rep: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(None, {iname: x})
        per_rep.append((time.perf_counter() - t0) / runs)

    med = statistics.median(per_rep)
    return {
        "model": os.path.basename(path),
        "size_MB": round(os.path.getsize(path) / 1e6, 1),
        "intra_op_threads": threads or "default",
        "reps_ms": [round(v * 1000, 1) for v in per_rep],
        "median_ms": round(med * 1000, 1),
        "median_fps": round(1.0 / med, 2),
        "best_fps": round(1.0 / min(per_rep), 2),
        "worst_fps": round(1.0 / max(per_rep), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["deploy/lite.onnx", "deploy/lite_simplified.onnx",
                             "deploy/lite_int8.onnx"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--reps", type=int, default=3, help="重复轮数，取中位数以抗噪")
    ap.add_argument("--runs", type=int, default=20, help="每轮推理次数")
    ap.add_argument("--threads", type=int, default=0, help="0=默认(全部核心)")
    ap.add_argument("--out", default="outputs/cpu_benchmark.json")
    args = ap.parse_args()

    env = _env_info()
    print(f"[bench] {env['platform']}")
    print(f"[bench] CPU 核心 {env['cpu_count']} / onnxruntime {env['onnxruntime']} / "
          f"输入 {args.image_size}x{args.image_size} / batch=1")
    print(f"[bench] 每模型 {args.reps} 轮 x {args.runs} 次，取中位数\n")

    rows = []
    for p in args.models:
        if not os.path.exists(p):
            print(f"[bench] 跳过（不存在）：{p}")
            continue
        r = bench_onnx(p, (1, 3, args.image_size, args.image_size),
                       args.reps, args.runs, args.threads)
        r["meets_cpu_8fps"] = r["median_fps"] >= 8
        rows.append(r)
        print(f"[bench] {r['model']:<26s} {r['size_MB']:>6.1f} MB  "
              f"中位 {r['median_ms']:>6.1f} ms  {r['median_fps']:>6.2f} img/s  "
              f"(最快 {r['best_fps']:.2f} / 最慢 {r['worst_fps']:.2f})  "
              f"{'达标 ✓' if r['meets_cpu_8fps'] else '未达标 ✗'}")

    report = {"env": env, "image_size": args.image_size,
              "reps": args.reps, "runs_per_rep": args.runs, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[bench] 已写入 {args.out}")


if __name__ == "__main__":
    main()
