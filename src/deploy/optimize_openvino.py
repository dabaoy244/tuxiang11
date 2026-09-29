"""3.1 CPU 加速优化 —— OpenVINO IR 转换与性能测试

申报书 3.1 要求：
    采用 OpenVINO 2024.1 工具套件对 ONNX 模型做 Intel CPU 指令集优化，
    转换为 OpenVINO IR 格式；用 benchmark_app 做性能测试与调优。

⚠ 版本注意（写进技术报告的"环境说明"）：
    OpenVINO 2024.x 官方 wheel 只支持到 Python 3.12；
    若本机是 Python 3.13，请用 OpenVINO 2025.0+（API 兼容），
    或另建 Python 3.10/3.11 环境专用于部署端。

用法：
    # 方式 A：Python API（推荐，跨版本最稳）
    python -m src.deploy.optimize_openvino --onnx deploy/vibnet.onnx --out deploy/vibnet_ir

    # 方式 B：命令行工具（申报书里写的做法）
    mo --input_model deploy/vibnet.onnx --output_dir deploy/vibnet_ir
    benchmark_app -m deploy/vibnet_ir/vibnet.xml -d CPU -api async
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np


def convert_to_openvino(onnx_path: str, out_dir: str = "deploy/vibnet_ir",
                        fp16: bool = False) -> Optional[str]:
    """ONNX -> OpenVINO IR。返回 .xml 路径，失败返回 None。"""
    os.makedirs(out_dir, exist_ok=True)
    try:
        import openvino as ov
    except ImportError:
        print("[openvino] 未安装。安装：pip install openvino  "
              "（Python 3.13 需 openvino>=2025.0）")
        print("[openvino] 或使用命令行：mo --input_model <onnx> --output_dir <dir>")
        return None

    print(f"[openvino] 版本 {ov.__version__}")
    model = ov.convert_model(onnx_path)
    xml_path = os.path.join(out_dir, os.path.basename(onnx_path).replace(".onnx", ".xml"))
    ov.save_model(model, xml_path, compress_to_fp16=fp16)
    size = os.path.getsize(xml_path) / 1e6
    bin_path = xml_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size += os.path.getsize(bin_path) / 1e6
    print(f"[openvino] IR 已生成：{xml_path}  合计 {size:.1f} MB  fp16={fp16}")
    return xml_path


def benchmark(onnx_path: str = "", ir_path: str = "", image_size: int = 224,
              n_warmup: int = 5, n_runs: int = 50, device: str = "CPU") -> dict:
    """测量 CPU 推理速度（张/秒）与模型体积，核对申报书指标。

    同时测 ONNXRuntime 和 OpenVINO 两条路径，便于写"加速比"对比表。
    """
    result: dict = {"device": device, "image_size": image_size}
    x = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    # ---- ONNXRuntime
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device.upper() == "GPU" else ["CPUExecutionProvider"],
            )
            name = sess.get_inputs()[0].name
            for _ in range(n_warmup):
                sess.run(None, {name: x})
            t0 = time.perf_counter()
            for _ in range(n_runs):
                sess.run(None, {name: x})
            dt = time.perf_counter() - t0
            result["onnxruntime"] = {
                "fps": n_runs / dt,
                "latency_ms": dt / n_runs * 1000,
                "size_mb": os.path.getsize(onnx_path) / 1e6,
            }
        except ImportError:
            print("[bench] 未安装 onnxruntime")

    # ---- OpenVINO
    if ir_path and os.path.exists(ir_path):
        try:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(ir_path)
            compiled = core.compile_model(model, device)
            infer = compiled.create_infer_request()
            for _ in range(n_warmup):
                infer.infer({0: x})
            t0 = time.perf_counter()
            for _ in range(n_runs):
                infer.infer({0: x})
            dt = time.perf_counter() - t0
            size = os.path.getsize(ir_path) / 1e6
            binp = ir_path.replace(".xml", ".bin")
            if os.path.exists(binp):
                size += os.path.getsize(binp) / 1e6
            result["openvino"] = {
                "fps": n_runs / dt,
                "latency_ms": dt / n_runs * 1000,
                "size_mb": size,
                "version": ov.__version__,
            }
        except ImportError:
            print("[bench] 未安装 openvino")

    # ---- 达标判定（申报书 6.3：CPU >= 8 张/秒，模型 <= 120MB）
    best_fps = max((v.get("fps", 0) for k, v in result.items() if isinstance(v, dict)),
                   default=0.0)
    sizes = [v.get("size_mb", 999) for k, v in result.items() if isinstance(v, dict)]
    result["check"] = {
        "CPU_fps": round(best_fps, 2),
        "target_fps>=8": bool(best_fps >= 8),
        "size_mb": round(min(sizes), 2) if sizes else None,
        "target_size<=120MB": bool(sizes and min(sizes) <= 120),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def quantize_int8(ir_path: str, out_dir: str = "deploy/vibnet_ir_int8") -> Optional[str]:
    """OpenVINO NNCF 训练后 INT8 量化。这是把模型压到 120MB 以内的关键一步。

    量化后 CLIP 骨干约 86MB，全模型可稳定落在 120MB 以内，且 Intel CPU 上
    吞吐通常还能再提升 1.5~2 倍（配合 AVX-512/VNNI 指令集）。

    需要：pip install nncf；并提供 100~300 张校准图（真实与伪造各半最佳）。
    """
    try:
        import nncf
    except ImportError:
        print("[int8] 未安装 nncf。安装：pip install nncf  "
              "（若无 NNCF，可用 Legacy POT：pot -c pot_config.yml）")
        return ir_path
    try:
        import openvino as ov
    except ImportError:
        return ir_path

    import glob

    core = ov.Core()
    model = core.read_model(ir_path)
    calib_imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        calib_imgs += glob.glob(os.path.join("data", "**", ext), recursive=True)
    if not calib_imgs:
        print("[int8] ⚠ 未在 data/ 下找到校准图像，请先准备 100~300 张校准图。")
        return ir_path

    import numpy as np
    from PIL import Image

    def transform(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.array([0.48145466, 0.4578275, 0.40821073])) / \
            np.array([0.26862954, 0.26130258, 0.27577711])
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    calib = [transform(p) for p in calib_imgs[:300]]
    print(f"[int8] 使用 {len(calib)} 张校准图")
    qmodel = nncf.quantize(model, nncf.Dataset(calib))
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, os.path.basename(ir_path))
    ov.save_model(qmodel, xml_path)
    size = os.path.getsize(xml_path) / 1e6
    bin_path = xml_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        size += os.path.getsize(bin_path) / 1e6
    print(f"[int8] 量化模型已生成：{xml_path}  合计 {size:.1f} MB")
    return xml_path


# --------------------------------------------------------------------------
def estimate_model_size(params_millions: float,
                        target_mb: float = 120.0) -> dict:
    """估算不同精度下的模型体积，并判断是否满足申报书 ≤120MB 指标。

    ⚠ 这是本项目最容易"翻车"的一条指标，必须提前算清楚：
      CLIP-ViT-B/16 视觉塔单独就有约 86M 参数（fp32 ≈ 344MB），
      再加上 Mobile-UNetv2（≈3.7M）和各模块，fp32 全模型约 350MB，
      **远超 120MB**。因此想达标只有三条路（按性价比排序）：

        1) INT8 量化（OpenVINO NNCF / POT 训练后量化）
           86M 参数 -> 约 86MB + 少量开销，可稳定落在 120MB 以内，且 CPU 更快；
        2) 换更小的 CLIP 变体（MobileCLIP / TinyCLIP / ViT-B/32 蒸馏版），
           参数量可降到 10~30M；
        3) 知识蒸馏：用 CLIP 教师指导一个小骨干学生（例如 MobileNetV3 + 蒸馏），
           这样既能保住泛化性，又能真正"轻量"。

      仅做 FP16 只能到约 172MB，仍然不达标 —— 这点在中期检查时一定要说明。
    """
    return {
        "参数量(M)": round(params_millions, 2),
        "FP32(MB)": round(params_millions * 4, 1),
        "FP16(MB)": round(params_millions * 2, 1),
        "INT8(MB)": round(params_millions * 1, 1),
        "target_MB": target_mb,
        "FP32达标": params_millions * 4 <= target_mb,
        "FP16达标": params_millions * 2 <= target_mb,
        "INT8达标": params_millions * 1 <= target_mb,
        "建议": ("INT8 量化即可达标" if params_millions * 1 <= target_mb
                 else "需要换更小的骨干或做知识蒸馏"),
    }


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="deploy/vibnet.onnx")
    ap.add_argument("--out", default="deploy/vibnet_ir")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true",
                    help="用 NNCF 做训练后 INT8 量化（需 pip install nncf）")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--size-only", action="store_true",
                    help="只按参数量估算各精度体积，不做实际转换")
    ap.add_argument("--params-m", type=float, default=None,
                    help="配合 --size-only 使用：模型参数量（百万）")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    if args.size_only:
        if args.params_m is None:
            print("请用 --params-m 指定参数量（百万），例如 --params-m 33.84")
            return
        print(json.dumps(estimate_model_size(args.params_m), ensure_ascii=False, indent=2))
        return

    ir = None
    if os.path.exists(args.onnx):
        ir = convert_to_openvino(args.onnx, args.out, args.fp16)
        if args.int8 and ir:
            ir = quantize_int8(ir, args.out)
        out_json = os.path.join(os.path.dirname(args.out) or ".", "benchmark.json")
        res = benchmark(args.onnx, ir or "", args.image_size, n_runs=args.runs)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[bench] 结果已写入 {out_json}")
    else:
        print(f"[openvino] 未找到 {args.onnx}，请先运行 src/deploy/export_onnx.py")


if __name__ == "__main__":
    main()
