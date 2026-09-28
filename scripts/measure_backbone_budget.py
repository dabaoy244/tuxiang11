"""骨干规模与 CPU 吞吐的实测脚本（产出 docs/09 的表格数据）。

    python scripts/measure_backbone_budget.py
    python scripts/measure_backbone_budget.py --variants b16 s16 ti16 ti8

输出 outputs/backbone_budget.json，包含每个档位的：
    参数量 / FP32 体积 / PyTorch CPU 延迟 / 是否满足「≤120MB」与「≥8 张/秒」
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.models.vibnet import build_model, load_config  # noqa: E402


def measure(variants, cfg_path: str = "configs/default.yaml",
            image_size: int = 224, runs: int = 10, warmup: int = 3) -> list:
    base = load_config(cfg_path)
    # 强制走离线可构造骨干，否则会去联网找 CLIP 权重
    base["model"]["spatial"]["backbone"] = "tiny_vit"
    base["data"]["image_size"] = image_size

    rows = []
    x = torch.randn(1, 3, image_size, image_size)
    for v in variants:
        cfg = copy.deepcopy(base)
        cfg["model"]["spatial"]["backbone_variant"] = v
        model = build_model(cfg).eval()
        s = model.summary()
        with torch.no_grad():
            for _ in range(warmup):                      # 充分预热，避免首调用线程池开销污染
                model(x)
            t0 = time.perf_counter()
            for _ in range(runs):
                model(x)
            dt = (time.perf_counter() - t0) / runs
        row = {
            "config": os.path.basename(cfg_path),
            "variant": v,
            "params_M": round(s["total(M)"], 2),
            "fp32_MB": round(s["fp32_size(MB)"], 1),
            "torch_cpu_ms": round(dt * 1000, 1),
            "torch_cpu_fps": round(1.0 / dt, 2),
            "meets_size_120MB": s["fp32_size(MB)"] <= 120,
            "meets_cpu_8fps": (1.0 / dt) >= 8,
        }
        rows.append(row)
        print(f"  {os.path.basename(cfg_path):<16s} {v:<5s} {row['params_M']:>7.2f} M  "
              f"{row['fp32_MB']:>7.1f} MB  {row['torch_cpu_ms']:>7.1f} ms  "
              f"{row['torch_cpu_fps']:>6.2f} img/s  "
              f"体积{'✓' if row['meets_size_120MB'] else '✗'} "
              f"速度{'✓' if row['meets_cpu_8fps'] else '✗'}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["configs/default.yaml", "configs/lite.yaml"],
                    help="可给多个配置，用于对比「骨干档位」与「头部通道宽度」的影响")
    ap.add_argument("--variants", nargs="+", default=["b16", "s16", "ti16", "ti8"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", default="outputs/backbone_budget.json")
    args = ap.parse_args()

    print(f"[budget] 输入 {args.image_size}x{args.image_size}，PyTorch CPU，batch=1，"
          f"{args.runs} 次取均值")
    rows = []
    for c in args.configs:
        rows.extend(measure(args.variants, c, args.image_size, args.runs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"image_size": args.image_size, "runs": args.runs, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"[budget] 已写入 {args.out}")


if __name__ == "__main__":
    main()
