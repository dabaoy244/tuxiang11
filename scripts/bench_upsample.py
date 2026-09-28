"""单算子微基准：定位支路里「14x14 -> 224x224 双线性上采样」的**反向**是否吃掉全部时间。

背景（2026-09-26）：
  stage1 (use_tasks=['cls']) 0.18 s/iter，stage2 (use_tasks=['loc','edge']) 1.52 s/iter，
  差 8.3 倍。而 stage1 里定位头的 forward 照样在跑（VIBNet.forward 里定位支路是无条件
  执行的），只是 loss 不依赖 mask_logits ⇒ **autograd 把整条定位支路的 backward 剪掉**。
  所以 8.3 倍的全部差额 = 定位支路的 backward。

  本地 CPU 实测该支路 fwd/bwd = 1:2.5（正常），且它的 FLOPs 只占全模型 ~7%，
  ⇒ 绝不是算力问题。三个后端探针（bf16+cuDNN / 纯 fp32 / 关 cuDNN）耗时相同
  ⇒ 出问题的 kernel 必须**不经过 cuDNN**（否则关掉 cuDNN 一定会变）。

  最符合这两条的候选：`upsample_bilinear2d_backward`（纯 ATen CUDA kernel）。
  它按**输出像素**并行、用 atomicAdd 累加到 grad_input：本处输出
  16x64x224x224=51.4M 个像素，grad_input 只有 16x64x14x14=200k 个位置，
  平均 **257 个线程争抢同一个地址** —— 典型的原子竞争灾难，且与 cuDNN 无关。

本脚本就是用真实形状把这件事钉死：
  1) 同一形状下 fwd / fwd+bwd 各多少 ms；
  2) 把通道数从 64 降到 1（=只上采样 1 通道 logits），能否线性变快；
  3) 换成 nearest（无原子累加）对比；
  4) 拿同分辨率的 3x3 卷积做参照，说明"正常算子该有多快"。
运行：python scripts/bench_upsample.py            # 有卡自动用 cuda
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[bench_upsample] device={dev}  torch={torch.__version__}")
if dev == "cuda":
    print(f"[bench_upsample] cudnn.enabled={torch.backends.cudnn.enabled} "
          f"benchmark={torch.backends.cudnn.benchmark} "
          f"| {torch.cuda.get_device_name(0)}")


def bench(fn, n: int = 10) -> float:
    for _ in range(3):                       # warmup
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


B, SRC, DST = 16, 14, 224


def up_fwd(ch: int, mode: str):
    x = torch.randn(B, ch, SRC, SRC, device=dev)
    return lambda: F.interpolate(x, size=(DST, DST), mode=mode, align_corners=False)


def up_fwd_bwd(ch: int, mode: str):
    kwargs = {"align_corners": False} if mode == "bilinear" else {}

    def run():
        x = torch.randn(B, ch, SRC, SRC, device=dev, requires_grad=True)
        y = F.interpolate(x, size=(DST, DST), mode=mode, **kwargs)
        y.sum().backward()
    return run


# 参照：同一张 224x224 特征图上、定位头真正的 3x3 卷积（64->32）
ref_conv = torch.nn.Conv2d(64, 32, 3, 1, 1, bias=False).to(dev)
ref_x = torch.randn(B, 64, DST, DST, device=dev, requires_grad=True)


def ref_run():
    ref_conv(ref_x).sum().backward()


print(f"\n{'case':46s} {'ms':>9}")
print("-" * 58)
print(f"{'参照: conv3x3 64->32 @224  fwd+bwd':46s} {bench(ref_run):9.1f}")
for ch in (64, 32, 1):
    print(f"{f'bilinear 14->224  C={ch}  fwd only':46s} {bench(up_fwd(ch, 'bilinear')):9.1f}")
for ch in (64, 32, 1):
    print(f"{f'bilinear 14->224  C={ch}  fwd+bwd':46s} {bench(up_fwd_bwd(ch, 'bilinear')):9.1f}")
for ch in (64, 1):
    print(f"{f'nearest  14->224  C={ch}  fwd+bwd':46s} {bench(up_fwd_bwd(ch, 'nearest')):9.1f}")
print("-" * 58)
if dev == "cuda":
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"peak VRAM {peak:.2f} GiB")
