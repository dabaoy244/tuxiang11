"""Per-stage timing probe: splits loader / forward / loss / backward / opt, prints peak VRAM
and a CUDA op table, so "which stage is slow and where is the second spent" is one run.

    python scripts/bench_gpu_stage.py --data-root /root/autodl-tmp/data/Datasets --limit 30
    python scripts/bench_gpu_stage.py --stage 1 --skip-loader --cudnn bench
    python scripts/bench_gpu_stage.py --device cpu --fallback-backbone --skip-loader --batch-size 1

Context: probes showed stage2 at ~1.44 s/iter vs stage1 at ~0.14 s/iter (same process, same
weights, same loader) and identically slow with cuDNN on / off / pure fp32. A local rebuild of
the very same graph (stage1 tasks vs stage2 tasks, same model, same batch) costs the same in
both (1.39 s vs 1.42 s per step, op table nearly identical), so stage2's extra layers
(localization head backward + bce/dice/edge) cannot account for a 10x gap. The gap must be a
runtime effect -> this probe separates the three candidates (loader / compute / memory) in one go.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import islice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from src.data.datasets import build_dataloaders  # noqa: E402
from src.losses.multi_task import MultiTaskLoss  # noqa: E402
from src.models.vibnet import build_model, load_config  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def loader_probe(cfg, limit):
    print("\n===== [1] dataloader only (no model) =====")
    t = time.perf_counter()
    ld = build_dataloaders(cfg)
    print(f"  build {time.perf_counter()-t:.1f}s  train={len(ld['train'].dataset)} "
          f"val={len(ld['val'].dataset)}")
    it = iter(ld["train"])
    t = time.perf_counter()
    next(it)
    print(f"  first batch (worker spawn) {time.perf_counter()-t:.2f}s")
    t, n = time.perf_counter(), 1
    for _ in islice(it, max(0, limit - 1)):
        n += 1
    dt = time.perf_counter() - t
    print(f"  steady {dt/max(1,n-1)*1000:.1f} ms/batch  ({n/max(dt,1e-9):.1f} batch/s) "
          f"-> this is the floor for any stage")


def stage_probe(cfg, model, crit, dev, idx, iters):
    st = cfg["train"]["stages"][idx]
    model.apply_stage(st["freeze"], st["train"])
    bs, sz = cfg["train"]["batch_size"], cfg["data"]["image_size"]
    x = torch.randn(bs, 3, sz, sz, device=dev)
    y = torch.randint(0, 2, (bs,), device=dev)
    mask = (torch.rand(bs, 1, sz, sz, device=dev) > 0.9).float()
    mv = torch.ones(bs, dtype=torch.bool, device=dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    dtype = getattr(torch, cfg["train"].get("amp_dtype", "bfloat16"))
    amp = bool(cfg["train"]["amp"]) and dev.type == "cuda"

    def step():
        opt.zero_grad(set_to_none=True)
        sync(dev); t = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=amp):
            out = model(x, sample_vib=True, beta=0.0)
            sync(dev); f = time.perf_counter() - t; t = time.perf_counter()
            loss = crit(out, y, mask, beta=0.0, active_tasks=st["use_tasks"],
                        update_norm=True, mask_valid=mv)["total"]
            sync(dev); l = time.perf_counter() - t; t = time.perf_counter()
        loss.backward()
        sync(dev); b = time.perf_counter() - t; t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sync(dev)
        return f, l, b, time.perf_counter() - t

    for _ in range(2):
        step()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    acc = [0.0] * 4
    t0 = time.perf_counter()
    for _ in range(iters):
        acc = [a + v for a, v in zip(acc, step())]
    wall = (time.perf_counter() - t0) / iters
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n===== [stage{idx+1}] {st['name']}  tasks={st['use_tasks']} =====")
    print(f"  trainable {ntr/1e6:.2f}M  amp={amp} dtype={cfg['train'].get('amp_dtype')}  bs={bs}")
    print(f"  forward {acc[0]/iters*1e3:8.1f} ms | loss {acc[1]/iters*1e3:6.1f} ms | "
          f"backward {acc[2]/iters*1e3:8.1f} ms | opt {acc[3]/iters*1e3:6.1f} ms "
          f"|| {wall*1e3:8.1f} ms/iter  ({1/wall:.2f} iter/s)")
    if dev.type == "cuda":
        print(f"  peak VRAM: allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB / "
              f"reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
    try:
        from torch.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        key = "cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
        with profile(activities=acts) as pr:
            step()
        print(pr.key_averages().table(sort_by=key, row_limit=12))
    except Exception as e:  # noqa: BLE001
        print(f"  [profiler skipped] {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/default.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--limit", type=int, default=30, help="batches for the loader probe")
    ap.add_argument("--iters", type=int, default=3, help="timed steps per stage")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--cudnn", default="on", choices=["on", "off", "bench"])
    ap.add_argument("--amp", default="on", choices=["on", "off"])
    ap.add_argument("--amp-dtype", default=None, choices=["bfloat16", "float16"])
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--skip-loader", action="store_true")
    ap.add_argument("--fallback-backbone", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.batch_size:
        cfg["train"]["batch_size"] = a.batch_size
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    cfg["train"]["amp"] = (a.amp == "on")
    if a.fallback_backbone:
        cfg["model"]["spatial"]["backbone"] = "tiny_vit"

    from scripts.train import apply_runtime_flags
    apply_runtime_flags(cfg, a.cudnn, a.amp_dtype)

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev} cudnn.enabled={torch.backends.cudnn.enabled} "
          f"benchmark={torch.backends.cudnn.benchmark} amp={cfg['train']['amp']} "
          f"amp_dtype={cfg['train'].get('amp_dtype')} channels_last={a.channels_last}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu={torch.cuda.get_device_name(0)} vram={p.total_memory/2**30:.1f}GiB "
              f"torch={torch.__version__} cuda={torch.version.cuda}")

    if not a.skip_loader:
        loader_probe(cfg, a.limit)

    model = build_model(cfg)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(dev)
    crit = MultiTaskLoss(cfg).to(dev)
    for i in ([a.stage] if a.stage is not None else list(range(len(cfg["train"]["stages"])))):
        stage_probe(cfg, model, crit, dev, i, a.iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
