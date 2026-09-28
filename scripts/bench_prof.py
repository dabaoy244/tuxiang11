"""Run the REAL training (scripts/train.py) under torch.profiler and print three op tables
+ memory stats. Every argument is forwarded verbatim to train.py.

    python scripts/bench_prof.py --data-root /root/autodl-tmp/data/Datasets \
        --start-stage stage2_loc_pretrain --limit-batches 20 --epochs-scale 0.05 --amp \
        --out-dir /root/autodl-tmp/probe_prof --tag probe_prof
    python scripts/bench_prof.py --demo --epochs-scale 0.05 --limit-batches 2 --device cpu

Why: the A/B/C probes (cuDNN on / off / pure fp32) all gave ~1.45 s/iter for stage2 while
stage1 needs only ~0.15 s/iter, and a local CPU rebuild of the same graph says the two stages
cost the SAME (1.39 s vs 1.42 s per step). So stage2 is not doing 10x the arithmetic -- some
single kernel is being executed on a pathological path. These tables name it (op + shape + ms).

Three tables, three different failure modes:
  ① device time  -> who is expensive, and at what resolution / channel count
  ② call count   -> "tens of thousands of tiny kernels" (launch-overhead bound)
  ③ self CPU time -> stuck on host (sync, Python, copies)

Pitfalls kept here on purpose:
  * "is there any device time in this table" must be decided by summing the WHOLE table,
    not by looking at the first sorted row: CPU-only runtime calls (e.g. cudaDeviceSynchronize)
    have zero device time yet can sort to the top by CPU time. An earlier version keyed off
    ka[0] and therefore printed the CPU table -- it looked like "no convolutions ran at all".
  * `input_shapes` only carries values with `key_averages(group_by_input_shape=True)`.
  * train.py ends with sys.exit(): catch SystemExit OUTSIDE the `with profile(...)` block.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402


def _shapes(k, width: int = 58) -> str:
    return " ".join(str(getattr(k, "input_shapes", ""))[:width].split())


def top_by(pr, key: str, n: int) -> str:
    ka = pr.key_averages(group_by_input_shape=True)
    if not len(ka):
        return "(empty profile)"
    rows = sorted(ka, key=lambda k: getattr(k, key, 0.0) or 0.0, reverse=True)[:n]
    out = ["%11s %7s %10s  %-52s  %s" % ("total", "calls", "percall", "op", "input shapes"),
           "-" * 126]
    for k in rows:
        tot = getattr(k, key, 0.0) or 0.0
        c = max(1, k.count)
        out.append("%11.2f %7d %10.4f  %-52s  %s"
                   % (tot / 1e3, k.count, tot / 1e3 / c, k.key[:52], _shapes(k)))
    return "\n".join(out)


def device_key(pr) -> str:
    dev = "device_time_total" if torch.cuda.is_available() else "self_device_time_total"
    total = sum((getattr(k, dev, 0.0) or 0.0) for k in pr.key_averages())
    return dev if total > 0 else "self_cpu_time_total"


def report(pr) -> None:
    dk = device_key(pr)
    print("\n" + "=" * 126)
    print("① 按设备耗时排序（ms）—— 谁最贵、跑在什么形状上      [key=%s]" % dk)
    print(top_by(pr, dk, 15))
    ka = pr.key_averages()
    tot = sum((getattr(k, dk, 0.0) or 0.0) for k in ka) / 1e3
    print("---- 全表合计 %.1f ms；把它和 train.py 打印的每 iter 耗时对一下差多少" % tot)
    print("\n② 按调用次数排序 —— 次数极多说明是 kernel 启动开销/逐元素循环")
    print(top_by(pr, "count", 12))
    print("\n③ 按自我 CPU 耗时排序（ms）—— 大说明卡在 host 侧（同步/搬运/Python）")
    print(top_by(pr, "self_cpu_time_total", 10))
    if torch.cuda.is_available():
        st = torch.cuda.memory_stats()
        print("\n④ 显存与分配器")
        print("   peak allocated %.2f GiB / peak reserved %.2f GiB"
              % (torch.cuda.max_memory_allocated() / 2 ** 30,
                 torch.cuda.max_memory_reserved() / 2 ** 30))
        for k in ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free"):
            print("   %-18s %s" % (k, st.get(k)))
        print("   ⚠ num_alloc_retries > 0 ⇒ 分配器反复重试（碎片/顶到天花板），"
              "这本身就是慢的直接来源，且与 dtype、cuDNN 都无关。")


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--prof"]
    sys.argv = [os.path.join(ROOT, "scripts", "train.py")] + argv

    dev_arg = None
    if "--device" in argv:
        i = argv.index("--device")
        if i + 1 < len(argv):
            dev_arg = argv[i + 1]
    on_cuda = torch.cuda.is_available() and dev_arg != "cpu"
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_cuda else [])
    print("[bench_prof] device=%s cwd=%s torch=%s"
          % ("cuda" if on_cuda else "cpu", os.getcwd(), torch.__version__))
    print("[bench_prof] 透传给 train.py：%s" % " ".join(argv))

    pr = profile(activities=acts, record_shapes=True)
    try:
        with pr:
            runpy.run_path(os.path.join(ROOT, "scripts", "train.py"), run_name="__main__")
    except SystemExit:
        pass                    # train.py 以 sys.exit(main()) 结尾，不能让它吞掉下面的打印
    except Exception as exc:    # noqa: BLE001
        print("[bench_prof] 训练中断：%s: %s" % (type(exc).__name__, exc))

    report(pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
