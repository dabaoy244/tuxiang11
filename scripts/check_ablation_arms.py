"""逐模块消融臂自检：确认每个 preset 真的改变了模型，而不是"空开关"。

为什么需要这个检查
------------------
`src/evaluation/ablation.py` 的 `ABLATION_PRESETS` 用**点号字符串路径**改写配置
（如 `"model.vib.use_vib"`）。这带来两种静默失效：

1. **路径失效**：配置结构调整后 `_set_nested` 会 KeyError（这个会报错，还好）。
2. **开关失效**：路径存在、值也改了，但 `forward` 里根本没读这个开关 ——
   于是消融臂照常训练、照常出报告，结果与 `full` 逐位相同。
   这类错误不报错、不告警，只在答辩时被问"为什么去掉模块指标一点没变"才暴露。

本脚本对每个 preset 做三件事，任何一件不符即 `exit 1`：

* **参数量**：与 full 的差额必须与"该模块的参数量"量级相符（不是 0）；
* **前向可用**：用固定随机输入能跑通（防止改坏结构直接崩溃）；
* **输出确实不同**：同一输入下 logits 与 full 不一致（证明开关真的作用到了计算图）。

用法
----
    PY=C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe
    "$PY" scripts/check_ablation_arms.py
    "$PY" scripts/check_ablation_arms.py --config configs/default_crossval.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.ablation import ABLATION_PRESETS, apply_preset   # noqa: E402
from src.models.vibnet import build_model, load_config                # noqa: E402

# 每个 preset "应该"影响的模块组（用于给出可读的参数量差解释）
EXPECT_MODULE = {
    "no_cross_attention": "fusion",
    "no_vib": "vib",
    "no_edge": "localization",
    "no_learnable_mask": "freq",
    "no_phase": "freq",
    "no_grad_stop": "localization",
}


def _build(cfg):
    r = build_model(cfg)
    return r[0] if isinstance(r, tuple) else r


def _module_params(model, group: str):
    mod = getattr(model, group, None)
    if mod is None:
        return None
    return sum(p.numel() for p in mod.parameters())


def _grad_flow_flag(model, x, beta: float = 0.0):
    """探测「定位支路拿到的那份特征是否带梯度」。

    `use_grad_stop` 是**无参数开关** —— 它只对特征做 `detach()`，既不改结构、
    也不改 eval 前向输出，所以"参数量"和"输出是否相同"两把尺子**都抓不到它**。
    靠这两条去判它，只会得到假失败（本脚本第一版就是这么误报的）。

    唯一可靠的判据是：**定位支路输入张量的 `requires_grad`**。
    开了 grad_stop 应为 False（梯度被截断），关了应为 True。
    ⚠ 这里**不能用 `torch.no_grad()`** —— 那会让所有张量的 requires_grad
      变成 False，把信号本身抹掉。
    """
    seen = {}

    def hook(mod, args):
        # register_forward_pre_hook 不带 with_kwargs 时回调签名就是 (module, args)；
        # 写成 (mod, args, kwargs) 会在每次前向时抛 TypeError。
        t = args[0] if args else None
        seen["requires_grad"] = bool(getattr(t, "requires_grad", False))

    h = model.localization.register_forward_pre_hook(hook)
    try:
        model(x, sample_vib=False, beta=beta)
    finally:
        h.remove()
    return seen.get("requires_grad")


def _summary(model) -> dict:
    return getattr(model, "summary", lambda: {})()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_crossval.yaml")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--presets", nargs="*", default=None,
                    help="默认检查除 full 外的全部臂")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    presets = args.presets or [p for p in ABLATION_PRESETS if p != "full"]

    torch.manual_seed(args.seed)
    x = torch.randn(2, 3, args.image_size, args.image_size)

    print("=" * 72)
    print(f"[check] 配置 {args.config}")
    print(f"[check] 输入 {tuple(x.shape)}，seed={args.seed}")
    print("=" * 72)

    with torch.no_grad():
        full = _build(base_cfg).eval()
        full_sum = _summary(full)
        full_out = full(x, sample_vib=False, beta=0.0)
        full_logits = full_out["cls_logits"].clone()
        full_total = full_sum.get("total(M)") or sum(p.numel() for p in full.parameters()) / 1e6
        print(f"full  总参数 {full_total:.3f} M   "
              f"cls_logits {tuple(full_logits.shape)}   "
              f"sum|logits|={float(full_logits.abs().sum()):.6f}")

    failures = []
    print("-" * 72)
    print(f"{'preset':<22}{'总参数(M)':>11}{'Δ参数(M)':>11}{'Δ模块(M)':>10}"
          f"{'输出':>7}{'定位支路带梯度':>15}")
    print("-" * 72)

    with torch.no_grad():
        full_grad_flow = _grad_flow_flag(full, x)

    for p in presets:
        try:
            cfg = apply_preset(base_cfg, p)
        except Exception as e:                                     # noqa: BLE001
            print(f"{p:<22}  ✗ apply_preset 失败：{e}")
            failures.append((p, f"apply_preset: {e}"))
            continue

        try:
            with torch.no_grad():
                m = _build(cfg).eval()
                out = m(x, sample_vib=False, beta=0.0)
                lg = out["cls_logits"]
                tot = _summary(m).get("total(M)") or sum(
                    pp.numel() for pp in m.parameters()) / 1e6
            grad_flow = _grad_flow_flag(m, x)
        except Exception as e:                                     # noqa: BLE001
            print(f"{p:<22}  ✗ 前向失败：{type(e).__name__}: {e}")
            failures.append((p, f"forward: {e}"))
            continue

        d_tot = tot - full_total
        grp = EXPECT_MODULE.get(p)
        d_grp = None
        if grp:
            a, b = _module_params(full, grp), _module_params(m, grp)
            if a is not None and b is not None:
                d_grp = (b - a) / 1e6

        same = bool(torch.allclose(lg, full_logits, atol=0.0, rtol=0.0))
        same_loose = bool(torch.allclose(lg, full_logits, atol=1e-6, rtol=1e-6))
        out_tag = "相同" if same else ("近似" if same_loose else "不同")
        gf_tag = {True: "是", False: "否", None: "?"}[grad_flow]

        print(f"{p:<22}{tot:>11.3f}{d_tot:>+11.3f}"
              f"{(d_grp if d_grp is not None else float('nan')):>+10.3f}"
              f"{out_tag:>7}{gf_tag:>15}")

        # 判据：三种签名里**至少要有一项**真的变了，否则该臂是空开关。
        #   no_grad_stop 专项：它唯一可观测的签名就是"定位支路是否还带梯度"，
        #   必须由 False 变成 True，否则这条臂等于什么都没做。
        params_changed = abs(d_tot) > 1e-9
        output_changed = not (same or same_loose)
        gradflow_changed = grad_flow != full_grad_flow
        if not (params_changed or output_changed or gradflow_changed):
            failures.append((p, "参数量、前向输出、定位支路梯度流三者都没变 —— 该开关没有作用到计算图"))
        if p == "no_grad_stop" and not gradflow_changed:
            failures.append((p, f"梯度流未改变（定位支路 requires_grad={grad_flow}，"
                                f"full={full_grad_flow}）—— grad_stop 没生效"))

    print("-" * 72)
    print(f"full 基准：总参数 {full_total:.3f} M，定位支路带梯度 = "
          f"{ {True: '是', False: '否'}.get(full_grad_flow, '?') }")
    if failures:
        print("[check] ✗ 失败项：")
        for p, why in failures:
            print(f"   - {p}: {why}")
        print("[check] 消融臂存在『空开关』，修好之前不要拿它出论文数据。")
        return 1
    print(f"[check] ✓ {len(presets)} 个消融臂全部真的改变了模型"
          f"（参数量 / 前向输出 / 梯度流 至少一项已变化）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
