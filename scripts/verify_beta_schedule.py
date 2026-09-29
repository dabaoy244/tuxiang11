"""零成本验算 β 退火轨迹 —— 不训练、不占卡，直接把每个全局轮次的 β 算出来。

为什么需要它（2026-09-29）：
    docs/22 §2.2(b) 记录的失效模式是「β 退火对任何东西都没有影响」：
    配置合法、日志里的 β 曲线也漂亮，但升温区间落在 VIB 被冻结、不进损失的轮次上，
    于是退火在语义上是空操作。修法有两个（换锚 / 调窗口），但**修好没有不能靠感觉**：

      * 跑一次短试点验证不了 —— 锚值到不了 warmup_start，日志只会打
        「β 全程恒为 0，退火未启用」，那是"没启用"，不是"修好了"。
      * 真正能验证退火的规模要 ~0.5 epochs-scale（约 4 小时 GPU）。

    可是这件事**本来就不需要 GPU**：β 只是 `(轮次, 阶段配置)` 的确定性函数。
    把 `trainer.py` 里那段循环照着算一遍，就能把每个轮次的 β 精确列出来，
    还能顺手把"旧锚（global）为什么会失效"对照出来。

    所以：先在这里把轨迹钉死，再去烧卡。训练日志里的 β 应该与这张表**逐行一致**；
    不一致就说明训练循环和这里的算法有分歧，那才是需要查的 bug。

用法：
    python scripts/verify_beta_schedule.py --config configs/default_crossval.yaml
    python scripts/verify_beta_schedule.py --config configs/default_crossval.yaml --scale 0.5
    python scripts/verify_beta_schedule.py --compare      # 同时列出 vib 锚 / global 锚
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.engine.trainer import StageConfig  # noqa: E402
from src.models.vib import beta_schedule  # noqa: E402


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate(stages, vib, anchor_mode: str, scale: float):
    """照抄 trainer.train() 里 253-290 行那段循环，产出逐轮的 (global, stage, anchor, beta)。

    刻意**不**调用 trainer 的方法：这里要的是"可独立复核的计算"，
    如果直接复用同一段代码，就变成"用自己的输出去验证自己"，没有交叉验证价值。
    但 beta_schedule 必须复用真实实现 —— 曲线本身不该有两份。
    """
    beta_max = float(vib.get("beta_max", 0.1))
    warm_s = float(vib.get("beta_warmup_start", 20))
    warm_e = float(vib.get("beta_warmup_end", 40))

    rows = []
    vib_ep = 0
    g = 0
    for st in stages:
        ep = int(round(st.epochs * scale)) or 1     # 至少 1 轮，否则短跑会空转
        for _ in range(ep):
            cls_on = "cls" in st.use_tasks
            anchor = (vib_ep if anchor_mode == "vib" else g) + st.beta_epoch_offset
            rows.append(dict(g=g, stage=st.name, anchor=anchor,
                             beta=beta_schedule(anchor, beta_max, warm_s, warm_e),
                             anchor_kind=anchor_mode, cls_on=cls_on))
            if cls_on:
                vib_ep += 1
            g += 1
    return rows, (beta_max, warm_s, warm_e)


def verdict(rows, warm_s, warm_e):
    """复刻 trainer._check_beta_window 的三个判据，把结论摆到台面上。"""
    warming = [r for r in rows if warm_s < r["anchor"] < warm_e]
    max_anchor = max(r["anchor"] for r in rows)
    if not warming:
        if max_anchor < warm_s:
            return "NOT_ENABLED", f"最大锚值 {max_anchor:g} < warmup_start {warm_s:g} ⇒ β 全程为 0（短跑的正常现象，但**不能**用它证明退火修好了）"
        return "SKIPPED", f"锚越过整个区间 [{warm_s:g},{warm_e:g}) 且无一轮落入 ⇒ **退火被跳过**（09-29 的原始症状）"
    overlap = [r for r in warming if r["cls_on"]]
    if not overlap:
        return "NO_OVERLAP", f"升温的 {len(warming)} 轮里 VIB 都不进损失 ⇒ 退火是空操作"
    return "OK", (f"升温 {len(warming)} 轮（global {warming[0]['g']}~{warming[-1]['g']}，"
                  f"阶段 {warming[0]['stage']}），其中 {len(overlap)} 轮 VIB 真的进损失 ⇒ 退火真实生效")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_crossval.yaml")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--compare", action="store_true", help="同时算 vib 锚与 global 锚")
    ap.add_argument("--chart", action="store_true", help="额外打印一行 β 曲线（每 5 轮一个点）")
    a = ap.parse_args()

    cfg = load(a.config)
    stages = [StageConfig.from_dict(d) for d in cfg["train"]["stages"]]
    vib = cfg.get("model", {}).get("vib", {}) or {}

    anchor = str(vib.get("beta_anchor", "vib")).lower()
    print("=" * 92)
    print(f"β 退火轨迹验算   配置={a.config}   epochs-scale={a.scale}   锚模式={anchor}")
    print(f"  beta_max={vib.get('beta_max')}  warmup=({vib.get('beta_warmup_start')}, "
          f"{vib.get('beta_warmup_end')})")
    print("  阶段：", "  ".join(f"{s.name}({int(round(s.epochs * a.scale)) or 1}ep,"
                                f"off={s.beta_epoch_offset},tasks={'/'.join(s.use_tasks)})"
                                for s in stages))
    print("=" * 92)

    modes = [anchor] + (["global"] if a.compare and anchor != "global" else [])
    for mode in modes:
        rows, (bmax, ws, we) = simulate(stages, vib, mode, a.scale)
        print(f"\n---- 锚 = {mode} ----")
        print(f"{'global':>6} {'阶段':<24} {'锚值':>5} {'β':>8}  VIB进损失")
        last_stage = None
        for r in rows:
            if r["stage"] != last_stage:
                print("  " + "-" * 74)
                last_stage = r["stage"]
            step = 0.01 if a.scale >= 1 else 0.02
            show = (r["g"] % (5 if a.scale >= 1 else 2) == 0) or r["beta"] != (
                rows[r["g"] - 1]["beta"] if r["g"] else -1)
            if show:
                print(f"{r['g']:>6} {r['stage']:<24} {r['anchor']:>5} {r['beta']:>8.4f}  "
                      f"{'是' if r['cls_on'] else '否'}")
        kind, msg = verdict(rows, ws, we)
        # 标签一律用 ASCII。实测 Windows 控制台字体（Consolas/宋体）里没有 U+2714/U+2718，
        # 会渲染成"豆腐块"方框；截图要给人看，这种小瑕疵会显得很业余。
        tag = {"OK": "[OK 生效]", "NOT_ENABLED": "[-- 未启用]",
               "SKIPPED": "[XX 被跳过]", "NO_OVERLAP": "[XX 无交集]"}[kind]
        print(f"\n  {tag} {msg}")

        if a.chart:
            pts = rows[::max(1, len(rows) // 20)]
            print("  β 曲线: " + " ".join(f"{r['beta']:.3f}" for r in pts))

    # 正式训练前的验收判据：锚=vib、scale=1 必须是 OK
    rows, (bmax, ws, we) = simulate(stages, vib, anchor, a.scale)
    kind, msg = verdict(rows, ws, we)

    # ---- 新旧对照（放在最末尾，截图时正好落在可见区）----
    old_rows, _ = simulate(stages, vib, "global", a.scale)
    print("\n" + "-" * 92)
    print("新旧对照：同一个全局轮次上的 β（旧 = global 锚，新 = vib 锚）")
    print(f"{'global':>6} {'阶段':<24} {'旧(global锚)':>13} {'新(vib锚)':>11}   说明")
    for g in (40, 42, 46, 50, 54, 60):
        if g >= len(rows):
            continue
        o, n = old_rows[g]["beta"], rows[g]["beta"]
        why = "stage3 一开就是满值 ⇐ 09-28 的实际行为" if o > n + 1e-9 else "一致"
        print(f"{g:>6} {rows[g]['stage']:<24} {o:>13.4f} {n:>11.4f}   {why}")
    n_bad = sum(1 for i in range(min(len(rows), len(old_rows)))
                if old_rows[i]["beta"] > rows[i]["beta"] + 1e-9)
    print(f"旧锚在 {n_bad} 个轮次上 β 偏高 —— 那些轮次的 KL 项被过早压满，退火形同虚设。")

    print("=" * 92)
    if kind == "OK":
        print("[结论] 正式训练配置下 β 退火**真实生效**，可以开跑。")
        print("       训练日志里每个 epoch 的 beta= 值应与上表逐行一致；不一致就是训练循环跑偏了。")
        return 0
    if kind == "NOT_ENABLED":
        print("[结论] 当前 scale 太小，看不到退火 —— 这是短跑的预期行为。")
        print("       要验证退火请加大 --scale（锚值需越过 beta_warmup_start）。")
        return 0
    print(f"[结论] ✘ {msg}")
    print("       先修配置再开跑：把各 stage 的 beta_epoch_offset 置 0，并保持 beta_anchor='vib'。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
