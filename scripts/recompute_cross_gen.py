"""从已存盘的逐样本分数**免推理重算**跨生成器指标。

为什么需要它
------------
`eval_cross_generator.py` 最贵的一步是推理（本机 CPU 上 2600 张约 9 分钟/臂）。
一旦指标实现被发现有错（本项目已发生过两次：`np.trapz` 被 NumPy 2.0 移除导致
整表算不出来；AP 的并列处理导致退化臂被算低），如果指标只能"跟着推理一起算"，
那就意味着**每次改一行度量代码都要重烧一小时**。

所以实跑时用 `--scores-out` 把 `(生成器, 标签, 分数)` 落盘，之后改指标只跑本脚本：
它调用 `eval_cross_generator` 里**同一段** `summarize()`，保证口径不会分叉。

用法
----
    # 1) 实跑时存分数
    python scripts/eval_cross_generator.py --ckpt ... --per-class 100 \
        --scores-out outputs/cross_gen_scores/pretrained.npz

    # 2) 改完指标实现后重算（秒级）
    python scripts/recompute_cross_gen.py --scores outputs/cross_gen_scores/pretrained.npz \
        --out outputs/run_ablation_eval/cross_gen_pretrained
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.eval_cross_generator import summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="实跑时用 --scores-out 存的 .npz")
    ap.add_argument("--out", required=True, help="重算结果的输出目录")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = np.load(args.scores, allow_pickle=True)
    names, y, probs = d["names"], d["y"], d["probs"]
    per_gen_meta = json.loads(str(d["per_gen_meta"])) if "per_gen_meta" in d else {}
    if not per_gen_meta:  # 兼容早期没有存 meta 的 npz
        for g in sorted(set(names.tolist())):
            sel = names == g
            per_gen_meta[str(g)] = {
                "n_real": int((y[sel] == 0).sum()), "n_fake": int((y[sel] == 1).sum()),
                "avail_real": -1, "avail_fake": -1,
            }
        print("⚠ 该 .npz 没存 per_gen_meta（可用张数会显示为 -1）")

    meta = {
        "config": str(d["ckpt"]) + "（由存盘分数重算）",
        "ckpt": str(d["ckpt"]),
        "ckpt_name": str(d["ckpt"]),
        "per_class": int(d["per_class"]) if "per_class" in d else -1,
        "seed": -1,
        "dedup_real": True,
        "n_total": int(len(y)),
        "seconds": float("nan"),
        "batch_size": -1,
        "note": "指标由已存盘分数重算，未重新推理",
    }
    per_gen, overall, md = summarize(names, y, probs, per_gen_meta, meta)
    print(md)

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    md_path = os.path.join(args.out, f"cross_generator{suffix}.md")
    js_path = os.path.join(args.out, f"cross_generator{suffix}.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "per_generator": per_gen, "overall": overall},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写出：\n  {md_path}\n  {js_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
