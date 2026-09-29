#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并「分轮跑」的消融结果，生成最终的差值表。

为什么必须用它
--------------
`src/evaluation/ablation.py` 一次调用只写一份汇总表（`outputs/ablation/ablation.json|.md`），
而且**每轮都覆盖**。更麻烦的是它算 Δvs full 的方式：`base = table.get("full", {})` ——
本轮只跑某几个 arm 时，`table` 里没有 full，它会拿 **0** 当基线，
安静地写出 `0 - 本项 ACC` 这种形如 `-0.65` 的假差值（现已改成写 null 并告警）。

消融为什么要分轮跑：一组训练约 6.6 小时，一次调用里串 3 组就是 20 小时，
中途任何一组崩掉，**前面已跑完组的汇总也一起丢掉**（结果只在全部 presets 循环结束后
才写入）。分轮跑把风险切成小块，代价就是汇总表要合并 —— 那就是本脚本的职责。

用法
----
    python scripts/merge_ablation.py                       # 默认读 outputs/ablation/_archive
    python scripts/merge_ablation.py --archive outputs/ablation/_archive --out outputs/ablation

输入：`--archive` 下形如 `<preset>_<时间戳>/ablation.json`（`ablation_chain.sh` 每组跑完归档一份）
输出：`<out>/ablation_merged.json`、`<out>/ablation_merged.md`

★ 报数口径提醒（不要漏）
------------------------
本脚本汇总的是 `evaluate_model` 在 **test 全集**上的**池化**指标。
test 里含与训练同源的 `progan`（ACC 恒 ~1.0），会把这个池化 ACC 抬高。
跨生成器泛化的**宏观**口径另由 `scripts/eval_cross_generator.py` 产出，报数时按两行给：
    · 标准 13 生成器（含 progan）
    · 严格 9 个（去掉 progan 与留出的 biggan/cyclegan/stargan）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

CLS_KEYS = ("acc", "precision", "recall", "f1", "auc", "ap")
LOC_KEYS = ("miou", "pixel_acc", "dice", "f1")

#: 报告里的固定顺序（先 full 后其它），不在表里的 preset 追加在后面
ORDER = ("full", "no_vib", "no_cross_attention", "no_edge",
         "no_learnable_mask", "no_phase", "no_grad_stop")


def _f4(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def _p4(v) -> str:
    return "—" if v is None else f"{v:+.4f}"


def load_archived(archive: str):
    """读归档目录，返回 (merged, origin, dupes, files)。"""
    files = sorted(glob.glob(os.path.join(archive, "*", "ablation.json")))
    merged: Dict[str, dict] = {}
    origin: Dict[str, str] = {}
    dupes: List[Tuple[str, str, str]] = []
    for fp in files:
        tag = os.path.basename(os.path.dirname(fp))
        try:
            with open(fp, encoding="utf-8") as f:
                table = json.load(f)
        except Exception as e:                                   # noqa: BLE001
            print(f"[!] 跳过无法解析的 {fp}: {e}")
            continue
        if not isinstance(table, dict):
            print(f"[!] 跳过格式异常的 {fp}（顶层不是 dict）")
            continue
        for preset, row in table.items():
            if not isinstance(row, dict):
                continue
            if preset not in merged:
                merged[preset] = row
                origin[preset] = tag
                continue
            # 重复出现的 preset：保留「指标更全」的那份（本地/云端各跑过一次时会出现）
            old_acc = ((merged[preset].get("cls") or {}).get("acc"))
            new_acc = ((row.get("cls") or {}).get("acc"))
            dupes.append((preset, origin[preset], tag))
            if old_acc is None and new_acc is not None:
                merged[preset] = row
                origin[preset] = tag
    return merged, origin, dupes, files


def build_rows(merged: Dict[str, dict], full_preset: str = "full"):
    base = merged.get(full_preset)
    if base is None:
        print(f"[!] 归档里没有 `{full_preset}` 行 ⇒ Δ 列无法计算，全部记为 —。\n"
              f"    ★ 绝不能拿 0 当基线，那会得到 -0.65 这类假差值。")
    base_acc = (base.get("cls") or {}).get("acc") if base else None
    base_miou = (base.get("loc") or {}).get("miou") if base else None

    order = [p for p in ORDER if p in merged] + \
            [p for p in merged if p not in ORDER]
    rows = []
    for p in order:
        row = merged[p]
        cls = {k: (row.get("cls") or {}).get(k) for k in CLS_KEYS}
        loc = {k: (row.get("loc") or {}).get(k) for k in LOC_KEYS}
        d_acc = d_miou = None
        if base_acc is not None and cls.get("acc") is not None:
            d_acc = round(base_acc - cls["acc"], 4)
        if base_miou is not None and loc.get("miou") is not None:
            d_miou = round(base_miou - loc["miou"], 4)
        rows.append({
            "preset": p,
            "config": row.get("config", ""),
            "cls": cls,
            "loc": loc,
            "perf": row.get("perf", {}),
            "delta_vs_full": {"acc": d_acc, "miou": d_miou},
            "source": None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="合并分轮跑的消融结果")
    ap.add_argument("--archive", default="outputs/ablation/_archive",
                    help="归档目录：<preset>_<时间戳>/ablation.json")
    ap.add_argument("--out", default="outputs/ablation",
                    help="输出目录（写 ablation_merged.json / .md）")
    ap.add_argument("--full-preset", default="full")
    a = ap.parse_args()

    if not os.path.isdir(a.archive):
        print(f"[ABORT] 归档目录不存在：{a.archive}")
        return 2

    merged, origin, dupes, files = load_archived(a.archive)
    print(f"[merge] 扫到 {len(files)} 份 ablation.json；合并出 {len(merged)} 个消融项："
          f"{sorted(merged)}")
    if dupes:
        print(f"[merge] 重复项 {len(dupes)} 个（保留指标更全的那份）：{dupes[:6]}")
    if not merged:
        print(f"[ABORT] {a.archive} 下没有可用的 */ablation.json")
        return 2

    rows = build_rows(merged, a.full_preset)
    for r in rows:
        r["source"] = origin.get(r["preset"], "")

    missing = [p for p in ORDER if p not in merged]
    if missing:
        print(f"[merge] 尚未跑完的消融项（表里缺失）：{missing}")
        print(f"        报了消融结论就必须说明覆盖范围，别把「没跑」写成「无效」。")

    os.makedirs(a.out, exist_ok=True)
    js = os.path.join(a.out, "ablation_merged.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({r["preset"]: r for r in rows}, f,
                  ensure_ascii=False, indent=2, default=float)

    md = os.path.join(a.out, "ablation_merged.md")
    lines = ["| 消融项 | 说明 | ACC | F1 | AUC | mIoU | Dice | ΔACC | ΔmIoU | 归档 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, loc, dl = r["cls"], r["loc"], r["delta_vs_full"]
        lines.append(
            f"| {r['preset']} | {r['config']} | {_f4(c.get('acc'))} | {_f4(c.get('f1'))} | "
            f"{_f4(c.get('auc'))} | {_f4(loc.get('miou'))} | {_f4(loc.get('dice'))} | "
            f"{_p4(dl.get('acc'))} | {_p4(dl.get('miou'))} | {r['source']} |")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[merge] 已写入 {js}\n        Markdown：{md}")
    print("[merge] ★ test 为池化口径（含同源 progan，会抬高 ACC）；"
          "跨生成器宏平均请用 scripts/eval_cross_generator.py 另出一行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
