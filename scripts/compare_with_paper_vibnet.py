"""把本项目实测结果与外源论文（CVPR 2025 VIB-Net）报告值对齐比较。

数据来源（全部可追）：
  论文侧  D:/QQ临时文件/Towards_Universal_AI-Generated_Image_Detection_by_Variational_
          Information_Bottleneck_Network.pdf  → Table 3 (AP) / Table 4 (ACC)，ProGAN 训练源
          （原文 5.3 节：训练用 ProGAN，测试 18（表内 17）个模型）
  我方侧  outputs/cross_gen_20260928/cross_generator_20260928.json，逐生成器 acc/auc/ap
          （权重 outputs/upload/results_20260928/_pack/vibnet_best.pt，
            训练源 ForenSynths/ProGAN + CASIAv2，测试 ForenSynths/test 13 生成器 ×150/150）

口径说明（已核官方代码）：
  论文 main.py: `ap = average_precision_score(y_true, y_pred)`（sklearn）
               `acc = accuracy_score(y_true, y_pred > 0.5)`
  我方 src/engine/metrics.py: `average_precision()` 同为「按不同分数值分组、
               AP = Σ(R_n − R_{n−1})·P_n」，与 sklearn 同定义 → 两个 AP 可并列。
  ⇒ 两边 AP 口径一致；ACC 阈值同为 0.5。**但训练源/训练量/骨干/输入协议不同**，
     所以本脚本只在「双方都报过的同一个生成器」上做逐项对照，不做总体并列。

用法：
  python scripts/compare_with_paper_vibnet.py            # 打印 + 写出 outputs/diag/
"""
from __future__ import annotations

import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS_JSON = os.path.join(ROOT, "outputs", "cross_gen_20260928", "cross_generator_20260928.json")
OUT_DIR = os.path.join(ROOT, "outputs", "diag")

# ---- 论文 Table 3 (AP, ProGAN 训练源) --------------------------------------
# 表头顺序：ProGAN CycleGAN BigGAN StyleGAN StarGAN GauGAN CRN IMLE Deepfake SAN
#           SDV1.4 SDV1.5 ADM GLIDE Midjourney Wukong VQDM
PAPER_HEADER = [
    "progan", "cyclegan", "biggan", "stylegan", "stargan", "gaugan", "crn", "imle",
    "deepfake", "san", "sdv1.4", "sdv1.5", "adm", "glide", "midjourney", "wukong", "vqdm",
]
PAPER_T3_AP = {  # Table 3, row "Ours"
    "progan": 100.00, "cyclegan": 99.80, "biggan": 99.29, "stylegan": 98.79,
    "stargan": 99.72, "gaugan": 99.99, "crn": 90.89, "imle": 96.83,
    "deepfake": 92.64, "san": 91.62, "sdv1.4": 87.24, "sdv1.5": 86.98,
    "adm": 87.88, "glide": 88.53, "midjourney": 75.68, "wukong": 90.92, "vqdm": 96.51,
}
PAPER_T3_AP_UNIVFD = {  # Table 3, row "Univfd"（论文的基线）
    "progan": 100.00, "cyclegan": 99.21, "biggan": 98.31, "stylegan": 97.98,
    "stargan": 99.35, "gaugan": 99.80, "crn": 96.72, "imle": 99.00,
    "deepfake": 82.04, "san": 82.18, "sdv1.4": 85.48, "sdv1.5": 82.30,
    "adm": 84.34, "glide": 84.04, "midjourney": 69.10, "wukong": 90.13, "vqdm": 94.96,
}
PAPER_T4_ACC = {  # Table 4, row "Ours"
    "progan": 99.99, "cyclegan": 99.00, "biggan": 95.75, "stylegan": 91.25,
    "stargan": 98.95, "gaugan": 99.70, "crn": 71.25, "imle": 86.75,
    "deepfake": 83.20, "san": 70.50, "sdv1.4": 71.55, "sdv1.5": 70.00,
    "adm": 71.45, "glide": 69.40, "midjourney": 61.25, "wukong": 75.90, "vqdm": 86.65,
}
PAPER_T4_ACC_UNIVFD = {
    "progan": 99.90, "cyclegan": 98.50, "biggan": 94.50, "stylegan": 84.40,
    "stargan": 95.85, "gaugan": 99.50, "crn": 59.50, "imle": 72.00,
    "deepfake": 67.40, "san": 56.50, "sdv1.4": 63.10, "sdv1.5": 63.57,
    "adm": 66.90, "glide": 61.70, "midjourney": 57.85, "wukong": 71.06, "vqdm": 85.00,
}
# 论文正文 5.3 声明的 Average 列（用于校验上表抄录是否完整）
PAPER_T3_AVG_CLAIM, PAPER_T4_AVG_CLAIM = 93.14, 82.50
PAPER_T3_AVG_UNIVFD, PAPER_T4_AVG_UNIVFD = 90.88, 76.31

# 我方与论文同名的生成器：在 main() 里与我方实际 key 求交后填入
OVERLAP_ALL: list[str] = []
# 论文表中属"未见过的生成器"（训练源是 ProGAN，故自身不算泛化域）
OVERLAP_UNSEEN: list[str] = []


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    d = json.load(io.open(OURS_JSON, encoding="utf-8"))
    pg = d["per_generator"]
    ours = {k.lower(): v for k, v in pg.items()}

    # 只在「论文报了 且 我方也测了」的生成器上做逐项对照
    global OVERLAP_ALL, OVERLAP_UNSEEN
    OVERLAP_ALL = [g for g in PAPER_HEADER if g in ours]
    OVERLAP_UNSEEN = [g for g in OVERLAP_ALL if g != "progan"]

    L = []
    L.append("# 我方实测 vs 外源论文 VIB-Net(CVPR2025) — 同生成器逐项对照")
    L.append("")
    L.append("> 由 `scripts/compare_with_paper_vibnet.py` 生成，全部数字可复算。")
    L.append(f"> 我方来源：`{os.path.relpath(OURS_JSON, ROOT)}`")
    L.append("> 论文来源：PDF Table 3 / Table 4 中 “Ours” 行（训练源 ProGAN，与我方同源）")
    L.append("")

    # ---- 抄录自校验：表内数字均值是否与论文声明的 Average 对上 ----
    L.append("## 0. 抄录自校验（防我抄错论文）")
    L.append("")
    chk = [
        ("Table 3 Ours", mean(PAPER_T3_AP.values()), PAPER_T3_AVG_CLAIM),
        ("Table 4 Ours", mean(PAPER_T4_ACC.values()), PAPER_T4_AVG_CLAIM),
        ("Table 3 Univfd", mean(PAPER_T3_AP_UNIVFD.values()), PAPER_T3_AVG_UNIVFD),
        ("Table 4 Univfd", mean(PAPER_T4_ACC_UNIVFD.values()), PAPER_T4_AVG_UNIVFD),
    ]
    L.append("| 行 | 我抄录 17 列之均值 | 论文声明的 Average | 差 |")
    L.append("|---|---|---|---|")
    for name, got, claim in chk:
        L.append(f"| {name} | {got:.2f} | {claim:.2f} | {got - claim:+.2f} |")
    L.append("")
    L.append("> 若「差」≈0，说明 17 列抄录完整；论文正文 5.3 称测试 18 个模型，"
             "而表内只有 17 列 —— 这是论文自身的不一致，下面按表内 17 列计算。")
    L.append("")

    # ---- 逐生成器对照 ----
    def table(gens, title):
        rows = []
        rows.append(f"## {title}")
        rows.append("")
        rows.append("| 生成器 | 论文 AP(Table3) | 我方 AP | Δ | 论文 ACC(Table4) | 我方 ACC | Δ | 我方 real_acc | 我方 fake_acc |")
        rows.append("|---|---|---|---|---|---|---|---|---|")
        for g in gens:
            pa, oa = PAPER_T3_AP[g], ours[g]["ap"] * 100
            pc, oc = PAPER_T4_ACC[g], ours[g]["acc"] * 100
            rows.append(
                f"| {g} | {pa:.2f} | {oa:.2f} | **{oa - pa:+.2f}** | "
                f"{pc:.2f} | {oc:.2f} | **{oc - pc:+.2f}** | "
                f"{ours[g]['real_acc']*100:.2f} | {ours[g]['fake_acc']*100:.2f} |"
            )
        rows.append(
            f"| **宏平均** | **{mean(PAPER_T3_AP[g] for g in gens):.2f}** | "
            f"**{mean(ours[g]['ap']*100 for g in gens):.2f}** | "
            f"**{mean(ours[g]['ap']*100 for g in gens) - mean(PAPER_T3_AP[g] for g in gens):+.2f}** | "
            f"**{mean(PAPER_T4_ACC[g] for g in gens):.2f}** | "
            f"**{mean(ours[g]['acc']*100 for g in gens):.2f}** | "
            f"**{mean(ours[g]['acc']*100 for g in gens) - mean(PAPER_T4_ACC[g] for g in gens):+.2f}** | "
            f"{mean(ours[g]['real_acc']*100 for g in gens):.2f} | "
            f"{mean(ours[g]['fake_acc']*100 for g in gens):.2f} |"
        )
        rows.append("")
        return rows

    L += table(OVERLAP_ALL, "1. 双方都测过的生成器（含训练源 progan）")
    L += table(OVERLAP_UNSEEN, "2. 只看「双方都测过、且都是未见生成器」的那几个 ★ 唯一干净口径")

    # ---- 论文自己定义的"跨系列泛化"口径：纯 GAN 列（不含 Deepfake/SAN）----
    PURE_GAN = ["cyclegan", "biggan", "stylegan", "stargan", "gaugan"]
    pure = [g for g in PURE_GAN if g in ours]
    L.append("## 2b. 按论文自己的「跨系列泛化」定义（纯 GAN 列，不含 Others）")
    L.append("")
    L.append("> 依据：论文摘要的 +12.48% AP / +23.59% ACC 只能由 Table 1/2 的"
             "**纯 GAN 那 6 列**（ProGAN/CycleGAN/BigGAN/StyleGAN/StarGAN/GauGAN）复算出来"
             "（本仓库已用 `scripts/audit_paper_deltas.py` 逐个命中）——"
             "Deepfake/SAN 在表里被归入 “Others”，不计入该声明。")
    L.append("")
    L.append("| 生成器 | 论文 AP | 我方 AP | Δ | 论文 ACC | 我方 ACC | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for g in pure:
        L.append(
            f"| {g} | {PAPER_T3_AP[g]:.2f} | {ours[g]['ap']*100:.2f} | "
            f"{ours[g]['ap']*100 - PAPER_T3_AP[g]:+.2f} | "
            f"{PAPER_T4_ACC[g]:.2f} | {ours[g]['acc']*100:.2f} | "
            f"{ours[g]['acc']*100 - PAPER_T4_ACC[g]:+.2f} |"
        )
    L.append(
        f"| **宏平均（{len(pure)} 个）** | "
        f"**{mean(PAPER_T3_AP[g] for g in pure):.2f}** | "
        f"**{mean(ours[g]['ap']*100 for g in pure):.2f}** | "
        f"**{mean(ours[g]['ap']*100 for g in pure) - mean(PAPER_T3_AP[g] for g in pure):+.2f}** | "
        f"**{mean(PAPER_T4_ACC[g] for g in pure):.2f}** | "
        f"**{mean(ours[g]['acc']*100 for g in pure):.2f}** | "
        f"**{mean(ours[g]['acc']*100 for g in pure) - mean(PAPER_T4_ACC[g] for g in pure):+.2f}** |"
    )
    L.append("")
    L.append(f"- 同口径下论文基线 Univfd：AP **{mean(PAPER_T3_AP_UNIVFD[g] for g in pure):.2f}**、"
             f"ACC **{mean(PAPER_T4_ACC_UNIVFD[g] for g in pure):.2f}**")
    L.append("")

    # ---- 只在这几个生成器上，我方 vs 论文基线 Univfd ----
    L.append("## 3. 同口径下我方 vs 论文自己的基线 Univfd")
    L.append("")
    L.append("| 生成器 | Univfd AP | 我方 AP | Δ | Univfd ACC | 我方 ACC | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for g in OVERLAP_UNSEEN:
        L.append(
            f"| {g} | {PAPER_T3_AP_UNIVFD[g]:.2f} | {ours[g]['ap']*100:.2f} | "
            f"{ours[g]['ap']*100 - PAPER_T3_AP_UNIVFD[g]:+.2f} | "
            f"{PAPER_T4_ACC_UNIVFD[g]:.2f} | {ours[g]['acc']*100:.2f} | "
            f"{ours[g]['acc']*100 - PAPER_T4_ACC_UNIVFD[g]:+.2f} |"
        )
    L.append(
        f"| **宏平均** | **{mean(PAPER_T3_AP_UNIVFD[g] for g in OVERLAP_UNSEEN):.2f}** | "
        f"**{mean(ours[g]['ap']*100 for g in OVERLAP_UNSEEN):.2f}** | "
        f"**{mean(ours[g]['ap']*100 for g in OVERLAP_UNSEEN) - mean(PAPER_T3_AP_UNIVFD[g] for g in OVERLAP_UNSEEN):+.2f}** | "
        f"**{mean(PAPER_T4_ACC_UNIVFD[g] for g in OVERLAP_UNSEEN):.2f}** | "
        f"**{mean(ours[g]['acc']*100 for g in OVERLAP_UNSEEN):.2f}** | "
        f"**{mean(ours[g]['acc']*100 for g in OVERLAP_UNSEEN) - mean(PAPER_T4_ACC_UNIVFD[g] for g in OVERLAP_UNSEEN):+.2f}** |"
    )
    L.append("")

    # ---- 我方全量 13 生成器 ----
    allg = sorted(ours)
    L.append("## 4. 我方全量 13 生成器（论文未覆盖其中的 seeingdark / stylegan2 / whichfaceisreal）")
    L.append("")
    L.append(f"- 宏平均 ACC **{mean(ours[g]['acc']*100 for g in allg):.2f}**、"
             f"宏平均 AP **{mean(ours[g]['ap']*100 for g in allg):.2f}**、"
             f"宏平均 AUC **{mean(ours[g]['auc']*100 for g in allg):.2f}**")
    L.append(f"- 去掉 progan（与训练同源）后：ACC **{mean(ours[g]['acc']*100 for g in allg if g!='progan'):.2f}**、"
             f"AP **{mean(ours[g]['ap']*100 for g in allg if g!='progan'):.2f}**")
    L.append(f"- 论文未测的 3 个："
             + "、".join(f"{g} ACC {ours[g]['acc']*100:.2f}/AP {ours[g]['ap']*100:.2f}"
                         for g in ["seeingdark", "stylegan2", "whichfaceisreal"]))
    L.append("")

    L.append("## 5. 论文侧覆盖但本方无数据的分支")
    L.append("")
    L.append("论文 Table 1/2（训练源 = Stable Diffusion v1.4，测 15 个模型）"
             "包含 SDV1.4/1.5、ADM、GLIDE、Midjourney、Wukong、VQDM 共 7 个扩散模型 —— "
             "**本项目本地没有任何扩散模型测试图，这一整块无法对照，也无法复现论文的主打结论。**")
    L.append("")

    txt = "\n".join(L)
    os.makedirs(OUT_DIR, exist_ok=True)
    p_md = os.path.join(OUT_DIR, "compare_paper_vibnet.md")
    io.open(p_md, "w", encoding="utf-8").write(txt)

    summary = {
        "overlap_all": OVERLAP_ALL,
        "overlap_unseen": OVERLAP_UNSEEN,
        "paper_T3_AP_ours_mean_all": round(mean(PAPER_T3_AP[g] for g in OVERLAP_ALL), 2),
        "paper_T4_ACC_ours_mean_all": round(mean(PAPER_T4_ACC[g] for g in OVERLAP_ALL), 2),
        "our_AP_mean_all": round(mean(ours[g]["ap"] * 100 for g in OVERLAP_ALL), 2),
        "our_ACC_mean_all": round(mean(ours[g]["acc"] * 100 for g in OVERLAP_ALL), 2),
        "paper_T3_AP_ours_mean_unseen": round(mean(PAPER_T3_AP[g] for g in OVERLAP_UNSEEN), 2),
        "paper_T4_ACC_ours_mean_unseen": round(mean(PAPER_T4_ACC[g] for g in OVERLAP_UNSEEN), 2),
        "our_AP_mean_unseen": round(mean(ours[g]["ap"] * 100 for g in OVERLAP_UNSEEN), 2),
        "our_ACC_mean_unseen": round(mean(ours[g]["acc"] * 100 for g in OVERLAP_UNSEEN), 2),
        "delta_AP_unseen": round(mean(ours[g]["ap"] * 100 for g in OVERLAP_UNSEEN)
                                 - mean(PAPER_T3_AP[g] for g in OVERLAP_UNSEEN), 2),
        "delta_ACC_unseen": round(mean(ours[g]["acc"] * 100 for g in OVERLAP_UNSEEN)
                                  - mean(PAPER_T4_ACC[g] for g in OVERLAP_UNSEEN), 2),
        "our_13gen_macro_ACC": round(mean(ours[g]["acc"] * 100 for g in allg), 2),
        "our_13gen_macro_AP": round(mean(ours[g]["ap"] * 100 for g in allg), 2),
        "our_13gen_macro_AUC": round(mean(ours[g]["auc"] * 100 for g in allg), 2),
        "our_13gen_macro_ACC_no_progan": round(
            mean(ours[g]["acc"] * 100 for g in allg if g != "progan"), 2),
        "our_13gen_macro_AP_no_progan": round(
            mean(ours[g]["ap"] * 100 for g in allg if g != "progan"), 2),
    }
    p_json = os.path.join(OUT_DIR, "compare_paper_vibnet.json")
    json.dump(summary, io.open(p_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(txt)
    print("\n[write]", p_md)
    print("[write]", p_json)


if __name__ == "__main__":
    main()
