"""复算 CVPR2025 VIB-Net 论文摘要里 6 个「提升幅度」声明的出处。

动机：论文摘要给了 6 个百分数（+5.55 / +9.33 / +12.48 / +23.59 / +20.22 / +34.17 /
+20.46 / +26.41），但没说是在哪几列上平均的。若不把口径钉死，
引用这些数字时无法判断该拿哪一组我们的结果去对。

做法：把 Table 1（AP）/ Table 2（ACC）逐列抄进脚本，穷举 4 种列子集 × 2 个基线，
看哪个组合能命中论文声明的值。

结论（运行输出可见）：
  全部 15 列均值         → +20.22(AP vs Univfd) / +5.55(AP vs NPR)
                           +20.46(ACC vs Univfd) / +9.33(ACC vs NPR)
  纯 GAN 6 列（不含 Deepfake/SAN，表中归入 "Others"）
                         → +34.17(AP vs Univfd) / +12.48(AP vs NPR)
                           +26.41(ACC vs Univfd) / +23.59(ACC vs NPR)
  ⇒ 6 个声明全部精确命中。论文数据自洽；同时反推出论文「跨生成模型系列泛化」
     的口径 = 纯 GAN 那 6 列，**不含 Deepfake / SAN**。

用法：python scripts/audit_paper_deltas.py
"""
from __future__ import annotations

COLS_ALL = ["SDV1.4", "SDV1.5", "ADM", "GLIDE", "Midjourney", "Wukong", "VQDM",
            "ProGAN", "CycleGAN", "BigGAN", "StyleGAN", "StarGAN", "GauGAN",
            "Deepfake", "SAN"]

# Table 1 —— AP，训练源 Stable Diffusion v1.4
T1 = {
    "CNNSpot": [99.98, 99.83, 51.10, 58.80, 67.93, 99.80, 49.92, 53.15, 50.23, 49.79, 55.98, 47.07, 56.08, 54.86, 54.03],
    "Fusing":  [99.90, 97.98, 69.30, 94.20, 81.20, 99.90, 84.60, 67.63, 87.79, 69.37, 67.90, 91.20, 43.08, 72.56, 89.42],
    "Lgrad":   [99.94, 99.92, 58.52, 84.00, 91.06, 99.72, 56.34, 83.59, 90.24, 47.51, 82.74, 99.19, 49.25, 66.49, 65.09],
    "Univfd":  [96.04, 96.26, 66.34, 93.73, 92.08, 90.98, 74.53, 51.77, 63.42, 75.81, 54.12, 54.93, 65.99, 70.24, 83.34],
    "NPR":     [100.00, 99.97, 94.70, 95.80, 95.50, 100.00, 86.30, 83.30, 94.90, 72.00, 82.70, 97.30, 66.00, 85.30, 95.90],
    "CLIPping":[93.97, 93.10, 68.00, 87.44, 77.34, 86.52, 77.17, 88.54, 88.44, 85.33, 77.23, 89.82, 81.56, 61.19, 57.37],
    "Ours":    [100.00, 99.97, 95.49, 97.13, 97.81, 99.93, 97.00, 96.59, 98.44, 97.17, 84.31, 97.60, 96.94, 81.32, 93.27],
}
# Table 2 —— ACC，训练源 Stable Diffusion v1.4，阈值固定 0.5
T2 = {
    "CNNSpot": [99.48, 99.35, 50.10, 50.90, 56.42, 97.90, 50.04, 50.27, 49.81, 50.10, 50.98, 49.77, 50.38, 51.98, 50.22],
    "Fusing":  [99.90, 99.91, 51.30, 57.50, 52.30, 99.90, 64.20, 51.20, 52.40, 53.50, 50.20, 58.20, 49.32, 51.02, 64.84],
    "Lgrad":   [99.12, 99.05, 53.00, 64.24, 76.34, 97.53, 50.93, 61.61, 60.74, 48.82, 61.43, 50.17, 49.70, 50.17, 56.49],
    "Univfd":  [83.55, 84.80, 53.35, 75.30, 71.60, 73.55, 55.10, 58.65, 59.30, 61.45, 56.80, 61.45, 55.30, 58.40, 72.00],
    "NPR":     [100.00, 99.90, 73.00, 89.70, 82.30, 100.00, 68.30, 60.30, 67.20, 59.20, 58.00, 73.20, 52.00, 74.80, 89.60],
    "CLIPping":[96.07, 95.48, 70.14, 85.00, 77.66, 88.86, 79.35, 88.78, 88.48, 89.57, 80.69, 90.82, 85.94, 66.91, 61.64],
    "Ours":    [99.55, 99.20, 73.85, 74.25, 88.05, 98.25, 89.35, 89.70, 88.60, 91.20, 74.10, 80.70, 87.15, 72.00, 81.50],
}

IDX_ALL = set(range(15))
IDX_DIFF = set(range(0, 7))                              # 7 列扩散模型
IDX_GAN_ALL = set(range(7, 15))                           # 8 列 = GAN(6) + Others(2)
IDX_GAN_PURE = {7, 8, 9, 10, 11, 12}                      # 6 列纯 GAN（论文跨系列口径）
SUBSETS = [(IDX_ALL, "全部 15 列"), (IDX_DIFF, "扩散 7 列"),
           (IDX_GAN_ALL, "GAN+Others 8 列"), (IDX_GAN_PURE, "★纯 GAN 6 列")]

# 论文声明的幅度：{指标: {基线: [目标值...]}}
CLAIMS = {
    "AP":  {"Univfd": [20.22, 34.17], "NPR": [5.55, 12.48]},
    "ACC": {"Univfd": [20.46, 26.41], "NPR": [9.33, 23.59]},
}


def mean(vals, idx):
    return sum(vals[i] for i in idx) / len(idx)


def main():
    ok = miss = 0
    for metric, table in (("AP", T1), ("ACC", T2)):
        print("#" * 10, f"Table {'1' if metric == 'AP' else '2'} ({metric}, 训练源 SDv1.4)")
        for base in ("Univfd", "NPR"):
            for idx, sname in SUBSETS:
                d = mean(table["Ours"], idx) - mean(table[base], idx)
                hits = [t for t in CLAIMS[metric][base] if abs(d - t) < 0.35]
                tag = ""
                if hits:
                    tag = f"   <== 命中论文声明 {hits}"
                    ok += 1
                print(f"  Ours − {base:<7} {sname:<14} = {d:+7.2f}{tag}")
        u = mean(table["Ours"], IDX_ALL)
        print(f"  参考均值：Ours {u:.2f} / Univfd {mean(table['Univfd'], IDX_ALL):.2f}"
              f" / NPR {mean(table['NPR'], IDX_ALL):.2f}")
        print()
    total = sum(len(v) for v in CLAIMS.values())
    for m in CLAIMS.values():
        for v in m.values():
            miss += len(v)
    print(f"[结果] 论文声明共 {miss} 条，脚本命中 {ok} 条"
          f"{'（全部命中，论文数据自洽）' if ok == miss else f'，未命中 {miss - ok} 条需人工核对'}")
    print("[结论] 论文「跨生成模型系列泛化」的口径 = 纯 GAN 6 列，不含 Deepfake / SAN。")


if __name__ == "__main__":
    main()
