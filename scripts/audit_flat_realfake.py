"""审计「扁平结构」的真实/伪造图像对（无目录层级、无掩码）。

适用场景：外部同学/合作者给来的一对文件夹，形如
    <fake_dir>/000_sdv4_00000.png      伪造（整图生成）
    <real_dir>/n01440764_10043.JPEG    真实

本脚本做四件事，全部只读、不改动任何数据：
  1. 计数与命名解析：每个类别/前缀各多少张，是否有命名异常
  2. 类别集合关系：两边的类别集合交集 / 各自独有
     —— 这决定了能否做「内容可控」的真假对照
  3. 平凡可分线索（★ 最重要），分两组：
     3a **文件级**：分辨率、压缩率（每像素字节）、格式、JPEG 块伪影
        —— 模型评测时只拿到解码后的像素，看不到这些，但**人做数据清洗时必须看**，
           因为「PNG vs JPEG」这种差异会通过频域支路间接泄漏给模型。
     3b **像素级**：高频能量占比、径向功率谱斜率、Laplacian 方差、饱和度
        —— 模型能看到的只有像素，所以**这组才是真正决定成败的一关**。
     两组都给出单特征 AUC。**只要有一个特征 AUC 接近 1，就说明不用学任何
     篡改痕迹也能把这批数据分开** —— 这种数据直接进训练/评测会得到虚高指标。
  4. 建议的清洗动作（打印，不执行）

用法：
    python scripts/audit_flat_realfake.py \
        --fake-dir  "D:/data/smallai/smallai" \
        --real-dir  "D:/data/smallnature/smallnature" \
        --fake-class-regex '^(?P<cls>\\d+)_' \
        --real-class-regex '^(?P<cls>n\\d+)_' \
        --sample 600

    # 可选：给出 ImageNet 索引->synset 映射后，直接算两边的类别交集
    #   --index-map imagenet_index_synset.txt   （每行 "793 n04209133"）

    # 只跑本脚本自己的口径自检（不读数据），已挂进 scripts/smoke_test.py
    python scripts/audit_flat_realfake.py --self-test
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("需要 Pillow：pip install pillow", file=sys.stderr)
    raise

IMG_EXT = (".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".webp")
BAR = "=" * 74


# ---------------------------------------------------------------- 基础工具
def list_images(d: str) -> List[str]:
    if not os.path.isdir(d):
        raise SystemExit(f"目录不存在：{d}")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def parse_class(name: str, pattern: Optional[str]) -> Optional[str]:
    """按正则从文件名提取类别键。未给 pattern 时退化为「下划线前第一段」。"""
    if pattern:
        m = re.match(pattern, name)
        return m.group("cls") if m and "cls" in (m.groupdict() or {}) else (m.group(1) if m else None)
    return name.split("_")[0]


def auc(score, label) -> float:
    """并列分数取平均秩的 AUC（不依赖 sklearn）。"""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    sp = s[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def blockiness(gray: np.ndarray) -> float:
    """JPEG 8x8 网格伪影强度 = 块边界处的水平梯度 / 块内部的水平梯度。

    接近 1 表示没有明显块效应（未 JPEG 压缩或已重采样）；
    明显 > 1 表示存在 8 像素周期的量化台阶。
    """
    g = np.abs(np.diff(gray.astype(np.float32), axis=1))
    xb = np.arange(8, g.shape[1], 8) - 1
    if len(xb) == 0:
        return float("nan")
    mask = np.ones(g.shape[1], dtype=bool)
    mask[xb] = False
    on = g[:, xb].mean() if g[:, xb].size else 0.0
    off = g[:, mask].mean() if g[:, mask].size else 1e-6
    return float(on / (off + 1e-6))


def pixel_feats(gray: np.ndarray, rgb: np.ndarray) -> Dict[str, float]:
    """从**解码后的像素**里抽与语义无关的内容统计量。

    为什么必须单独看这一组
    ----------------------
    上面的 `fmt` / `bpp` / `px` 是**文件级**线索：模型评测时只拿到解码后的像素
    （本项目的 pipeline 是 `_read_image()` → resize 224 → 归一化），**根本看不到
    文件大小和扩展名**。所以「统一格式/分辨率」之后，即使 bpp 还有残余可分性，
    也不等于模型能利用它。真正决定成败的是：**像素本身有没有平凡可分性**。

    这里抽的都是「不涉及语义、不看标签」的低级统计量：
      hf     高频能量占比（Laplacian 能量 / 总能量）—— 越糊越小
      slope  径向功率谱斜率（log 幅度 ~ log 频率 的拟合斜率）—— 扩散模型输出的
             频谱衰减与真实照片不同，这是文献里最常用的一条「生成痕迹」线索，
             **但它检验的是「是不是 AI 生成」，不是「有没有局部篡改」**
      lapvar Laplacian 方差
      sat    饱和度均值
    """
    g = gray.astype(np.float32) / 255.0
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    energy = float((g ** 2).mean()) + 1e-12
    hf = float((lap ** 2).mean() / energy)

    f = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    nb = min(cy, cx)
    radial = np.bincount(r.ravel(), weights=(f ** 2).ravel(), minlength=nb + 1)[1:nb + 1]
    cnt = np.maximum(np.bincount(r.ravel(), minlength=nb + 1)[1:nb + 1], 1)
    radial = radial / cnt
    k = np.arange(1, nb + 1)
    ok = radial > 0
    if ok.sum() > 8:
        slope = float(np.polyfit(np.log(k[ok]), np.log(radial[ok]), 1)[0])
    else:
        slope = float("nan")

    rgbf = rgb.astype(np.float32)
    mx, mn = rgbf.max(2), rgbf.min(2)
    sat = float((mx - mn).mean() / 255.0)
    return {"hf": hf, "slope": slope, "lapvar": float(lap.var()), "sat": sat}


def probe(d: str, n: int, tag: str, pattern: Optional[str],
          want_pixel: bool = False) -> List[dict]:
    """抽样读取图像，抽取与内容无关的元信息特征（可选：像素级内容统计）。"""
    fs = list_images(d)
    rng = random.Random(3407)
    rng.shuffle(fs)
    fs = fs[:n]
    rows = []
    for f in fs:
        p = os.path.join(d, f)
        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = (im.format or "?").upper()
                gray = np.asarray(im.convert("L").resize((256, 256), Image.BILINEAR))
                rgb = None
                if want_pixel:
                    rgb = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR))
        except Exception as exc:                        # 坏图要报出来，不要吞掉
            rows.append({"file": f, "err": str(exc)[:60]})
            continue
        nb = os.path.getsize(p)
        rec = {
            "file": f, "tag": tag, "cls": parse_class(f, pattern),
            "w": w, "h": h, "px": w * h, "bytes": nb,
            "bpp": nb / max(w * h, 1),
            "blk": blockiness(gray),
            "fmt": fmt, "is_png": 1.0 if fmt == "PNG" else 0.0,
        }
        if rgb is not None:
            rec.update(pixel_feats(gray, rgb))
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- 自检
def self_test() -> int:
    """把审计脚本自己的口径钉死（可回归，挂进 scripts/smoke_test.py）。

    为什么需要：审计脚本给出的结论直接决定「这份数据能不能用」。
    如果它自己的 `auc` 算错（例如并列分数不做秩平均 → 全同分返回 0.0 而不是 0.5），
    或者 `pixel_feats` 的某个量恒为常数，**审计会照常打印结论**，
    只是把「干净的数据」判成「有捷径」或反之 —— 不抛异常，没人会发现。
    """
    fails = []

    def chk(name, got, want, tol=1e-9):
        ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
        print(f"    {'✅' if ok else '❌'} {name}: {got}")
        if not ok:
            fails.append(f"{name}: got={got} want={want}")

    print("  [自检] auc 的边界与并列处理")
    chk("完美分开 → 1.0", auc([0.1, 0.9], [0, 1]), 1.0)
    chk("完全反向 → 0.0", auc([0.9, 0.1], [0, 1]), 0.0)
    chk("全同分 → 0.5（秩平均）", auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)
    chk("并列但有区分 → 0.875（手算：3.5/4）",
        auc([0.5, 0.5, 0.9, 0.1], [1, 0, 1, 0]), 0.875)
    chk("单类缺失 → nan", float(np.isnan(auc([0.1, 0.2], [1, 1]))), 1.0)

    print("  [自检] blockiness：8 像素周期台阶应明显 > 1，平滑图应 ≈ 1")
    step = np.tile((np.arange(64) // 8) % 2, (64, 1)).astype(np.uint8) * 200 + 20
    smooth = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 2 + 20
    chk("周期台阶 > 1.2", blockiness(step) > 1.2, True)
    chk("平滑图 < 1.2", blockiness(smooth) < 1.2, True)

    print("  [自检] pixel_feats：噪声 vs 强模糊，各量必须单调且有限")
    rng = np.random.RandomState(3407)
    noise = rng.randint(0, 256, (256, 256)).astype(np.uint8)
    blur = np.asarray(Image.fromarray(noise).resize((32, 32), Image.BILINEAR)
                      .resize((256, 256), Image.BILINEAR))
    fn = pixel_feats(noise, np.stack([noise] * 3, -1))
    fb = pixel_feats(blur, np.stack([blur] * 3, -1))
    chk("噪声 hf > 模糊 hf", fn["hf"] > fb["hf"], True)
    chk("噪声 lapvar > 模糊 lapvar", fn["lapvar"] > fb["lapvar"], True)
    # 谱斜率：白噪声近似平坦（≈0），低通后明显转负（实测约 -3）。
    # ⚠ 不要断言"噪声的 slope 一定为负" —— 量化噪声会让它微微为正（实测 +0.031）。
    chk("模糊图 slope 比噪声图低 1.0 以上",
        np.isfinite(fn["slope"]) and np.isfinite(fb["slope"])
        and fb["slope"] < fn["slope"] - 1.0, True)
    chk("饱和度落在 [0,1]", 0.0 <= fn["sat"] <= 1.0, True)

    print("  [自检] parse_class：补零的数字前缀必须原样保留")
    chk("'000_sdv4_00000.png'", parse_class("000_sdv4_00000.png", r"^(?P<cls>\d+)_"), "000")
    chk("'n01440764_10043.JPEG'",
        parse_class("n01440764_10043.JPEG", r"^(?P<cls>n\d+)_"), "n01440764")
    chk("不匹配 → None", parse_class("weird.jpg", r"^(?P<cls>n\d+)_"), None)

    if fails:
        print(f"\n  ❌ 自检失败 {len(fails)} 项：{fails}")
        return 1
    print("\n  ✅ 审计脚本自检全部通过。")
    return 0


# ---------------------------------------------------------------- 报告
def sec(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def main() -> int:
    ap = argparse.ArgumentParser(description="审计扁平结构的外部真实/伪造图像对")
    ap.add_argument("--fake-dir", default=None, help="伪造（AI 生成）图像目录")
    ap.add_argument("--real-dir", default=None, help="真实图像目录")
    ap.add_argument("--self-test", action="store_true",
                    help="只跑本脚本自己的口径自检（可回归，挂进 smoke_test.py），不读数据")
    ap.add_argument("--fake-class-regex", default=None,
                    help=r"从伪造文件名提取类别的正则，需含命名组 cls，如 '^(?P<cls>\d+)_'")
    ap.add_argument("--real-class-regex", default=None,
                    help=r"从真实文件名提取类别的正则，如 '^(?P<cls>n\d+)_'")
    ap.add_argument("--index-map", default=None,
                    help="可选：`索引 synset` 映射文件（每行一条），用于把数字前缀翻成 synset")
    ap.add_argument("--sample", type=int, default=600, help="每侧抽样张数（默认 600）")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.self_test:
        print(f"\n{BAR}\n审计脚本自检（不读数据）\n{BAR}")
        return self_test()
    if not args.fake_dir or not args.real_dir:
        ap.error("--fake-dir 与 --real-dir 为必填（或用 --self-test 只跑自检）")

    # 1) 计数
    sec("1. 计数与命名解析")
    counts: Dict[str, Dict[str, int]] = {}
    names: Dict[str, Dict[str, List[str]]] = {}
    for tag, d, pat in (("fake", args.fake_dir, args.fake_class_regex),
                        ("real", args.real_dir, args.real_class_regex)):
        fs = list_images(d)
        c: Dict[str, int] = {}
        nm: Dict[str, List[str]] = {}
        unparsed = 0
        for f in fs:
            k = parse_class(f, pat)
            if k is None:
                unparsed += 1
                continue
            c[k] = c.get(k, 0) + 1
            nm.setdefault(k, []).append(f)
        counts[tag], names[tag] = c, nm
        print(f"  [{tag}] {d}")
        print(f"        文件 {len(fs)} 张，类别/前缀 {len(c)} 个，无法解析 {unparsed} 张")
        print(f"        每类张数：最多 {max(c.values())}，最少 {min(c.values())}，"
              f"中位 {int(np.median(list(c.values())))}")
        if unparsed:
            print("        ⚠ 有文件不符合命名规则，会被下游脚本漏掉")

    # 2) 类别集合关系
    sec("2. 类别集合关系（决定能否做内容可控的真假对照）")
    idx2syn: Dict[int, str] = {}
    if args.index_map and os.path.exists(args.index_map):
        for line in open(args.index_map, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                idx2syn[int(parts[0])] = parts[1]

    fa, ra = set(counts["fake"]), set(counts["real"])
    if idx2syn:
        # 把两边的类别键都归一化到 synset 再比较。
        # ⚠ 数字前缀可能是补零的（"000"），必须先转 int 再查表 —— 否则
        #   `counts.get("0")` 会查不到 "000"，把张数悄悄算成 0。
        def to_syn(counter: Dict[str, int]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for k, v in counter.items():
                s = idx2syn.get(int(k)) if k.isdigit() else k
                if s:
                    out[s] = out.get(s, 0) + v
            return out

        fs_, rs_ = to_syn(counts["fake"]), to_syn(counts["real"])
        inter = sorted(set(fs_) & set(rs_))
        print(f"  已加载索引映射 {len(idx2syn)} 条（数字前缀已按索引翻成 synset 后比较）")
        print(f"  交集（两边类别一致，可做内容可控对照）：{len(inter)} 个")
        print(f"  仅伪造侧有：{len(set(fs_) - set(rs_))} 个    "
              f"仅真实侧有：{len(set(rs_) - set(fs_))} 个")
        overlap = len(inter) / max(min(len(fs_), len(rs_)), 1)
        pf = sum(fs_[s] for s in inter)
        pr = sum(rs_[s] for s in inter)
        pairs = sum(min(fs_[s], rs_[s]) for s in inter)      # ★ 逐类取 min 再求和
        if inter:
            print(f"\n  交集类别明细（按 1:1 配对后可用于「内容可控」对照）：")
            print(f"    {'synset':>12} {'伪造':>7} {'真实':>7} {'可配对':>7}")
            for s in inter:
                print(f"    {s:>12} {fs_[s]:>7} {rs_[s]:>7} {min(fs_[s], rs_[s]):>7}")
        print(f"\n  交集类别合计：伪造 {pf} 张 / 真实 {pr} 张；严格 1:1 可配 {pairs} 对"
              f"（{pairs * 2} 张）")
        print(f"  重合度：占伪造侧 {overlap * 100:.0f}%、占真实侧 "
              f"{len(inter) / max(len(rs_), 1) * 100:.0f}%")
        if overlap < 0.5:
            print(f"  ⚠ 两类类别集合重合度低 —— 大部分类别只出现在一侧。")
            print(f"     此时「所有该类样本都是同一标签」，模型可能把**类别身份**"
                  f"当成真假线索（捷径学习），跨数据集泛化会被高估。")
            print(f"     建议：只保留交集类别做内容可控对照，或显式声明这一限制。")
    else:
        inter = sorted(fa & ra)
        print(f"  按原始键直接比较（未提供 --index-map，数字前缀与 synset 无法对齐）")
        print(f"  交集 {len(inter)} 个；仅伪造侧 {len(fa - ra)} 个；仅真实侧 {len(ra - fa)} 个")
        if not inter:
            print("  ⚠ 交集为 0 —— 两侧命名体系不同，必须先提供 --index-map 才能比较。")
    if len(fa) and len(ra):
        print(f"  ⚠ 若两侧类别基本不重合，模型可能把「类别身份」当成真假线索（捷径学习）。")

    # 3) 平凡可分线索
    sec("3. ★ 平凡可分线索（3a 文件级 / 3b 像素级；都不涉及「篡改痕迹」）")
    rows = (probe(args.fake_dir, args.sample, "fake", args.fake_class_regex, True)
            + probe(args.real_dir, args.sample, "real", args.real_class_regex, True))
    bad = [r for r in rows if "err" in r]
    rows = [r for r in rows if "err" not in r]
    if bad:
        print(f"  ❌ 有 {len(bad)} 张打不开（前 3 个）：{[(b['file'], b['err']) for b in bad[:3]]}")
    print(f"  抽样：伪造 {sum(1 for r in rows if r['tag']=='fake')} 张 / "
          f"真实 {sum(1 for r in rows if r['tag']=='real')} 张\n")

    feats = [("px", "原图分辨率(像素数)"), ("bpp", "每像素字节数(压缩率)"),
             ("blk", "JPEG 块伪影比"), ("is_png", "是否 PNG")]
    px_feats = [("hf", "高频能量占比"), ("slope", "径向功率谱斜率"),
                ("lapvar", "Laplacian 方差"), ("sat", "饱和度均值")]
    for tag, label in (("real", "真实"), ("fake", "伪造(AI)")):
        s = [r for r in rows if r["tag"] == tag]
        if not s:
            continue
        fmts: Dict[str, int] = {}
        for r in s:
            fmts[r["fmt"]] = fmts.get(r["fmt"], 0) + 1
        print(f"  --- {label}（n={len(s)}）格式分布 {fmts}")
        for k, nm in feats + px_feats:
            v = np.array([r[k] for r in s], dtype=float)
            print(f"      {nm:<20} 中位 {np.median(v):>12.4f}   均值 {v.mean():>12.4f}   "
                  f"范围 [{v.min():.4f}, {v.max():.4f}]")
        print()

    y = [1 if r["tag"] == "fake" else 0 for r in rows]
    print("  单特征对真/假的判别力（AUC：0.5=随机，1.0=完美分开）")
    verdict = []
    for k, nm in feats + px_feats:
        a = auc([r[k] for r in rows], y)
        flag = ""
        if a >= 0.95:
            flag = "   ← ★★ 几乎完美分开，必须消除"
        elif a >= 0.80:
            flag = "   ← ★ 明显可分，应消除或说明"
        elif a >= 0.65:
            flag = "   ← 弱线索，建议关注"
        print(f"      {nm:<20} AUC = {a:.4f}{flag}")
        if a >= 0.80:
            verdict.append(nm)

    if px_feats:
        print()
        print("  ⚠ 上面「像素级」四项（高频能量 / 功率谱斜率 / Laplacian 方差 / 饱和度）")
        print("     是从**解码后的像素**算的 —— 模型评测时也只能看到像素，")
        print("     所以它们才是真正能决定成败的一类线索；")
        print("     而「格式 / 分辨率 / 字节数」是**文件级**线索，模型看不到。")
        print("     若像素级线索全部落到 0.65 以下，即使 bpp 还有残余可分性，")
        print("     也说明模型没有「不看内容就能赢」的捷径。")
        print("     注意：功率谱斜率检验的是「是不是 AI 生成」，不是「有没有局部篡改」；")
        print("     它偏高时要在论文里区分「生成痕迹」与「篡改痕迹」两种口径。")

    # 4) 结论
    sec("4. 结论与建议动作（脚本不执行，仅供人工决策）")
    if verdict:
        print("  ❌ 这批数据存在与篡改痕迹无关的强可分线索：")
        for v in verdict:
            print(f"       - {v}")
        print("     直接用于训练/评测，模型可以「不学任何伪造线索」就拿到接近满分，")
        print("     指标不可信。**必须先把两侧统一到同一格式、同一压缩质量、同一分辨率**")
        print("     （例如全部重编码为 JPEG q=90 且长边统一到 512），再重新跑一次本脚本确认 AUC 落到 0.6 以下。")
    else:
        print("  ✅ 未发现强平凡线索。")
    if idx2syn and len(inter) < 0.5 * min(len(counts["fake"]), len(counts["real"])):
        print(f"  ⚠ 两类集合重合度低（交集 {len(inter)} 个）——若做「内容可控」对照，")
        print("     建议只保留交集类别，或明确声明类别不匹配这一限制。")
    print("  ⚠ 整图生成（无掩码）只能用于真伪分类，**不能用于像素级定位**。")
    print("  ⚠ 记住数据出处与 split（train/val）；若日后在 GenImage 上训练，")
    print("     需核对本批是否与训练集重叠，否则是数据泄漏。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
