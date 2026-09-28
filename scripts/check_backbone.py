#!/usr/bin/env python
# ============================================================================
#  check_backbone.py —— 在烧算力之前，回答一个问题：
#      「这次训练用的骨干，到底是真 CLIP 预训练权重，还是静默兜底的随机初始化？」
#
#  为什么必须有这个脚本：
#    SpatialBranch 里 allow_fallback 写死为 True —— 一旦 CLIP 权重取不到
#    （没装 transformers / torch 与 transformers 版本互斥 / 没网且无本地权重），
#    代码会**不报错**地换成 TinyViTFallback 继续跑。训练照常收敛、指标照常产出，
#    但"用了 CLIP 语义先验"这个论文前提已经不成立了（见 docs/08 D12）。
#    这种错不会自己暴露，只会在答辩/审稿被追问时爆掉。
#
#  判据（三条互斥，看的是模型**实际装配完**的状态，不是配置里写了什么）：
#    OK    is_fallback=False                    真 CLIP 骨干
#    WARN  is_fallback=True  + pretrained_loaded  离线兜底骨干 + 装了本地权重
#                                                 （= 路线 B，论文必须声明"骨干替换"）
#    FAIL  is_fallback=True  + not pretrained_loaded  随机初始化 → 指标无意义
#
#  退出码：0=OK  2=WARN  3=FAIL
#
#  用法：
#      python scripts/check_backbone.py                 # 用 configs/default.yaml
#      python scripts/check_backbone.py --require-clip  # 非 OK 即返回 3（云端硬闸门用）
#      python scripts/check_backbone.py --config configs/lite.yaml
# ============================================================================
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, WARN, FAIL = 0, 2, 3


def parse_ver(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.split("+")[0].split(".")[:2])
    except Exception:                       # noqa: BLE001
        return (0, 0)


def diagnose_torch_transformers() -> dict:
    """把"CLIP 取不到"的**最常见根因**单独查一遍，别让人去猜。"""
    info: dict = {"torch": None, "transformers": None, "torch_backend": None}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception as e:                  # noqa: BLE001
        print(f"[X] 连 torch 都导不进来：{e}")
        return info

    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception as e:                  # noqa: BLE001
        print(f"[!] transformes 未安装或导入失败：{type(e).__name__}: {e}")
        print("    修： python -m pip install 'transformers<5'")
        return info

    try:
        from transformers.utils import is_torch_available
        info["torch_backend"] = bool(is_torch_available())
    except Exception:                       # noqa: BLE001
        info["torch_backend"] = None
        return info

    if info["torch_backend"] is False:
        print("[X] transformers 认为 torch 不可用 —— CLIP 一定会加载失败。")
        print(f"    实测：torch {info['torch']} / transformers {info['transformers']}")
        if parse_ver(info["torch"]) < (2, 6):
            print("    根因：新版 transformers（>=4.56）要求 torch>=2.6，当前 torch 更低。")
            print("    二选一 —— 强烈建议选 ①（不动 torch，秒级完成）：")
            print("      ① 降 transformers（约 10MB）：")
            print("         python -m pip install 'transformers==4.44.2'")
            print("         python -c \"from transformers import CLIPVisionModel\"")
            print("      ② 升 torch 到 >=2.6（约 2.5GB）：")
            print("         ⚠️ 别用 cu121 源 —— 它最高只到 torch 2.5.1，过不了这道闸门，")
            print("            必须换 cu124（镜像驱动需 >= 550）：")
            print("         python -m pip install torch==2.6.0 torchvision==0.21.0 \\")
            print("             --index-url https://download.pytorch.org/whl/cu124")
        else:
            print("    torch 版本够高，逐条看上面的导入报错。")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="检查骨干是否真的是预训练 CLIP")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--require-clip", action="store_true",
                    help="非 OK 状态一律返回 3（云端 probe/train 的硬闸门）")
    args = ap.parse_args()

    print("=" * 68)
    print("骨干真伪检查 —— 这是「能不能写进论文」的前置条件")
    print("=" * 68)

    print("\n[1] torch / transformers 环境")
    env = diagnose_torch_transformers()
    if env.get("torch") and env.get("transformers") and env.get("torch_backend"):
        print(f"    [OK] torch {env['torch']} + transformers {env['transformers']}"
              f"（torch 后端可用）")

    # ---- 真正的判据：把模型装出来，看它实际长什么样 ----------------------
    print("\n[2] 按配置实际装配模型（这一步会打印骨干装载过程）")
    from src.models.vibnet import build_model, load_config
    cfg = load_config(args.config)
    s = cfg["model"]["spatial"]
    print(f"    配置：backbone={s.get('backbone')} variant={s.get('backbone_variant')}")
    print(f"          clip_pretrained={s.get('clip_pretrained')}")
    print(f"          clip_local_dir={s.get('clip_local_dir')}")
    print(f"          backbone_weights={s.get('backbone_weights')}")

    model = build_model(cfg)
    bb = model.spatial.backbone
    is_fallback = bool(getattr(bb, "is_fallback", True))
    loaded = bool(getattr(bb, "pretrained_loaded", False))

    print("\n[3] 结论")
    if not is_fallback:
        print("    [OK] 骨干 = 真 CLIP 预训练权重")
        print("    → 与申报书口径一致，可以开始训练。")
        code = OK
    elif loaded:
        print("    [!] 骨干 = 离线兜底架构 + 本地预训练权重（即「路线 B」）")
        print("    → 训练可以跑，指标有效，但**不能**把结论写成"
              "「验证了 CLIP 语义先验的作用」。")
        print("       论文里必须显式声明「骨干替换为 ImageNet 预训练 ViT」。")
        code = WARN
    else:
        print("    [X] 骨干 = 随机初始化（既没 CLIP 权重，也没本地预训练权重）")
        print("    → 指标无意义。ablation 已证明此时 tn=0：模型把一切都判成伪造，")
        print("      边界根本不存在（见 docs/14）。**先修环境，别烧 GPU。**")
        code = FAIL

    if args.require_clip and code != OK:
        print("\n    [闸门] --require-clip 生效：非 OK 状态一律失败。")
        print("    确实要带着兜底骨干跑（只做代码链路验证）时，显式设：")
        print("        ALLOW_BACKBONE_FALLBACK=1")
        return FAIL

    print("\n[4] 建议的下一步")
    if code == OK:
        # ⚠️ 别无条件推荐 probe：无卡模式下 probe 的 CUDA 闸门必然把它拦下，
        #    而那时真正该做的是"下数据 + 数据体检"。
        cuda = False
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except Exception:                   # noqa: BLE001
            pass
        if cuda:
            print("    有卡： bash scripts/cloud_autodl.sh probe")
        else:
            print("    当前看不到 CUDA（无卡模式）—— 别急着 probe，先把数据准备好：")
            print("        bash scripts/cloud_autodl.sh data    # 约 90GB，务必 nohup 挂后台")
            print("        bash scripts/cloud_autodl.sh verify  # 全绿后 → 关机 → 切【有卡模式】→ probe")
    else:
        print("    先跑： bash scripts/cloud_autodl.sh clip   （修完再回来）")
    return code


if __name__ == "__main__":
    sys.exit(main())
