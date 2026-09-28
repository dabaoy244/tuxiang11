#!/usr/bin/env bash
# ============================================================================
#  AutoDL 上云一键脚本 —— 空频双分支轻量 VIB-Net 图像篡改检测
#
#  用法（在云实例上，项目根目录执行）：
#      bash scripts/cloud_autodl.sh doctor     # 体检：GPU / torch / 磁盘 / 依赖
#      bash scripts/cloud_autodl.sh status     # 只读：我现在走到哪一步 / 下一步跑哪个 stage
#      bash scripts/cloud_autodl.sh install    # 装依赖（云端子集）
#      bash scripts/cloud_autodl.sh clip       # 取真 CLIP-ViT-B/16 权重（路线 A）
#      bash scripts/cloud_autodl.sh data       # 下数据（val / test / CASIA / train 子集）
#      bash scripts/cloud_autodl.sh verify     # 数据体检 + 全量自检
#      bash scripts/cloud_autodl.sh probe      # 短跑 100 步，实测速度并外推卡时与费用
#      bash scripts/cloud_autodl.sh train      # 正式三阶段训练（后台 + 结束自动关机）
#      bash scripts/cloud_autodl.sh pack       # 打包权重/日志/结果，便于拉回本地
#
#  train 后面可以跟任意 scripts/train.py 的参数，会原样透传（见下面 TRAIN_EXTRA）：
#      bash scripts/cloud_autodl.sh train --resume checkpoints/vibnet_best.pt \
#          --start-stage stage2_loc_pretrain        # 跳过已训完的 stage1，直接续 stage2
#      bash scripts/cloud_autodl.sh train --tag probe2 --limit-batches 200
#
#  设计原则：**先无卡模式把所有不烧算力的活做完，再开机跑训练。**
#  doctor/install/clip/data/verify/pack 都应在【无卡模式】下执行（¥0.1/时）。
#
#  可覆盖的环境变量：
#      PROJ=/root/autodl-tmp/vibnet-forgery-detector   项目根（必须在数据盘上）
#      DATA_ROOT=/root/autodl-tmp/data                 fetch_*.py 的 --root（会自建 Datasets/）
#      DATASETS=/root/autodl-tmp/data/Datasets         训练脚本的 data.root
#      CLASSES="car cat chair horse"                   只解压这些训练集类别
#      JOBS=8                                          DataLoader 进程数
#      PY=python                                       解释器
#      AUTOSHUTDOWN=1                                  训练结束后自动关机
#      CONFIG=default                                  configs/<CONFIG>.yaml
#      CUDNN=on                                        卷积后端：on / off / bench
#                                                      （off = 关 cuDNN 走 torch 原生卷积，
#                                                        排查"反向卷积掉进慢路径"时做对照）
#      ALLOW_BACKBONE_FALLBACK=1                       跳过"骨干必须是真 CLIP"闸门
#                                                      （仅用于纯代码链路验证！）
# ============================================================================
set -uo pipefail

PROJ="${PROJ:-/root/autodl-tmp/vibnet-forgery-detector}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
DATASETS="${DATASETS:-$DATA_ROOT/Datasets}"
CLASSES="${CLASSES:-car cat chair horse}"
JOBS="${JOBS:-8}"
PY="${PY:-python}"
AUTOSHUTDOWN="${AUTOSHUTDOWN:-1}"
CONFIG="${CONFIG:-default}"
# 必须有默认值：runner 里写的是 `--cudnn $CUDNN`，未设置时展开成空串，
# 那一行的结尾会变成孤零零的 `--cudnn`，argparse 直接 "expected one argument"
# 退出码 2 —— 训练连模型都没建起来就死了，而报错在 outputs/gpu_train_log.txt
# 的最前面，很容易被当成"环境问题"。
CUDNN="${CUDNN:-on}"
ALLOW_BACKBONE_FALLBACK="${ALLOW_BACKBONE_FALLBACK:-0}"
STAGE="${1:-doctor}"
# 附加参数：`bash scripts/cloud_autodl.sh train --resume X --start-stage Y`
# 直接透传给 scripts/train.py。有了它就不用为了换一个开关去改脚本文本 ——
# 改文本的做法既容易改错（改完忘了改回来），也无法在两条命令之间复用。
if [ "$#" -gt 1 ]; then
    shift
    TRAIN_EXTRA="$*"
else
    TRAIN_EXTRA=""
fi

hr()   { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[X]\033[0m %s\n' "$*" >&2; exit 1; }

# 无卡模式容器的内存上限比有卡模式小得多，在 CPU 上跑整个模型很容易被杀。
# exit=137 = 进程收到 SIGKILL，Linux 下几乎总是 OOM killer —— 那是环境，不是代码。
# 2026-09-24 云上实测踩到：smoke_test 与 2-batch 试跑都被 Killed，
# 而它们被杀与"GPU 上能不能跑"毫无关系，白耽误半天。
oom_note() {
    warn "$1 被 OOM killer 杀掉（exit=137）—— 这是容器内存不足，**不是代码错**。"
    echo "       本步只回答「数据能不能喂进模型」，不是性能测试。先看真实内存与核数："
    echo "         free -g ; nproc"
    echo "       再用轻量参数重跑（结论等价，内存约降一个数量级）："
    echo "         $PY scripts/train.py --config configs/$CONFIG.yaml --device cpu \\"
    echo "             --workers 0 --batch-size 2 --data-root $DATASETS \\"
    echo "             --out-dir outputs/_io_probe --limit-batches 2 --tag ioprobe"
    echo "       跑完 2 个 batch 就算过 —— GPU 上快不快，由 probe 实测说话。"
}

# ============================================================================
#  内部工具
# ============================================================================
# "transformers 能不能真的用 torch" —— 这是 CLIP 取不到的头号根因，
# 而且**只表现为 from transformers import CLIPVisionModel 抛 ImportError**，
# 看起来像网络问题，实际和网络无关（见 docs/08 D38）。
tf_torch_ok() {
    $PY -c "
import sys
try:
    from transformers.utils import is_torch_available as f
except Exception:
    sys.exit(1)
sys.exit(0 if f() else 1)
" >/dev/null 2>&1
}

explain_tf_torch() {
    $PY - <<'PYEOF'
import sys
try:
    import torch, transformers
except Exception as e:                      # noqa: BLE001
    print(f"[!] 导入失败：{type(e).__name__}: {e}")
    sys.exit(0)
def mm(v):
    try:
        return tuple(int(x) for x in v.split("+")[0].split(".")[:2])
    except Exception:                       # noqa: BLE001
        return (0, 0)
print(f"    torch {torch.__version__}  |  transformers {transformers.__version__}")
if mm(torch.__version__) < (2, 6):
    print("    根因：新版 transformers（>=4.56）要求 torch>=2.6，当前 torch 更低 ——")
    print("          于是 `from transformers import CLIPVisionModel` 直接抛 ImportError，")
    print("          脚本会误报成「两个源都失败」。这和网络无关。")
    print("    修（二选一，强烈建议 ① —— 不动 torch，秒级完成）：")
    print("      ① 降 transformers（约 10MB，推荐）：")
    print("         python -m pip install 'transformers==4.44.2'")
    print("      ② 升 torch 到 >=2.6（约 2.5GB）：")
    print("         ⚠️ 不要用 cu121 源：它最高只到 torch 2.5.1，仍会被这道闸门拦下。")
    print("            必须换 cu124（镜像驱动需 >= 550）：")
    print("         python -m pip install torch==2.6.0 torchvision==0.21.0 \\")
    print("             --index-url https://download.pytorch.org/whl/cu124")
PYEOF
}

# 骨干闸门：烧 GPU 之前，确认这次的骨干**真的是**预训练 CLIP。
# 为什么必须是硬闸门：allow_fallback 写死为 True，CLIP 缺失时代码会**不报错**地
# 换成随机初始化骨干继续跑，指标照样产出但没有意义（见 docs/08 D12）。
gate_backbone() {
    at_proj
    if [ "$ALLOW_BACKBONE_FALLBACK" = "1" ]; then
        warn "ALLOW_BACKBONE_FALLBACK=1 —— 已跳过骨干闸门。"
        warn "这只允许用于「验证代码链路能不能跑通」，产出的指标不可进论文。"
        return 0
    fi
    hr "骨干闸门：这次训练用的是真 CLIP 吗？"
    if $PY scripts/check_backbone.py --config "configs/$CONFIG.yaml" --require-clip; then
        ok "骨干 = 真 CLIP 预训练权重，放行"
        return 0
    fi
    die "骨干不是预训练 CLIP —— 已拦下，**别烧 GPU**（修法见上面文字）"
}

at_proj() {
    cd "$PROJ" 2>/dev/null && return 0
    {
        echo "项目目录不存在：$PROJ"
        echo "  先在 /root/autodl-tmp 下解包代码（注意 -C，漏了会把文件摊平到当前目录）："
        echo "      cd /root/autodl-tmp"
        echo "      mkdir -p $(basename "$PROJ")"
        echo "      tar -xzf vibnet_code.tar.gz -C $(basename "$PROJ")"
        echo "      cd $(basename "$PROJ") && ls      # 应看到 src scripts configs docs ..."
    } >&2
    exit 1
}

# 最近一个"已存在"的祖先路径。
# 为什么需要它：目标目录还没建时，`df /root/autodl-tmp/data` 会直接报错并回落到
# 系统盘，屏幕上显示成 30GB —— 看起来像"数据盘只有 30GB，得换机器"，
# 其实大容量盘就在旁边。把 df 指到存在的祖先上，数字才是真的（见 docs/08 D37）。
probe_path() {
    local p="$1"
    while [ -n "$p" ] && [ "$p" != "/" ] && [ ! -e "$p" ]; do p="$(dirname "$p")"; done
    printf '%s' "$p"
}

# 文件系统可用空间（GB）。取 DATA_ROOT 所在分区，因为数据才是大头。
free_gb() {
    df -BG --output=avail "$(probe_path "$1")" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# ============================================================================
#  doctor —— 体检。任何阶段前都可以先跑一次，成本为零。
# ============================================================================
stage_doctor() {
    # 体检是**诊断**，不是执行 —— 所以任何一项不合格都不提前 exit，
    # 而是把"致命/不致命"记下来，等整份报告出完再在 §5 给结论。
    # 教训：早退会让后面几节（解包位置、旧副本）永远不显示，
    # 用户以为只差磁盘，其实位置也是错的（见 docs/08 D37 同一类）。
    local fatal=0

    hr "1. 机器与 GPU"
    uname -srm
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        warn "看不到 GPU —— 若当前是【无卡模式】，这是正常的（无卡模式不挂卡）"
    fi

    hr "2. Python / torch / 依赖"
    $PY -V 2>&1
    $PY - <<'PYEOF'
import sys
try:
    import torch
except Exception as e:                      # noqa: BLE001
    print(f"[X] 没装 torch：{e}")
    sys.exit(0)
cu = torch.version.cuda
print(f"torch {torch.__version__} | torch.version.cuda={cu} | "
      f"cuda.is_available={torch.cuda.is_available()}")
if cu is None:
    print("[X] 这是 CPU 版 torch —— GPU 跑训练必须换成 CUDA 版：")
    print("    pip install torch torchvision --index-url "
          "https://download.pytorch.org/whl/cu121")
else:
    print("[OK] torch 是 CUDA 版（无卡模式下 is_available=False 属正常）")
# 依赖里最要紧的是 transformers：它决定骨干走"真 CLIP"还是"ImageNet 兜底"
for m in ("cv2", "PIL", "numpy", "yaml", "tqdm", "transformers", "onnx"):
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "")
        print(f"[OK] {m}{(' ' + ver) if ver else ''}")
    except Exception as e:                  # noqa: BLE001
        print(f"[!] 缺 {m}（{type(e).__name__}）—— 跑 `bash scripts/cloud_autodl.sh install`")
PYEOF

    hr "3. 磁盘（整个上云流程最硬的约束）"
    local probe
    probe="$(probe_path "$DATA_ROOT")"
    echo "数据盘挂载点：$probe"
    df -h "$probe" 2>/dev/null
    case "$probe" in
        /root/autodl-tmp*|/root/autodl-fs*) ok "大容量盘已就绪" ;;
        *) warn "没落到 /root/autodl-tmp —— 数据盘可能没挂上，先确认再下数据" ;;
    esac
    local free
    free=$(free_gb "$DATA_ROOT")
    echo
    echo "数据分区可用：${free:-?} GB"
    echo "  参考阈值（峰值占用，含下载包与解压产物）："
    echo "    ≈145 GB  20 类全量（train 7 卷 74.9 + 解压 69.8）"
    echo "    ≈ 89 GB  只解 4 类（脚本默认；解压成功后自动删卷）"
    echo "    ≈ 25 GB  完全不下 train，只跑 test/val/CASIA（出泛化表，出不了正式训练）"
    if [ -n "$free" ] && [ "$free" -ge 90 ]; then
        ok "空间充足，可以走默认的 4 类路径"
    elif [ -n "$free" ] && [ "$free" -ge 25 ]; then
        warn "不够跑 train。只能先跑 test/val/CASIA；train 需要扩容数据盘（见 docs/18 §1③）"
    else
        warn "可用空间 ${free:-?} GB 太少 —— 需要先扩容数据盘（见 docs/18 §1③）"
        fatal=1
    fi

    hr "4. 路径检查"
    case "$PROJ" in
        /root/autodl-tmp/*|/root/autodl-fs/*) ok "项目路径在数据盘前缀上：$PROJ" ;;
        *) warn "项目不在数据盘（$PROJ）—— checkpoint 每份 362MB，写系统盘会爆。请改 PROJ" ;;
    esac
    if [ -d "$PROJ" ]; then
        ok "项目目录存在"
    else
        warn "项目目录还不存在：$PROJ"
        # 最高频的事故：tar 解包漏了 -C，源码被"摊平"到了上一级目录。
        local parent
        parent="$(dirname "$DATA_ROOT")"
        if [ -d "$parent/scripts" ] && [ -f "$parent/scripts/cloud_autodl.sh" ]; then
            warn "但我在 $parent 下看到了 scripts/ —— 你解包时漏了 -C，源码摊平了。一把修好："
            echo "    mv $parent/{src,scripts,configs,docs,app,tools,requirements.txt,README.md} $PROJ/"
            echo "    cd $PROJ && bash scripts/cloud_autodl.sh doctor"
        fi
    fi
    [ -d "$DATASETS/ForenSynths" ] && ok "已见 ForenSynths" || warn "还没有 $DATASETS/ForenSynths"

    # 项目存在 ≠ 环境干净。旧包（成员是散装的）可能也被解在了上一级目录，
    # 于是 "python scripts/xxx.py 到底跑的是哪一份代码" 取决于当前目录 ——
    # 典型的静默失败：脚本跑得起来、文件都在、行为却是旧版（见 docs/08 D39）。
    local up dup
    up="$(dirname "$PROJ")"
    dup="$(cd "$up" 2>/dev/null && ls -d src scripts configs docs app tools 2>/dev/null)"
    if [ -n "$dup" ]; then
        warn "$up 下还散落着同名的旧副本（多半是旧包解包漏了 -C）："
        echo "$dup" | sed 's/^/      /'
        echo "    ⚠️ 在那个目录里直接跑 python scripts/xxx.py 会用**旧代码**"
        echo "       （旧包没有骨干闸门、旧版训练参数），而且不会有任何报错。"
        echo "    归档（不是删除，可随时还原）："
        # 注意：$dup 是多行，拼进一行命令前必须把换行收成空格，
        # 否则打出来的命令中间会断行、根本没法复制执行。
        echo "      mkdir -p $up/_stale_code && mv $(echo "$dup" | tr '\n' ' ') $up/_stale_code/"
    else
        ok "上级目录干净，没有摊平的旧副本"
    fi

    hr "5. 结论"
    if [ "$fatal" != 0 ]; then
        warn "体检**未通过** —— 先把上面带 [!] 的项修好（磁盘那项是硬闸门），再跑 install。"
        return 1
    fi
    if [ -d "$PROJ" ]; then
        echo "项目已就位，接着跑（仍建议留在无卡模式）："
        echo "    bash scripts/cloud_autodl.sh install"
        echo "（随时想确认自己走到哪一步：bash scripts/cloud_autodl.sh status）"
    else
        echo "先把代码解包到 $PROJ（见上面第 4 节的提示），再回来跑 doctor。"
        echo "其余检查看上面：标 [OK] 的不用管，带 [!] 的按提示处理。"
    fi
}

# ============================================================================
#  install —— 只装云端需要的子集
# ============================================================================
stage_install() {
    at_proj
    # AutoDL 镜像通常已把 pip 指到清华源；这里显式设一遍，换机器也不慌
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export PIP_DISABLE_PIP_VERSION_CHECK=1

    hr "装云端子集依赖（不装 PyQt5 / openvino / reportlab）"
    echo "  为什么排除它们：PyQt5 是桌面工具用的，无头环境要 X 库；"
    echo "  openvino / reportlab 只在本地导出部署件时用，云端训练用不到，"
    echo "  装了只会让 pip 多花几分钟、多占几百 MB。"
    # shellcheck disable=SC2086
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q \
        "numpy>=1.24" "opencv-python-headless>=4.8" "pillow>=10.0" \
        "scipy>=1.10" "scikit-learn>=1.3" "pyyaml>=6.0" "tqdm>=4.66" \
        "transformers>=4.35,<5" "onnx>=1.14" "onnxruntime>=1.16" "onnx-simplifier>=0.4.35" \
        || die "依赖安装失败，看上面的 pip 报错"

    # 装依赖最容易踩的雷：某个包把 numpy 拉到 2.x，而镜像自带的 torch<2.4
    # 是冲着 numpy 1.x 编的 —— 表现是 `_ARRAY_API not found`，然后 torch 静默
    # 降级成"不支持 numpy"，collate 和指标计算全线崩。这里显式查一次。
    hr "torch / numpy 版本互斥自检"
    $PY - <<'PYEOF'
import numpy, torch

def mm(v):
    try:
        return tuple(int(x) for x in v.split("+")[0].split(".")[:2])
    except Exception:                       # noqa: BLE001
        return (0, 0)

nv, tv = mm(numpy.__version__), mm(torch.__version__)
print(f"numpy {numpy.__version__} | torch {torch.__version__}")
if nv >= (2, 0) and tv < (2, 4):
    print("[!] torch<2.4 与 numpy>=2 不兼容：会报 '_ARRAY_API not found'，")
    print("    torch 静默失去 numpy 支持，数据 collate 与指标计算会崩。")
    print("    修： python -m pip install 'numpy<2'")
else:
    print("[OK] 版本组合没问题")
PYEOF

    hr "torch / transformers 版本互斥自检与自动修复"
    if tf_torch_ok; then
        ok "transformers 可以正常使用 torch"
        $PY -c "import transformers;print('   transformers',transformers.__version__)"
    else
        warn "transformers 用不了 torch —— 这会让 CLIP 权重**永远取不到**（假装是网络问题）。"
        explain_tf_torch
        echo
        echo "  自动降级到兼容版本（逐个试，每个约 10MB）……"
        local fixed=0
        for v in 4.44.2 4.46.3 4.49.0; do
            echo "  --- 试 transformers==$v ---"
            $PY -m pip install -q "transformers==$v" || continue
            if $PY -c "from transformers import CLIPVisionModel" 2>/dev/null; then
                ok "transformers==$v 可用（CLIPVisionModel 导入成功）"
                fixed=1
                break
            fi
            warn "$v 仍不可用，换下一个"
        done
        if [ "$fixed" = "0" ]; then
            die "三个候选版本都不行。唯一剩下的路是把 torch 升到 >=2.6：
      ⚠️ 必须用 cu124 源（cu121 最高只到 torch 2.5.1，仍会被闸门拦下）：
      $PY -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
      装完复验：$PY -c \"from transformers import CLIPVisionModel; print('ok')\""
        fi
    fi

    hr "校验 torch 是不是 CUDA 版"
    if $PY -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"; then
        ok "torch 已是 CUDA 版，不动它"
        $PY -c "import torch;print('   torch',torch.__version__,'| cuda',torch.version.cuda)"
    else
        warn "当前 torch 不是 CUDA 版。执行下面这条（约 2.5GB，建议仍在无卡模式）："
        echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        echo "    python -c \"import torch;print(torch.version.cuda)\"   # 应打印 12.1"
        return 1
    fi

    hr "装完复检"
    stage_doctor
}

# ============================================================================
#  clip —— 取真 CLIP 权重（决定论文怎么写，见 docs/15 §5）
# ============================================================================
stage_clip() {
    at_proj

    hr "0. 先确认 transformers 能用 torch —— 否则下面两个源都会白白失败"
    if ! tf_torch_ok; then
        warn "transformers 当前用不了 torch：无论连 HuggingFace 还是 hf-mirror，"
        warn "都会以「ImportError: CLIPVisionModel requires the PyTorch library」收场。"
        warn "**这不是网络问题**，是 torch 与 transformers 版本互斥。先修："
        explain_tf_torch
        die "先把上面这条修好，再跑 clip"
    fi
    ok "transformers 的 torch 后端可用"

    # AutoDL 自带学术加速（github / huggingface 走代理）。没有这个文件就跳过。
    if [ -f /etc/network_turbo ]; then
        # shellcheck disable=SC1091
        source /etc/network_turbo && ok "已开启 AutoDL 学术加速"
    else
        warn "没有 /etc/network_turbo，直接试直连 + 镜像"
    fi
    export VIB_NET_ALLOW_DOWNLOAD=1
    mkdir -p pretrained

    for ep in "https://huggingface.co" "https://hf-mirror.com"; do
        hr "尝试从 $ep 拉 CLIP-ViT-B/16"
        if HF_ENDPOINT="$ep" $PY - <<'PYEOF'
import os, sys
from transformers import CLIPVisionModel
out = "pretrained/clip-vit-base-patch16"
m = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
m.save_pretrained(out)
print("[OK] 已保存到", out)
PYEOF
        then
            ok "路线 A 成功：$ep"
            ls -la pretrained/clip-vit-base-patch16
            hr "复核：模型装配之后，骨干到底是哪一个"
            $PY scripts/check_backbone.py --config "configs/$CONFIG.yaml" --require-clip \
                && ok "已确认骨干 = 真 CLIP" \
                || die "权重下下来了，但模型没能装成 CLIP —— 看上面 check_backbone 的输出"
            return 0
        fi
        warn "$ep 失败，换下一个源"
    done

    warn "两个源都失败 —— 先排除「版本互斥导致的假失败」（本节第 0 步已查过；"
    warn "确认不是版本问题，才轮到「网络」这个解释）。确属网络问题，再走路线 B。"
    cat <<'TIPEOF'
  路线 B 做法（论文里必须如实写"骨干替换"）：
    1) 本地执行：scp -P <端口> pretrained/vit_b16_imagenet.pt root@<host>:/root/autodl-tmp/vibnet-forgery-detector/pretrained/
    2) 改 configs/<config>.yaml：
         model.spatial.backbone: "tiny_vit"
         model.spatial.backbone_variant: "b16"
         model.spatial.backbone_weights: "pretrained/vit_b16_imagenet.pt"
         model.spatial.global_dim: 768
    3) 不能把结论写成"验证了 CLIP 语义先验的作用"。
    4) ⚠️ 第 2) 步里 backbone_weights 才是关键：**没有它，骨干就是随机初始化**，
       指标无意义（abl 已证此时 tn=0）。复核命令：
           python scripts/check_backbone.py
       它会明确告诉你现在是「真 CLIP」「路线 B（有意义）」还是「随机初始化（无意义）」。
TIPEOF
    return 1
}

# ============================================================================
#  data —— 在云端重新下（上传 90GB 比下载慢得多）
# ============================================================================
stage_data() {
    at_proj
    mkdir -p "$DATA_ROOT"

    hr "0. 空间预检"
    local free
    free=$(free_gb "$DATA_ROOT")
    echo "可用 ${free:-?} GB"
    if [ -n "$free" ] && [ "$free" -lt 95 ]; then
        warn "预计不足 95GB。train 那 7 卷 74.9GB 下完就快满了。"
        echo "  继续将只下【不含 train】的部分，train 请先扩容数据盘。"
        local do_train=0
    else
        local do_train=1
    fi

    hr "1. val（0.8GB，最便宜的一步，先验证链路）"
    $PY scripts/fetch_datasets.py --split val --root "$DATA_ROOT" \
        || warn "val 未完成（可重跑，支持续传）"

    hr "2. progan_test + CNN_synth_testset（约 20GB，泛化表要用）"
    $PY scripts/fetch_datasets.py --split progan_test test --root "$DATA_ROOT" \
        || warn "test 未完成（可重跑，支持续传）"

    hr "3. CASIA v2.0（3.3GB，定位指标 mIoU 必需）"
    $PY scripts/fetch_tamper_datasets.py --root "$DATA_ROOT" --workers "$JOBS" \
        || warn "CASIA 失败就稍后重试；它会断点续传"

    if [ "$do_train" = "1" ]; then
        hr "4. train 只解 ${CLASSES}（省 56GB；train 实际只取 4 万张，见 docs/18 §2）"
        echo "  如果不确定类名，先跑： python scripts/fetch_datasets.py --list-train --root $DATA_ROOT"
        $PY scripts/fetch_datasets.py --split train --classes $CLASSES --root "$DATA_ROOT" \
            || warn "train 阶段退出码非 0。**别急着重跑**（分卷若已删会重下 70GB），先看上面输出"

        # 结构性闸门：train 必须落在 ForenSynths/train/<类>/ —— GenSynthsDataset 的
        # split="train" 只认这一层（src/data/datasets.py 明确不拿 <base> 当回退）。
        # 放错位置时数据"在盘上但一条都扫不到"，属于不报错的静默失效，必须在这里拦。
        local miss=0
        for c in $CLASSES; do
            [ -d "$DATASETS/ForenSynths/train/$c/0_real" ] || miss=1
        done
        if [ "$miss" = "1" ]; then
            warn "train 未落在 ForenSynths/train/<类>/ —— split=\"train\" 会扫到 0 条样本"
            local legacy
            legacy=$(ls -d "$DATASETS"/ForenSynths/*/0_real 2>/dev/null | head -1)
            if [ -n "$legacy" ]; then
                echo "  检测到旧版脚本的误放位置：$legacy"
                echo "  一条命令归位（同盘 mv 是瞬时的）："
                echo "    cd $DATASETS/ForenSynths && mkdir -p train && mv $CLASSES train/"
            fi
        fi
    else
        warn "跳过 train 下载（空间不足）"
    fi

    hr "5. 目录与占用"
    du -sh "$DATASETS"/* 2>/dev/null
    df -h "$DATA_ROOT"
    echo
    echo "接着跑： bash scripts/cloud_autodl.sh verify"
}

# ============================================================================
#  verify —— 上云后第一件事：数据体检 + 全量自检
# ============================================================================
stage_verify() {
    at_proj
    gate_failed=0     # 1 = 有硬闸门没过（禁止切【有卡模式】）
    soft_failed=0     # 1 = 只是容器内存不足被 OOM，不是代码错
    hr "1. 数据脚本自检（合成数据，不联网）"
    $PY scripts/fetch_datasets.py --self-test || warn "fetch_datasets 自检未全绿"

    hr "2. 数据结构与计数"
    $PY scripts/prepare_datasets.py --root "$DATASETS" --check

    hr "3. 逐生成器清点（看哪个生成器样本少，抽样要分层）"
    $PY scripts/eval_cross_generator.py --list-only \
        --data-root "$DATASETS/ForenSynths/test" || warn "test 还没下？"

    hr "4. CASIA 掩码数必须 == 篡改图数（不等就别往下跑）"
    $PY - <<PYEOF
import sys
sys.path.insert(0, ".")
from src.data.datasets import TamperDataset
for sp in ("train", "val", "test"):
    try:
        TamperDataset("$DATASETS", "CASIAv2", split=sp, max_samples=None)
        print(f"[OK] CASIAv2/{sp}")
    except Exception as e:                  # noqa: BLE001
        print(f"[!] CASIAv2/{sp}: {type(e).__name__}: {e}")
PYEOF

    hr "5. 全量冒烟测试（含合成流水线）"
    # 本步会真跑一次 forward+backward（batch=4）—— 无卡模式内存小的话会被 OOM 杀掉，
    # 那与代码无关（见 oom_note）。137 单独判，别把它当成"冒烟测试没过"。
    $PY scripts/smoke_test.py; rc5=$?
    if [ "$rc5" -eq 137 ]; then
        oom_note "冒烟测试"
        soft_failed=1
    elif [ "$rc5" -ne 0 ]; then
        warn "冒烟测试未全绿（exit=$rc5）—— 先别开 GPU 跑正式训练"
        gate_failed=1
    fi

    hr "6. 训练集能不能被吃进去（用 2 个 batch 试）"
    # 用最小 batch + 0 worker：本步只回答"数据能不能喂进模型"，不是性能测试。
    # 按正式 batch(16)+workers(2) 跑，无卡模式很容易被 OOM 杀掉，而那种"被杀"
    # 与 GPU 上跑不跑得动毫无关系。也刻意不加 --amp：CPU 上 bf16 autocast
    # 明显慢于 fp32，加了只会白等（amp 只在 CUDA 上才有收益）。
    free -g 2>/dev/null | head -2 || true
    mkdir -p outputs/_io_probe
    io_log="outputs/_io_probe/verify_io.txt"
    $PY scripts/train.py --config configs/$CONFIG.yaml --device cpu \
        --workers 0 --batch-size 2 --data-root "$DATASETS" --out-dir outputs/_io_probe \
        --limit-batches 2 --tag ioprobe 2>&1 | tee "$io_log"
    rc6="${PIPESTATUS[0]}"     # 必须紧跟管道：中间插 local/echo 都会把 PIPESTATUS 冲掉
    if [ "$rc6" -eq 137 ]; then
        oom_note "2 batch 试跑"
        soft_failed=1
    elif [ "$rc6" -ne 0 ]; then
        warn "数据装载失败（exit=$rc6），看上面的报错。常见两类："
        echo "        ① 路径错：--data-root 应指向 <data>/Datasets"
        echo "        ② 布局错：某个数据集 0 条 —— 见下面的 layout_hint 归位命令"
        gate_failed=1
    elif grep -q "一个样本都没有" "$io_log"; then
        # ★ 硬闸门：加载器只在"整个 split 全空"时才抛错；**部分**数据集为空时
        #   只打一行 ⚠，训练照样"正常"跑完 —— 但少掉那部分数据（可能是全部训练集）。
        #   这是本项目最危险的一类静默失效，所以在此升级成硬闸门。
        warn "train split 里有数据集 0 条（数据在盘上但层级对不上）—— 训练会少掉这部分数据"
        echo "        log 里 [data] 那段给出的实际目录与归位命令："
        grep -A8 "一个样本都没有" "$io_log" | sed 's/^/        /'
        gate_failed=1
    else
        ok "数据能被吃进模型（2 个 batch 正常）"
    fi

    hr "7. 骨干真伪（现在就要知道，别等烧完 GPU 才发现用的是随机初始化骨干）"
    if ! $PY scripts/check_backbone.py --config "configs/$CONFIG.yaml"; then
        warn "骨干不是真 CLIP —— probe/train 会被闸门拦下，先回去跑 clip"
        gate_failed=1
    fi

    echo
    if [ "$gate_failed" = 1 ]; then
        die "verify 未通过 —— 上面标 [!] 的项必须先修好，**先别切【有卡模式】**。修完重跑 verify。"
    fi
    if [ "$soft_failed" = 1 ]; then
        warn "有 1 项因容器内存不足被 OOM 跳过（不是代码错、也不影响 GPU 运行）。"
        echo "       想 100% 全绿，就按上面 oom_note 给的轻量命令单独重跑那一步。"
    fi
    ok "verify 通过 —— 可以关机，切【有卡模式】跑 probe 了"
    echo "   bash scripts/cloud_autodl.sh probe"
}

# ============================================================================
#  probe —— 短跑 100 步实测速度，外推 70 轮要多少卡时、多少钱
#           这一步的价值：把"要花多少钱"变成实测数字，而不是拍脑袋
# ============================================================================
stage_probe() {
    at_proj
    # 烧 GPU 之前的第一道闸门：骨干必须真的是预训练 CLIP
    gate_backbone

    # 第二道：没有 CUDA 就不能 probe —— 在 CPU 上测出的单步耗时外推出来的
    # 「要跑几小时、花多少钱」是错的，而且错得很大（CPU 比 GPU 慢两个数量级）。
    if ! $PY -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.version.cuda) else 1)"; then
        die "看不到可用的 CUDA。probe 的意义是实测 GPU 速度，在 CPU 上测出来的数字没有意义。
      请确认实例已切成【有卡模式】并挂着卡（nvidia-smi 能看到 4090）。"
    fi

    local out="outputs/probe_$(date +%Y%m%d_%H%M)"
    mkdir -p "$out"

    hr "短跑 100 步（--amp bf16）"
    $PY -u scripts/train.py --config configs/$CONFIG.yaml --device cuda --amp \
        --workers "$JOBS" --data-root "$DATASETS" \
        --out-dir "$out" --limit-batches 100 --tag probe 2>&1 | tee "$out/probe_stdout.txt"

    hr "外推 70 轮（20+20+30）的卡时与费用"
    $PY - "$out/probe_stdout.txt" <<'PYEOF'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
n_ds = re.search(r"训练集\s*(\d+)\s*张", txt)
iters = re.findall(r"iter\s+(\d+)/(\d+)\s+loss=[-\d.]+\s+([\d.]+)s", txt)
if not n_ds or not iters:
    print("[!] 没解析到，用手工看：日志里 `iter 100/100  loss=...  XXs` 的秒数 ÷ 100 = 单步耗时")
    sys.exit(0)
N = int(n_ds.group(1))
i, tot, sec = iters[-1]
per_step = float(sec) / int(i)
batch = 16
steps = max(1, -(-N // batch))
epoch_min = per_step * steps / 60
hourly = 1.88
total_h = epoch_min * 70 * 1.2 / 60          # 70 轮 + 约 20% 评测开销
print(f"训练集        : {N:,} 张  ->  约 {steps:,} 步/epoch（batch={batch}）")
print(f"单步耗时      : {per_step:.3f} s")
print(f"单 epoch      : {epoch_min:.1f} min")
print(f"70 轮 + 20%开销: {total_h:.1f} 小时")
print(f"按 ¥{hourly}/时   : 约 ¥{total_h*hourly:.0f}（不含数据盘月费）")
print()
print("=> 若上面这个数字 > 150 小时，先别开跑：查 num_workers、是否真的吃到了 GPU、")
print("   数据是否落在数据盘（系统盘 IO 慢会拖死 dataloader）。")
PYEOF
    echo
    echo "数字合理后再跑： bash scripts/cloud_autodl.sh train"
}

# ============================================================================
#  train —— 正式三阶段训练，后台跑 + 结束自动关机
# ============================================================================
stage_train() {
    at_proj
    mkdir -p outputs

    # 最后一道闸门：真金白银开跑之前，再确认一次骨干
    gate_backbone

    if ! $PY -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.version.cuda) else 1)"; then
        die "现在看不到可用的 CUDA —— 请确认实例是【有卡模式】且挂着 4090"
    fi

    local runner="outputs/_run_train.sh"
    # 透传参数要在**生成时**展开（heredoc 里用 $TRAIN_EXTRA 而非 \$TRAIN_EXTRA）。
    # 空串时会留在行尾，等价于没写 —— 已验证不会产生空参数报错（见 smoke_test）。
    cat > "$runner" <<EOF
#!/usr/bin/env bash
cd "$PROJ"
echo "[\$(date '+%F %T')] 开始三阶段训练  extra_args=[$TRAIN_EXTRA]"
t0=\$(date +%s)
$PY -u scripts/train.py --config configs/$CONFIG.yaml --device cuda --amp \\
    --workers $JOBS --data-root "$DATASETS" --tag vibnet --cudnn $CUDNN $TRAIN_EXTRA
rc=\$?
t1=\$(date +%s)
el=\$(( t1 - t0 ))
echo "[\$(date '+%F %T')] train 退出码=\$rc  用时 \$(( el / 60 )) 分 \$(( el % 60 )) 秒"

# 运行报告：万一实例被关机，先把"为什么结束"落盘，省得重新开机才能知道。
{
    echo "训练运行报告  (由 _run_train.sh 生成)"
    echo "  结束时间 : \$(date '+%F %T')"
    echo "  退出码   : \$rc"
    echo "  用时     : \$(( el / 60 )) 分 \$(( el % 60 )) 秒"
    echo "  训练日志 : outputs/gpu_train_log.txt"
    echo "  早停记录 : grep -n 早停 outputs/gpu_train_log.txt"
    echo
    echo "------ 日志末尾 60 行 ------" 
    tail -n 60 outputs/gpu_train_log.txt
} > outputs/RUN_REPORT.txt 2>&1

if [ "\$AUTOSHUTDOWN" = "1" ]; then
    if [ "\$rc" != "0" ]; then
        echo "[\$(date '+%F %T')] train 非正常退出（退出码 \$rc）—— **不自动关机**，"
        echo "    实例保留供你排查。先看：tail -40 outputs/gpu_train_log.txt"
        echo "    处理完手动关机：/usr/bin/shutdown"
        echo "    ⚠ 从此刻起实例仍在计费（约 ¥1.88/时），别放着不管。"
        exit 0
    fi
    sync
    echo "[\$(date '+%F %T')] 自动关机（按量计费：关机后不保留这张卡）"
    /usr/bin/shutdown
fi
EOF
    chmod +x "$runner"

    hr "后台启动（SSH 断了也不影响）"
    AUTOSHUTDOWN="$AUTOSHUTDOWN" nohup bash "$runner" > outputs/gpu_train_log.txt 2>&1 &
    sleep 20

    echo "PID=$!  日志=outputs/gpu_train_log.txt"
    tail -20 outputs/gpu_train_log.txt
    echo
    ok "已启动。接下来："
    echo "    看进度 : tail -f outputs/gpu_train_log.txt"
    echo "    看状态 : python scripts/check_status.py"
    echo "    不要关实例；训练结束会自己关机（AUTOSHUTDOWN=$AUTOSHUTDOWN）"
    echo "    下次接着跑（断点续训）："
    echo "      python scripts/train.py --config configs/$CONFIG.yaml --device cuda --amp \\"
    echo "          --workers $JOBS --data-root $DATASETS --tag vibnet --resume checkpoints/vibnet_last.pt"
    echo "    某一阶段已经训完、只接着跑后面的阶段（省掉重跑的时间与卡时）："
    echo "      bash scripts/cloud_autodl.sh train --resume checkpoints/vibnet_best.pt \\"
    echo "          --start-stage stage2_loc_pretrain"
    echo "    ⚠ --resume 只加载权重，不会推进阶段/轮次；不写 --start-stage 就会从 stage1 第 1 轮重来。"
}

# ============================================================================
#  pack —— 打包结果。开实例期间传小文件很快，别等到释放实例才想起来
# ============================================================================
stage_pack() {
    at_proj
    local outfile="/root/autodl-tmp/vibnet_results_$(date +%Y%m%d_%H%M).tar.gz"
    local stage_dir="outputs/_pack"
    rm -rf "$stage_dir"; mkdir -p "$stage_dir"

    # 日志 / json / 图：小而且要进论文
    find outputs -maxdepth 2 -type f \
        \( -name '*.json' -o -name '*.txt' -o -name '*.png' -o -name '*.csv' \) \
        -not -name 'fetch_*_log.txt' -exec cp -f {} "$stage_dir"/ \; 2>/dev/null
    # 权重只带 best/last（每份 362MB，别把 epoch*.pt 全带上）
    cp -f checkpoints/vibnet_best.pt checkpoints/vibnet_last.pt "$stage_dir"/ 2>/dev/null

    tar -czf "$outfile" -C outputs _pack
    ls -lh "$outfile"
    echo
    ok "产物：$outfile"
    echo "拉回本地（在本地 Git Bash 执行，端口/主机看 AutoDL 控制台）："
    echo "    scp -P <端口> root@<host>:$outfile ./"
    echo "或者丢到 /root/autodl-fs（同地区 20GB 内免费）做跨实例中转。"
}

# ============================================================================
#  status —— 只读体检：我现在走到哪一步了？下一步该跑哪个 stage？
#            全程只读，不改任何东西；无卡模式下也能跑。
#            用途：中断几天后回来，不用回忆，一条命令定位当前位置。
# ============================================================================
stage_status() {
    hr "0. 位置与磁盘"
    if cd "$PROJ" 2>/dev/null; then ok "cwd = $(pwd)"; else
        warn "项目目录不存在：$PROJ —— 先解包代码（见 doctor 第 4 节）"; return 1
    fi
    df -h "$(probe_path "$DATA_ROOT")" 2>/dev/null | tail -1

    hr "1. 数据：下了多少"
    if [ -d "$DATA_ROOT/_downloads" ]; then
        ls -la "$DATA_ROOT/_downloads" | tail -n +2 | sed 's/^/    /'
        echo "    压缩包合计：$(du -sh "$DATA_ROOT/_downloads" 2>/dev/null | cut -f1)"
    else
        warn "还没有 $DATA_ROOT/_downloads —— data 这步没开始"
    fi
    if [ -d "$DATASETS" ]; then
        echo "    解压产物（大量小文件，可能要几十秒）："
        du -sh "$DATASETS"/* 2>/dev/null | sed 's/^/      /'
    else
        warn "还没有 $DATASETS"
    fi

    hr "2. 权重：CLIP 到手没有（决定指标能不能进论文）"
    if [ -f pretrained/clip-vit-base-patch16/config.json ]; then
        ok "路线 A：真 CLIP-ViT-B/16 已在本地目录"
        ls -la pretrained/clip-vit-base-patch16 | sed 's/^/    /'
    else
        warn "没有 pretrained/clip-vit-base-patch16/ → 还需要跑 clip"
    fi
    [ -f pretrained/vit_b16_imagenet.pt ] && \
        echo "    （另有兜底权重 pretrained/vit_b16_imagenet.pt）"

    hr "3. 依赖：torch 与 transformers 是否互相认识"
    $PY - <<'PYEOF'
import sys
try:
    import torch, transformers
except Exception as e:                      # noqa: BLE001
    print(f"[!] 缺包：{type(e).__name__}: {e}"); sys.exit(0)
print(f"    torch {torch.__version__} | cuda={torch.version.cuda}")
print(f"    transformers {transformers.__version__} | "
      f"is_torch_available={transformers.is_torch_available()}")
try:
    from transformers import CLIPVisionModel  # noqa: F401
    print("[OK] CLIPVisionModel 可导入 —— clip 这步不会再假失败")
except Exception as e:                      # noqa: BLE001
    print(f"[!] CLIPVisionModel 导入失败：{type(e).__name__}: {e}")
    print("    → 这通常是 torch 与 transformers 版本互斥，跟网络无关"
          "（见 docs/08 D38）；先按 install 的提示修它，别去换源")
PYEOF

    hr "4. 骨干真伪"
    $PY scripts/check_backbone.py 2>&1 | tail -14 || warn "check_backbone 未通过"

    hr "5. 结论：下一步跑哪个 stage"
    local s_install=0 s_clip=0 s_data=0
    $PY -c "import transformers as t,sys; sys.exit(0 if t.is_torch_available() else 1)" \
        >/dev/null 2>&1 && s_install=1
    [ -f pretrained/clip-vit-base-patch16/config.json ] && s_clip=1
    { [ -d "$DATASETS/ForenSynths/test/progan" ] && [ -d "$DATASETS/CASIAv2" ]; } \
        2>/dev/null && s_data=1
    # 训练集单独判：ForenSynths 少了 train/ 这一层时，14.4 万张训练图明明在盘上
    # 却一条都扫不到。status 若还把人推向 verify，等于把这个坑藏起来。
    local s_train=0
    for c in $CLASSES; do
        if [ -d "$DATASETS/ForenSynths/train/$c/0_real" ] \
           || [ -d "$DATASETS/ForenSynths/train/$c/1_fake" ]; then
            s_train=$((s_train + 1))
        fi
    done
    if   [ "$s_install" = 0 ]; then echo "  → bash scripts/cloud_autodl.sh install"
    elif [ "$s_clip"    = 0 ]; then echo "  → bash scripts/cloud_autodl.sh clip"
    elif [ "$s_data"    = 0 ]; then echo "  → bash scripts/cloud_autodl.sh data"
    elif [ "$s_train"   = 0 ]; then
        printf '\033[1;33m  → ⚠ 训练集布局不对：%s/ForenSynths/train/<类>/ 不存在\033[0m\n' "$DATASETS"
        echo "     数据在盘上但少了一层（**不用重下 70GB**，同盘 mv 是瞬时的）："
        echo "       cd $DATASETS/ForenSynths && mkdir -p train && mv $CLASSES train/"
        echo "     然后重跑： bash scripts/cloud_autodl.sh verify"
    else
        echo "  → bash scripts/cloud_autodl.sh verify"
        echo "     （verify 全绿之后再切【有卡模式】跑 probe，别在无卡模式下烧 probe）"
    fi
}

# ============================================================================
case "$STAGE" in
    doctor)  stage_doctor ;;
    status)  stage_status ;;
    install) stage_install ;;
    clip)    stage_clip ;;
    data)    stage_data ;;
    verify)  stage_verify ;;
    probe)   stage_probe ;;
    train)   stage_train ;;
    pack)    stage_pack ;;
    *)       sed -n '2,35p' "$0"; exit 2 ;;
esac
