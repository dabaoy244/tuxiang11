# 15 · 云端 GPU 完整训练与评测手册（照抄执行版）

> **这份手册解决一件事：本项目在 CPU 上永远做不完的那部分工作，怎么在云 GPU 上一次跑完。**
>
> 判据：申报书 6.3 的四条硬指标里，**「CASIA V2 篡改定位 mIoU ≥ 56%」必须靠三阶段正式训练**，
> 而三阶段训练（20+20+30 轮）在 7 核 CPU 上的实测速度是 **0.89 秒/样本**（b16），
> 按 14.4 万张训练集算，单轮就要 **35 小时**，70 轮约 **100 天** —— 不可能。
>
> **所以：CPU 用来"验证流程正确"，GPU 用来"跑出数字"。分工不能错。**
>
> 本文所有命令都已对照 `argparse` 逐个核过，可直接复制粘贴。
> 本文所有"已实测"结论都带原始文件路径，答辩时可当场翻出来。

---

## 0. 先看结论

| 你要的东西 | 在哪儿跑 | 大概要多久 |
|---|---|---|
| 数据下载（90 GB） | **本地或云端都行**，推荐云端 | 数小时（挂夜） |
| 分类支路能不能训起来 | **本地已做完**（见 §1） | ✅ 已完成 |
| 主实验 ACC / AP | 云 GPU，4 卡时～35 卡时 | 见 §6 |
| 定位 mIoU ≥ 56% | 云 GPU，**必须** | 见 §7 |
| 13 生成器泛化表 | 云 GPU | 见 §8 |
| 鲁棒性表 | 云 GPU | 见 §9 |
| 消融 4 项 | 云 GPU | 见 §10 |
| 体积 / 速度两项工程指标 | **本地已实测**（见 §1.4） | ✅ 已完成 |

---

## 0.1 平台已定：AutoDL（2026-09-22 确认）

> `docs/10` §6 里"云平台是哪家"这一项**已解决**，本章把它落成可执行清单。
>
> ✅ **实例已经开好了？** 直接看 **`docs/18_AutoDL上云执行清单.md`** —— 开服即用的分步清单、
> 每阶段该看哪个数字、以及 `scripts/cloud_autodl.sh` 八个 stage 的一键命令。
> 本章保留**平台选型的依据**与成本核算（为什么是 4090、为什么数据盘是硬闸门）。

### 选卡：RTX 4090

| 卡 | 显存 | 单精 | 时租（官方公示价 2026.09） | 判断 |
|---|---|---|---|---|
| RTX 3090 | 24GB | 35.6 TFLOPS | ¥1.32/时 | ✅ 够用，最便宜 |
| **RTX 4090** | **24GB** | **82.6 TFLOPS** | **¥1.88/时** | ✅✅ **首选**（每元算力最高） |
| V100 | 32GB | 15.7 TFLOPS | ¥1.88/时 | ❌ Volta 老架构、**不支持 bf16** |
| RTX 5090 | 32GB | 104.8 TFLOPS | ¥2.78/时 | ✅ 4090 抢不到时的备选 |
| RTX 6000D | 84GB | — | ¥6.80/时 | ❌ **贵 3.6 倍，本项目用不上** |

**显存需求核算**：b16 全模型 90.5M 参数，输入 224×224、batch 32 →
**24GB 足够且有余量**（16GB 需降 batch 或开梯度检查点）。
"单卡 ≥16 GB"这个下限（`docs/10` 主线 B1）在此满足。

**为什么不该用贵的卡**：本项目是 90M 参数的小模型 + 7 万张图量级的数据，
瓶颈在"跑完 70 个 epoch"，**不在单步算力**。换成 ¥6.80 的卡不会让 70 轮
从 30 小时变成 8 小时，只会让账单变成 3.6 倍。

### ⚠️ 比选卡更要紧的一件事：数据盘

AutoDL 默认给 **50GB 免费数据盘**（挂载在 `/root/autodl-tmp`），系统盘 30GB。
但本项目的数据规模是：

| 项目 | 体积 |
|---|---|
| `progan_train` 7 卷（压缩包） | **74.9 GB** |
| 解压产物（全 20 类 · 720119 张） | **69.8 GB** |
| 解压产物（只 4 类 · 144024 张） | 约 **14 GB** |
| `CNN_synth_testset`（压缩包 / 解压后） | 20.1 GB / 约 20 GB |
| CASIA v2.0 | 3.3 GB |
| 预训练权重 | 0.35 GB |

→ **50GB 装不下。** 全 20 类路径的峰值约 **145 GB**（7 卷与解压产物同时在盘上）。

**必须做两件事：**

1. **挑「可扩容」不为 0 的主机。** 租用页每台机器都标着
   `数据盘 50GB，可扩容 X GB`。**若显示 `可扩容 0 GB`，这台机器永远扩不了，直接跳过**
   —— 这是比显卡型号更硬的否决项。
2. **扩容到 150 GB（走 4 类）或 250 GB（走 20 类全量）。**
   按量计费实例的数据盘按**当天使用的最高容量**在 24 点计费，价格约
   **¥0.0065/日/GB**（会员价）→ 多 200GB ≈ **¥1.3/天 ≈ ¥39/月**，相比 GPU 时租可忽略。
   扩容/缩容在租用后也可操作（**以租用页显示为准**）。

> 💡 **省磁盘的关键一行**：`configs/default.yaml` 里 `max_samples: 40000` ——
> **无论盘上是 4 类（14.4 万张）还是 20 类（72 万张），训练实际只取 40000 张。**
> 所以"只解 4 类"能省下约 56GB 磁盘和数小时解压时间，且**不减小训练规模**
> （代价是真实图像的**对象类别多样性**下降，论文里注明训练集范围即可 ——
> 这正是 `docs/10` §5 已定的止损规则）。
>
> 配合"解压成功后自动删卷"的三道闸门，峰值磁盘从 145GB 降到约 **89 GB**：
> ```bash
> python scripts/fetch_datasets.py --split train --classes car cat chair horse
> ```

### 开实例的正确姿势

**① 镜像与依赖**
选官方 `PyTorch 2.x + Python 3.10 + CUDA 12.1` 镜像。`requirements.txt` 第 6–8 行
已注明 GPU 版 torch 必须单独装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# 装完立刻验，别让 CPU 版 torch 蒙混过关：
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**② 先用「无卡模式」把所有不烧算力的活做完**（¥0.1/时）
传代码、装环境、下数据、跑数据体检 —— 这一步能省掉 90% 以上的云费用。
> 无卡模式有区域限制（部分区仅 2核4G），**同一账号同时只能开一个无卡实例**。

**③ 数据不要从本地上传**，在云端重下（见 §2.3）。上传 90GB 比云端下载慢得多。
AutoDL 控制台自带「学术资源加速」（HuggingFace / GitHub），
`fetch_datasets.py` 的 `--mirror auto`（hf → hf-mirror）也可用。

**④ 数据一律放 `/root/autodl-tmp`**（数据盘），不要放系统盘。
同地区文件存储 `/root/autodl-fs`（20GB 内免费）可用于跨实例传权重与 checkpoint。

**⑤ 抢卡时机与"卡不保留"**
4090 在晚间高峰很难抢，**凌晨和工作日白天成功率高**。
按量计费**关机后不保留这张卡**，第二天开机原主机可能已无空卡 ——
所以**一旦开跑就连续跑完**，并让训练结束自动关机：

```bash
nohup bash -c "python scripts/train.py --config configs/default.yaml; /usr/bin/shutdown" > train.log 2>&1 &
```

**⑥ 释放实例前**：把 checkpoint、`outputs/*.json`、日志打包传回本地或 `/root/autodl-fs`。
连续关机太久实例可能被释放并清空数据。

### 成本总账（RTX 4090 @ ¥1.88/时）

| 项 | 量 | 费用 |
|---|---|---|
| 三阶段正式训练 | 23–35 卡时 | ¥43–66 |
| 消融 4 项 | 16–28 卡时 | ¥30–53 |
| 定位（CASIA v2） | 3–5 卡时 | ¥6–9 |
| 泛化 + 鲁棒性 | 2–4 卡时 | ¥4–8 |
| 数据盘 250GB | 1 个月 | 约 ¥39 |
| 无卡模式（传数据 / 装环境） | 约 20 小时 | ¥2 |
| **合计** | **75–120 卡时** | **约 ¥125–180** |

> 与 RTX 3090（¥1.32/时）对比：卡时单价低约 30%，但单卡慢约 2.3 倍，
> 总价相近，**4090 省的是你的时间**。

---

## 1. 上云前，本地已经证明了什么

**为什么要先看这一节**：上云前必须排掉"代码是错的"这类风险，否则会白烧卡时。
下面每一条都有原始文件可查。

### 1.1 全流程冒烟测试通过

```
outputs/smoke_after_tamperfix.txt    30/30 项通过
```

覆盖：模型构建、三阶段切换、损失（含 VIB KL 与不确定性加权）、
ONNX 导出、INT8 量化、OpenVINO IR、推理一致性。

### 1.2 分类支路"训不动"的根因已定位：**缺预训练权重**

早期训练（**s16 骨干、随机初始化**）呈现同一现象，且**换配置、换学习率、换轮数都不改变**：

| 现象 | 实测值 |
|---|---|
| 验证集准确率 | 恒等于 `0.5`（二分类 = 瞎猜） |
| AUC | 在 `0.49 ~ 0.52` 游走（≈随机排序） |
| 判决方向 | 在两个极端之间翻转：epoch2 全判真（`tn=800, tp=0`），epoch3 全判假（`tn=0, tp=800`） |

原始日志：`outputs/realval_s16_random_train_log.txt`

在 8 轮的小样本探针里同样恒定：`acc=0.40625`（= 13/32，始终判正）、
`auc` 全程 `0.446 ~ 0.510`。原始日志：`outputs/run_realval/history_probe_limitbatches.json`

根因：**骨干随机初始化**。代码本身没有 bug —— 判据是"损失在下降但判决边界不建立"，
这是典型的"特征提取器没能力"而非"损失函数写错"。

修复动作：写了 `scripts/fetch_pretrained_backbone.py`，
从 `download.pytorch.org` 取 ImageNet-1K 预训练 ViT-B/16（346 MB），
按本仓库兜底骨干的键名转换后落盘为 `pretrained/vit_b16_imagenet.pt`。
加载时用 `strict=True`（键不匹配直接抛错），启动日志会打印：

```
[CLIPViTBackbone] ✅ 已加载预训练骨干权重点 .../vit_b16_imagenet.pt （150 个张量）
```

**受控对照实验**（random vs pretrained，唯一变量 = 骨干权重）见
`docs/14_对照实验_预训练权重消融.md`，由 `scripts/run_ablation.py` 自动生成。

**⚡ 对照实验已完成（2026-09-18，两臂各 3 个 epoch）**

| 指标 | 随机初始化臂 | **ImageNet 预训练臂** |
|---|---|---|
| 最佳 val_AUC | 0.6428 | **0.6809** |
| 最佳 val_AP | 0.6463 | **0.7009** |
| 最佳 val_acc | 0.5000（恒等） | 0.5731 |
| `tn` / `tp` | **0 / 650 —— 退化** | **163 / 582 —— 有效** |

> **关键不是 AUC 差了多少（只差 +0.038），而是 `tn` 是 0 还是非 0。**
> 随机初始化臂的 `tn=0`：它把全部样本都判成"伪造"，判决边界**根本不存在**；
> 预训练臂 `tn=163 / tp=582`：两类都有被正确判出的样本，边界**建立起来了**。
> 这是"缺预训练权重"这一根因的直接证据。
>
> ⚠️ **但必须如实说明这个对照的强度边界**（答辩时会被追问）：
> ① 随机初始化臂的 `AUC=0.64` 说明它的**排序分数并非全无信号** —— 一个合理解释是
> 频域分支（2D-DFT）是**确定性变换**，输出与骨干是否预训练无关，分类头仍能从里面
> 读到一点痕迹；② 所以结论应表述为 **"预训练权重是判决边界能建立的必要条件
> （本机 CPU 小规模设定下），但远非充分条件"** —— 预训练臂 AUC 0.68 距申报指标
> （ACC≥91%）仍很远，那需要完整三阶段训练 + 70 GB 训练集 + GPU。
> ③ 注意区分两个随机初始化实验：**s16 随机**（AUC 0.49~0.52，两端翻转）与
> **b16 随机**（AUC 0.63，恒判假）不是一回事，不能混为一条结论。
>
> 完整两臂、逐 epoch 对比见 `docs/14`（由 `scripts/run_ablation.py` 自动生成）。

### 1.3 数据规模已核实（不是估算）

| 数据集 | 核实值 | 用途 | 状态 |
|---|---|---|---|
| ForenSynths `test` | **90329 张 / 13 生成器**（官方 90310） | 跨生成器泛化（论文核心表） | ✅ 完整 |
| ForenSynths `val` | 20 类 ProGAN 保留集 8000 张 | 小规模替代训练集 | ✅ 完整 |
| ForenSynths `train` | `progan_train.7z.001~007`，**74.9 GB**（20 类 720119 张，解压后 **69.8 GB**） | 正式训练集 | ✅ **下载已完成，并通过结构级只读验证**：7 卷字节数与远端逐字节相等；`--list-train` 在真实分卷上成功解析出 720119 条目 / 20 类。<br>⚠️ **本机不再解压**（D 盘只剩 44 GB < 69.8 GB）——**上传 7 卷到云端，在实例上解压**；解压走 `SplitVolumeReader` 分卷直读，零中间产物，峰值 69.8 GB 而非 149.8 GB |
| CASIA v2.0 | **7491 真实 + 5123 篡改 + 5123 掩码 = 17737 文件 / 3.28 GB** | 定位 mIoU | ✅ **已下载并整理**：train 10090 / val 1261 / test 1263 |
| COVERAGE | 100 对（复制-移动） | 定位补充 | ❌ 未取得（只在 GitHub，本机不可达） |

CASIA 的数据源是 ModelScope 的勘误版 `Sunnyhaze/CASIAv2-Manipulated-image`，
下载器 `scripts/fetch_tamper_datasets.py` 已处理**分页、续传、`.part` 原子落盘、
自动整理成训练布局**。实测 **17737/17737 文件、失败 0、11.1 分钟**（16 线程），
原始计数与官方公布数完全一致。掩码管线已用真实数据验证：
**4098 张篡改图 ↔ 4098 个不同掩码**（修复前会塌缩成 1 个）、5992 张真实图全部载入。

> ⚠️ **上云前必须先跑一次数据体检。** `build_dataloaders` 对空数据集是**静默丢弃**的，
> 所以"配置里写了 COVERAGE"不等于"COVERAGE 参与了训练"。启动时会打印：
>
> ```
> [data] split=train 共 68128 条 ← ForenSynths(train) 40000 条、CASIAv2(train) 6000 条
> [data] ⚠ split=train：配置了 1 个数据集但一个样本都没有，已跳过 —— ['COVERAGE(train)']
> ```
>
> **看到 `⚠` 就必须处理**（补数据 or 从配置里删掉），否则论文/结题报告里写的
> "训练集包含 COVERAGE"就是不实陈述。同一份数据集的构成也会写进 `outputs/eval_*.json`。

### 1.4 两项工程指标已实测达标

实测文件：`outputs/cpu_benchmark.json`、`outputs/cpu_benchmark_b16.json`、
`outputs/b16_int8_report.json`。测试机：7 核 Intel、ONNX Runtime 1.30.0、224×224。

| 模型 | 体积 | 中位延迟 | 中位 FPS | ≤120 MB | ≥8 FPS |
|---|---|---|---|---|---|
| `lite.onnx`（s16 FP32） | 99.2 MB | 137.7 ms | 7.26 | ✅ | ❌ |
| **`lite_simplified.onnx`（s16 + 图简化）** | **98.7 MB** | **105.7 ms** | **9.46** | ✅ | ✅ |
| `lite_simplified.onnx`（另一轮复测） | 98.7 MB | 89.2 ms | 11.22 | ✅ | ✅ |
| `lite_int8.onnx` | 25.7 MB | 298.6 ms | 3.35 | ✅ | ❌ |
| `b16.onnx`（FP32） | 346.3 MB | — | — | ❌ | — |
| `b16_int8.onnx` | 92.9 MB | 349.1 ms | 2.86 | ✅ | ❌ |

**两条必须写进论文的结论**：

1. **"用 CLIP-ViT-B/16 + ≤120 MB"这个组合在 FP32 下不可能同时成立**
   （b16 全模型 362.1 MB）。要同时满足，只能把骨干降到 s16 及以下，或做 INT8 量化。
2. **本机 INT8 反而更慢**（3.35 FPS < 9.46 FPS）。原因是这颗 CPU 没有 VNNI 指令集，
   ONNX Runtime 的 INT8 量化算子退化成慢路径。**所以"INT8 一定加速"是错的**，
   必须实测；换机器必须重测。

> ⚠️ 这两条是**架构级结论，与是否加载权重无关**，所以现在就能写进论文。

### 1.5 定位支路的两个静默 bug 已修复并验证

**这两个 bug 会让 mIoU 永远不可能达标，而且不报任何错**：

| 编号 | 症状 | 根因 | 验证 |
|---|---|---|---|
| C14 | 5123 个掩码**全部塌缩成一个键**，每张篡改图匹配到同一张（错误的）掩码 | 掩码索引用 `stem.split("_")[0]`，取第一个下划线前永远是 `"Tp"` | 见 `datasets.py` 内注释 |
| C15 | **7491 张真实图被整个跳过**，数据集里只剩篡改图 | 同上，真实图键 `"au"` 在掩码集里不存在 → 被判为"缺掩码"跳过 | 同上 |
| C16 | 整理后的 CASIA 目录（`Tp/`+`Au/`+`Gt/`）**一张图都读不到** | 目录探测只认一种布局 | 合成数据 6 条（3 真 + 3 篡改，3 个不同掩码）通过 |

修复后 `TamperDataset` 支持三种布局（已整理 / 三分目录 / 原始平铺），
并在加载时打印真实条数，例如：

```
[TamperDataset] CASIAv2/train: 载入 10124 条（篡改 5123 / 真实 5001，掩码 5123 个）
```

> **上云后第一件事就是核对这行数字。** 如果"掩码 N 个"远小于"篡改 N 张"，
> 说明掩码目录不完整，mIoU 必然偏 —— 不要往下跑。

---

## 2. 要同步什么到云端

### 2.1 代码（小，直接传）

```bash
# 在本地项目根目录执行（传之前先排除大目录）
cd /d/picture/vibnet-forgery-detector
tar -czf vibnet_code.tar.gz \
    --exclude='data' --exclude='outputs' --exclude='checkpoints' \
    --exclude='pretrained/_raw' --exclude='__pycache__' --exclude='.git' \
    src scripts configs docs app deploy tools requirements.txt README.md
# 体积约几 MB（deploy/*.onnx 若不想传可再加 --exclude='deploy'）
```

### 2.2 预训练权重（346 MB，**必须传**）

```bash
scp pretrained/vit_b16_imagenet.pt  <user>@<host>:<path>/pretrained/
```

> 不要指望在云端重新下载 —— `download.pytorch.org` 在境内部分机房也不通。
> 直接传是最省事的。`pretrained/_raw/` 是中间产物，不用传。

### 2.3 数据：**建议在云端重新下**，不要传

90 GB 上传比下载慢得多。云端下载命令：

```bash
# ForenSynths 全量（约 90 GB）
python scripts/fetch_datasets.py --split val progan_test          # 1.6 GB，先跑
python scripts/fetch_datasets.py --split test                     # 18.7 GB，泛化表
python scripts/fetch_datasets.py --split train --no-extract       # 70.4 GB，挂夜

# 只想先跑通 4 类子集（省磁盘、省时间，论文里注明范围即可）
python scripts/fetch_datasets.py --split train --classes car cat chair horse

# CASIA v2.0（3.28 GB，定位指标必需）
python scripts/fetch_tamper_datasets.py --workers 16
```

`fetch_datasets.py` 已内置**多镜像自动切换**（`huggingface.co` ↔ `hf-mirror.com`）
与**指数退避重试**（15 s → 300 s，12 次），`--mirror auto` 为默认值。
这是被真实故障逼出来的：早期单镜像下载在 `huggingface.co` 被间歇性阻断时，
7 个分卷全部以 `SSL: UNEXPECTED_EOF_WHILE_READING` 失败。

**下载任务卡住时怎么判断是死是活**：

```bash
python scripts/check_status.py            # 三证据交叉判定（进程/日志新鲜度/进度推进）
python scripts/check_status.py --watch --interval 30
```

> 提醒：本机（Git Bash 包装层）每执行一条 bash 命令都会多打印一行
> `[ERROR:...crashpad...] CreateFile: 系统找不到指定的文件。(0x2)`，
> **那是噪音，连 `ls` 都会打印，与脚本成败无关**。判定任务死活用 `check_status.py`。

### 2.4 磁盘预算

| 阶段 | 峰值占用 |
|---|---|
| 下载压缩包 | 74.9 GB（train 7 卷）+ 20.1 GB（test）+ 3.3 GB（CASIA） |
| 解压后 | ProGAN train **69.8 GB**（20 类）/ **约 14 GB**（4 类）、test 约 20 GB、val 约 0.8 GB |
| **峰值合计** | **约 145 GB**（走 20 类全量）；**约 89 GB**（只解 4 类，删卷后） |
| 训练 checkpoint | b16 每份 362 MB，最多保留 3 份 |

**AutoDL 上至少要 150 GB 数据盘（推荐 250 GB）。**
关键省钱动作：用 `--classes` 只解 4 类，并让默认的"解压成功后删卷"把峰值压到 89 GB。
见 §0.1 的磁盘核算与扩容价格。

---

## 3. 环境与依赖

```bash
# 建议镜像：CUDA 12.1+ 的 PyTorch 官方镜像
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 依赖
pip install -r requirements.txt
# 若云端能访问 HuggingFace，强烈建议装 transformers，用于加载真正的 CLIP 权重
pip install transformers
```

**关键：检查 `transformers` 是否装上**——它决定 §5 走哪条路线。

---

## 4. 数据准备与校验

### 4.1 目录规范（代码就是这么认的）

```
data/Datasets/
├── ForenSynths/                       # 整图真伪（无像素掩码）
│   ├── train/<class>/{0_real,1_fake}/*.png|jpg      # 20 类 LSUN 真实 + ProGAN 生成
│   ├── val/<class>/{0_real,1_fake}/*                # 官方 held-out ProGAN
│   └── test/<generator>/{0_real,1_fake}/*           # 13 种生成器
│       （progan/cyclegan/stylegan/stylegan2 是两级：<generator>/<class>/{0_real,1_fake}）
├── CASIAv2/                           # 拼接·复制移动·修图（有像素掩码）
│   └── {train,val,test}/{image,mask}/   # fetch_tamper_datasets.py 整理后的布局
└── COVERAGE/
    └── {train,val,test}/{image,mask}/
```

`GenSynthsDataset` 会**递归探测任意深度**的 `0_real`/`1_fake` 配对，所以三种官方层级都能吃；
`TamperDataset` 支持 **已整理 / 三分目录 / 原始平铺** 三种布局。

### 4.2 校验（上云后第一件事）

```bash
# 结构 + 计数
python scripts/prepare_datasets.py --root data/Datasets --check

# 逐生成器清点（顺便看哪个生成器样本少，抽样时要分层）
python scripts/eval_cross_generator.py --list-only

# 确认 CASIA 掩码数 == 篡改图数（不等就别往下跑）
python -c "
import sys; sys.path.insert(0,'.')
from src.data.datasets import TamperDataset
for sp in ('train','val','test'):
    TamperDataset('data/Datasets','CASIAv2',split=sp,max_samples=None)
"
```

---

## 5. 骨干权重：决定论文怎么写

代码的加载优先级（`src/models/spatial_branch.py::CLIPViTBackbone`）：

```
① 本地目录 clip_local_dir 里的 CLIP 权重        → 最理想
② HF 名称 openai/clip-vit-base-patch16（需 VIB_NET_ALLOW_DOWNLOAD=1）→ 次理想
③ 都拿不到 → 退回同构轻量骨干，再装 backbone_weights（ImageNet ViT-B/16）→ 兜底
```

### 路线 A（推荐）：真 CLIP 权重

```bash
export VIB_NET_ALLOW_DOWNLOAD=1
python -c "
from transformers import CLIPVisionModel
m = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
m.save_pretrained('pretrained/clip-vit-base-patch16')
print('OK')
"
```

`configs/default.yaml` 里已是 `backbone: "clip_vit_b16"` + `clip_local_dir: "pretrained/clip-vit-base-patch16"`，
**不需要改配置**，直接开训。论文可以如实写"采用 CLIP-ViT-B/16 语义先验"。

### 路线 B（兜底）：ImageNet ViT-B/16

本地已验证可用的替代（`pretrained/vit_b16_imagenet.pt`，150 张量，`strict=True` 装载成功）。
把 `configs/default.yaml` 改成：

```yaml
model:
  spatial:
    backbone: "tiny_vit"            # 跳过 CLIP 查找，直接用同构骨干
    backbone_variant: "b16"         # 必须与权重档位一致，否则 strict 装载会抛错
    backbone_weights: "pretrained/vit_b16_imagenet.pt"
    global_dim: 768
```

> **论文里必须如实说明骨干替换**：CLIP 权重因受限网络未取得，
> 改用 ImageNet-1K 预训练 ViT-B/16。两者同为 ViT-B/16 结构、同为大规模图像预训练，
> 作为对照是成立的，但**不能把结论写成"验证了 CLIP 语义先验的作用"**。

---

## 6. 主实验：三阶段正式训练

### 6.1 命令

```bash
# 先跑 1 个 epoch 实测速度（务必做，别直接上 70 轮）
python -m src.engine.trainer --help 2>/dev/null
python scripts/train.py --config configs/default.yaml --device cuda \
    --workers 8 --tag probe

# 实测单 epoch 耗时后，正式三阶段（20+20+30 = 70 轮）
nohup python -u scripts/train.py --config configs/default.yaml --device cuda \
    --workers 8 --tag vibnet > outputs/gpu_train_log.txt 2>&1 &
```

配置要点（`configs/default.yaml`）：

| 阶段 | 轮数 | lr | 冻结 | 任务 |
|---|---|---|---|---|
| `stage1_cls_pretrain` | 20 | `1e-4 → 1e-6` | `localization` | `cls` |
| `stage2_loc_pretrain` | 20 | `5e-5 → 5e-7` | `vib`, `cls_head` | `loc`, `edge` |
| `stage3_joint_finetune` | 30 | `1e-5 → 1e-7` | 无 | `cls`, `loc`, `edge` |

β 退火跨阶段承接：`stage3` 的 `beta_epoch_offset: 20`，即全局 t=20 起才开始升 β。

### 6.2 监控

```bash
tail -f outputs/gpu_train_log.txt
grep -E "acc=|auc=" outputs/gpu_train_log.txt | tail -20
```

**必须盯的三个数**：

| 指标 | 期望 | 若不达标说明 |
|---|---|---|
| `cls_acc` | 首轮就应 > 0.7，5 轮内 > 0.9 | 骨干权重没加载上（看启动那行 `✅ 已加载`） |
| `cls_tn` / `cls_tp` | 两个都 > 0 | 只看 acc 会被退化解骗过（全判真也能得 0.5） |
| `raw={'vib': ...}` | 稳定在 0.5 ~ 1.5 | 若恒为 0，VIB 被关闭了 |

> **只看 `cls_acc` 是本项目最容易犯的错。** 早期就是"acc=0.5 看起来还行"，
> 实际 `tn=0`（一张真图都没判对）。**必须同时看 `tn`/`tp`。**

### 6.3 断点续训

```bash
python scripts/train.py --config configs/default.yaml --device cuda \
    --workers 8 --resume checkpoints/vibnet_last.pt --tag vibnet
```

### 6.4 卡时预算（**实测外推法，不要照抄别人的估算**）

CPU 实测：b16 在 batch=8、224×224、7 线程下 **7.1 秒/iter** → **0.89 秒/样本**（前向+反向）。

上云后按这个流程算：

```bash
# 跑 1 个 epoch，从日志里读实际耗时 T1（分钟）
# 三阶段总耗时 ≈ T1 × 70
# 加上评测开销约 +20%
```

参考：单卡 24 GB（如 4090）配合 batch 32 时，b16 的吞吐通常比本机 7 核 CPU 高 **30～80 倍**，
即单 epoch 约 **10～25 分钟**，70 轮约 **12～30 小时**。
> 这是**经验区间，不是实测值**。上云第一件事就是实测 T1，再乘 70。

---

## 7. 定位任务：CASIA v2 → **mIoU ≥ 56%**

**这是四条硬指标里唯一还没测过、且只能靠 GPU 的一项。**

定位支路在 `stage2_loc_pretrain` 与 `stage3_joint_finetune` 中训练，
任务项是 `loc`（Dice + BCE）与 `edge`（Canny 边缘监督）。

### 7.1 确认配置里的定位数据已挂上

`configs/default.yaml` 的 `data.train_sets` 里应包含：

```yaml
    - name: "CASIAv2"
      kind: "tamper"
      split: "train"
      max_samples: 6000
```

### 7.2 训练后评测

```bash
# ★ 必须加 --dataset CASIAv2：configs/default.yaml 的 test_sets 里 ForenSynths
#   排在前面（max_samples 20000）且 test loader shuffle=False，
#   不加这个参数时 CASIAv2 可能**一批都轮不到**，定位指标静默为空（见 docs/08 D19）。
python -m src.evaluation.evaluate --config configs/default.yaml \
    --ckpt checkpoints/vibnet_best.pt --split test --dataset CASIAv2 --device cuda
```

输出会同时打印**三种 mIoU 口径**，并给出 `pixel_acc` / `dice` / `f1`：

| 口径 | 字段 | 定义 | 怎么用 |
|---|---|---|---|
| ① 池化（最宽松） | `miou` | 全局像素汇总后 `tp/(tp+fp+fn)`，**按像素加权** | 只看训练趋势 |
| ② **仅篡改图（★判定用）** | `miou_tampered_only` | 只在 GT 含篡改区域的图上逐图算 IoU 再平均 | **论文报这个**，与 ManTra-Net / SPAN / CAT-Net 可比 |
| ③ 全部非空图（最严格） | `miou_per_sample` | 含被误报的真实图（记 IoU=0） | 误报代价敏感性分析 |

⚠️ **三个口径能差一倍以上，报哪个必须写明，不要挑最高的那个。**
⚠️ 曾把口径写错成「报 `miou`，口径是每样本算 IoU 再平均」——那是把**池化字段名**
和**逐图平均的定义**混在一起，两者是不同口径（`docs/08` D19 附近）。
判定 `≥56%` 用的是 `miou_tampered_only`；三种口径的定义由
`python scripts/test_metric_definitions.py` 用手算样例做回归测试钉死，改实现就会报警。

> **另注意**：`checkpoints/abp_best.pt`（对照实验的预训练臂）**只训过 stage1_cls_pretrain**，
> 定位头基本是随机初始化 —— **不要拿它报 mIoU**，它只能证明"定位数据管线是通的"。
> mIoU 必须来自完整三阶段训练后的权重。

### 7.3 若不达标（< 56%）的排查顺序

| 顺序 | 检查项 | 判据 |
|---|---|---|
| 0 | 评测命令是否带了 `--dataset CASIAv2` | 不加时定位指标可能是空的（D19） |
| 0b | `n_tampered` 是否 > 0 | 为 0 说明这批样本里没有篡改图，mIoU 无意义（D18） |
| 1 | 掩码加载数是否 == 篡改图数 | §4.2 的打印行 |
| 2 | `stage2` 是否真的训练了 `localization` | 日志里 `训练=['spatial','freq','fusion','localization']` |
| 3 | 真图是否被算进定位 loss | 真图 `mask_valid=False`，应被过滤；若全被过滤则 loss 是空的 |
| 4 | `encoder_channels` 与输入分辨率是否匹配 | 224×224 → 14×14 特征图 |
| 5 | 是否只训了分类、忘了跑 stage2/3 | 看 `outputs/` 下有几个阶段的记录 |

---

## 8. 跨生成器泛化（论文核心表）

```bash
python scripts/eval_cross_generator.py --config configs/default.yaml \
    --ckpt checkpoints/vibnet_best.pt \
    --data-root data/Datasets/ForenSynths/test \
    --per-class 300 --batch-size 32 --device cuda \
    --out outputs/cross_gen --tag full \
    --scores-out outputs/cross_gen_scores/full.npz      # ★ 存逐样本分数
```

> **★ 一定要加 `--scores-out`。** 推理是这个流程最贵的一步（本机 CPU 上 2600 张
> 约 9 分钟/臂），而指标是最后一步才算。本项目吃过一次亏：两条臂各烧完约 9 分钟推理，
> 在算 AUC 那一行崩掉，**整份跨生成器结果全丢**。
> 存下分数后，改指标定义只需 `python scripts/recompute_cross_gen.py --scores <npz> --out <目录>`
> **秒级重算**（它复用 `eval_cross_generator.summarize()`，口径不会分叉）。
> GPU 上一次推理可能几十分钟，这一条更值得照做。
>
> **指标口径**：AUC 与 AP 只用 `src/engine/metrics.py` 的**唯一实现**
> （AUC 做并列秩平均、AP 按不同分数值分组积分）。两处口径不一致会让主实验表与
> 本表**不可比** —— 详见 `docs/08` 的 D21~D23。

**为什么必须用这个脚本，而不是 `evaluate.py`**：

`evaluate.py` 按 `max_samples` 抽样，而 `_maybe_subsample` **只按标签均衡，不按生成器分层**。
ForenSynths test 里 `stylegan2` 有 15976 张、`crn`/`imle` 各 12764 张，
而 `san` 只有 419 张、`seeingdark` 只有 360 张 —— 按 max_samples 抽，
抽到的几乎全是大集合的图，**小生成器基本不出现**，导致根本报不出"逐生成器精度"。

`eval_cross_generator.py` 改为**按生成器分层采样**，并且默认开启 `--dedup-real`：
ForenSynths 各生成器目录下的 `0_real` 来自**同一批真实图**（文件名相同），
不去重会让同一张真图被重复计入 13 次，把总准确率抬得虚高。

输出：`outputs/cross_gen/cross_generator_full.{md,json}`
（逐生成器 ACC / AUC / AP + 宏平均，宏平均是论文里要报的那个数）。

---

## 9. 鲁棒性实验

```bash
python -m src.evaluation.robustness --config configs/default.yaml \
    --ckpt checkpoints/vibnet_best.pt --device cuda --max-samples 500
```

覆盖扰动（见 `configs/default.yaml` 的 `eval.robustness`）：
JPEG 质量 `[10,20,30,50,70,90]`、缩放 `[0.5,0.75,1.25,1.5,2.0]`、
裁剪 `[0.1~0.5]`、旋转 `[0,90,180,270]`、高斯噪声、椒盐噪声。

---

## 10. 消融实验（**精简到 4 项**）

```bash
# 论文用：每项独立重训 + 同口径评测
python -m src.evaluation.ablation --mode train --config configs/default.yaml \
    --presets full no_grad_stop no_phase no_vib no_cross_attention
```

> **⚠ 不要用 `--mode eval` 出论文数据。**
> `--mode eval` 只是加载 full 的 ckpt、推理时关掉模块，
> 它只能说明"该模块对推理有影响"，**不能说明"该模块对训练/泛化有贡献"** ——
> 审稿人会直接指出这一点。`--mode eval` 只用于快速自检。

**为什么从 7 项砍到 4 项**：单人算力所限（中期检查需说明）。
保留的 4 项对应四个核心创新点：梯度停止层（GS）、相位支路、分层 VIB、CS-CAM 跨注意力。
砍掉的 3 项（`no_learnable_mask` / `no_edge` / 其余）在论文里用"未做，列为未来工作"交代。

---

## 11. ONNX 导出与 CPU 基准（**换机器必须重测**）

```bash
# 1) 导出（注意：导出时会自动把 DFT 从 torch.fft 切到矩阵乘法，
#    因为 aten::fft_fft2 无法导出到 opset 17）
python -m src.deploy.export_onnx --config configs/lite.yaml \
    --ckpt checkpoints/vibnet_best.pt --out deploy/lite.onnx --opset 17
python -m src.deploy.export_onnx --config configs/lite.yaml \
    --ckpt checkpoints/vibnet_best.pt --out deploy/lite_simplified.onnx \
    --opset 17 --simplify

# 2) INT8 量化（可选；本机实测反而更慢，见 §1.4）
python -m src.deploy.quantize_int8 --onnx deploy/lite_simplified.onnx \
    --out deploy/lite_int8.onnx --bench --report outputs/int8_report.json

# 3) OpenVINO IR（可选）
python -m src.deploy.optimize_openvino --onnx deploy/lite_simplified.onnx \
    --out deploy/lite_ir --size-only

# 4) CPU 基准（体积 + FPS）
python scripts/benchmark_cpu.py \
    --models deploy/lite.onnx deploy/lite_simplified.onnx deploy/lite_int8.onnx \
    --reps 3 --runs 20 --out outputs/cpu_benchmark.json
```

> 用哪套配置导出，取决于你要立哪个 flag：
> - 要**同时满足 ≤120 MB 且 ≥8 FPS** → `configs/lite.yaml`（s16）+ `--simplify`；
> - 要**方法先进性**（与文献可比）→ `configs/default.yaml`（b16），
>   但必须接受体积超标，或补一次 INT8 量化把体积压回来（速度不保证）。

---

## 12. 结果归档：数字填到哪张表

跑完后把这些数字回填，论文/结题报告直接用：

| 申报书 6.3 硬指标 | 数据来源 | 归档文件 |
|---|---|---|
| 检测准确率 ACC ≥ 91% | `evaluate.py --split test` | `outputs/eval_*.json` |
| **篡改定位 mIoU ≥ 56%** | `evaluate.py`（CASIA v2 test） | 同上 |
| 模型体积 ≤ 120 MB | `benchmark_cpu.py` | `outputs/cpu_benchmark.json` |
| CPU 推理 ≥ 8 张/秒 | `benchmark_cpu.py` | 同上 |
| 跨生成器泛化 | `eval_cross_generator.py` | `outputs/cross_gen/cross_generator_*.json` |
| 鲁棒性 | `robustness.py` | `outputs/robustness*.json` |
| 消融 | `ablation.py --mode train` | `outputs/ablation/*.json` |
| 骨干权重对照 | `run_ablation.py` | `docs/14_*.md` |

**每次训练必须记下**：命令、配置文件名、随机种子、轮数、卡型、单轮耗时。
答辩老师问"你这个 91% 怎么测的、跑了几轮、什么卡"，要能当场翻出来。

---

## 13. 诚实口径：**不能宣称**什么

这一节是给答辩和论文兜底的。以下四条如果被问到，照实说，比被拆穿好得多。

1. **训练集规模**：若最终用的是 4 类子集（约 14.4 万张）而非 20 类全量（约 72 万张），
   论文必须写明训练集范围。**不能声称"在 ForenSynths 全量上训练"。**
2. **骨干**：若走路线 B（ImageNet ViT-B/16 替代 CLIP），
   必须写明骨干替换及原因。**不能把结论写成"验证了 CLIP 语义先验"。**
3. **对比实验**：DIRE 等基线在现有算力下不可复现，
   基线数字**引用原论文公开报告值并注明来源与测试设置**（中文核心期刊完全可接受）。
   **不能声称"我们复现了全部基线"。**
4. **INT8 加速**：本机实测 INT8 更慢（无 VNNI）。**不能笼统写"INT8 量化带来加速"**，
   必须写"在具备 VNNI 的平台上 INT8 预期加速，本测试机未观察到加速"。
5. ★ **不要拿"在 ForenSynths 上训的分类器"去判传统篡改（PS 拼接/复制移动）。**
   已实测：对照实验的预训练臂权重（只在 ProGAN 生成式伪造上训过）在
   **CASIA v2 的 320 张上 `cls AUC = 0.4920`、`ACC = 0.5125` —— 接近随机**；
   定位三个口径全为 0（该权重只训过 stage1，定位头未训练）。
   这说明两件事：① 生成式伪造与传统篡改的判别线索**不共享**，
   要同时覆盖就必须像 `default.yaml` 那样把 `ForenSynths` + `CASIAv2` 一起放进
   `train_sets`；② **演示/答辩时要用对应的数据** —— 用 GAN/扩散生成图演示检测，
   用 CASIA 演示定位，不要交叉使用。
   **不能声称"模型对各类图像篡改都有效"。**
6. ★ **不要拿"本机 CPU 小规模 stage1 权重"的跨生成器数字去对标申报指标。**
   已实测：对照实验的预训练臂（`abp_best.pt`，只训了 stage1 分类预训练，
   CPU + 小规模数据）在 **ForenSynths test 的 13 个生成器 × 各 100 真/100 假**上：

   | | AUC | ACC |
   |---|---|---|
   | 宏平均（逐生成器平均） | **0.6076** | 0.5412 |
   | 全局（合并全部样本） | **0.5840** | 0.5412 |

   逐生成器差异极大：`stylegan2` 0.8297、`imle` 0.8210、`crn` 0.7894 尚可，
   但 **`progan` 只有 0.5279**（经典基准，接近随机）、
   `whichfaceisreal` 0.4665（**低于随机**）、`biggan` 0.4973、`deepfake` 0.5052。

   → 正确表述：**"训练流程与评测链路已跑通，本机小规模设定下的跨生成器泛化能力
   仍很弱（AUC≈0.58~0.61）；申报指标（ACC≥91%）需要完整三阶段训练 +
   官方 70 GB 训练集 + GPU。"** 这份数字的价值是**证明流程可用**与**暴露泛化缺口**，
   不是证明方法有效。**不能宣称"已达到跨生成器泛化要求"。**

---

## 14. 故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动日志没有 `✅ 已加载预训练骨干权重点` | `backbone_weights` 路径不对 / 档位不匹配 | 检查路径；`strict=True` 会直接抛错并列出不匹配的键 |
| `骨干权重键不匹配` 抛错 | `backbone_variant` 与权重档位不一致 | b16 权重必须配 `backbone_variant: b16` |
| `cls_acc` 卡在 0.5、`tn` 或 `tp` 为 0 | 骨干随机初始化（权重没加载上） | 见 `docs/14`；先跑对照实验确认 |
| `CLIPViTBackbone 未取得 CLIP 预训练权重` | 本地无 CLIP 且未开联网 | `export VIB_NET_ALLOW_DOWNLOAD=1`，或走路线 B |
| 下载报 `SSL: UNEXPECTED_EOF_WHILE_READING` | 镜像被间歇性阻断 | 已内置多镜像自动切换 + 指数退避，重跑即可续传 |
| 下载日志"很久没动" | 可能是缓冲，也可能真挂了 | `python scripts/check_status.py` 三证据交叉判定 |
| 报告说"该 split 没有产生任何像素级样本" | 混了无掩码的数据集，或带掩码的排在后面没轮到 | 加 `--dataset CASIAv2`；报告里的 `loc_absent_reason` 会说明是哪种 |
| `n_tampered = 0` 但 `n_loc_samples > 0` | 抽到的全是真实图（样本列表真图在前） | 已在 `_interleave_labels()` 修掉；若仍出现说明用的是旧代码 |
| 三个 mIoU 口径全是 `0.0000` | 预测掩码全空（定位头没训）或 `n_tampered=0` | 先看 `n_tampered`；若 >0 则确认权重真的训过 stage2/3 |
| 日志出现 `⚠ 配置了 k 个数据集但一个样本都没有` | 该数据集没下载/目录空，被静默跳过 | 补数据或从配置删掉；**不要**在论文里声称用了它 |
| `AttributeError: 'NoneType' object has no attribute 'splitlines'` | `subprocess(text=True)` 在中文 Windows 上解码失败 | 已修（见 `docs/08` C21）；改任何子进程解析代码都按字节取回 |
| 定位 mIoU 异常低 | 掩码未加载全（C14/C15 类问题） | 核对 `载入 N 条（篡改 N / 真实 N，掩码 N 个）` |
| ONNX 导出报 `fft_fft2` 不支持 | DFT 实现方式 | 导出会自动切 `matmul`；若仍失败，确认 `dft_mode` 未被手改 |
| `taskkill //F //PID` 无效 | Git Bash 会把 `//F` 当路径改写 | 用 `taskkill /F /PID`，或 PowerShell 的 `Stop-Process -Id` |
| 每条 bash 命令都打印 crashpad `CreateFile: 0x2` | Git Bash 包装层噪音 | **忽略**，连 `ls` 都会打印 |

---

## 15. 相关文件

| 文件 | 作用 |
|---|---|
| `scripts/run_ablation.py` | **新增**：骨干权重受控对照实验（训练+评测+出报告） |
| `scripts/finalize_ablation.py` | **新增**：等对照实验结束 → 重出报告 → 把预训练臂权重提升为默认权重（幂等，可反复执行） |
| `scripts/make_ablation_configs.py` | **新增**：从同一模板生成两臂配置（除标签/路径外只有骨干权重不同） |
| `scripts/fetch_pretrained_backbone.py` | **新增**：ImageNet ViT-B/16 下载与键名转换 |
| `scripts/fetch_tamper_datasets.py` | **新增**：CASIA v2.0 下载与布局整理 |
| `scripts/check_status.py` | **新增**：后台任务三证据存活判定 |
| `scripts/eval_cross_generator.py` | **新增**：按生成器分层采样的泛化评测（`--scores-out` 存逐样本分数） |
| `scripts/recompute_cross_gen.py` | **新增**：从已存盘分数**免推理重算**跨生成器指标（改度量后必用） |
| `scripts/test_metric_definitions.py` | **新增**：指标口径回归测试（三种 mIoU + AUC 并列校正 + AP 分组 + 跨入口一致） |
| `docs/14_对照实验_预训练权重消融.md` | **新增**：预训练权重对照实验报告（自动生成） |
| `docs/09_指标可达性与骨干选型.md` | 体积-精度权衡实测分析 |
| `docs/08_风险与坑清单.md` | 工程坑与规避（C14~C23、D9~D23） |
| `docs/10_剩余工作与单人执行路线图.md` | 单人时间表与取舍规则（§0 有状态更新） |
