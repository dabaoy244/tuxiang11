# 18 · AutoDL 上云执行清单（开服即用版）

> 配套读本：`docs/15_云端GPU完整训练与评测手册.md`（平台无关的完整流程）。
> 本文只回答一个问题：**实例已经开好了，现在这一分钟该做什么。**
>
> 🔰 **还不知道怎么登进服务器？先看 `docs/19_云服务器怎么用_从零上手.md`**
> —— 三种入口怎么点、密码怎么粘、文件怎么传、任务怎么跑不死，都在那一页。

---

## 0. 先读懂你截图上那条实例

| 项 | 截图里的值 | 判断 |
|---|---|---|
| 实例 | 内蒙B区 / 089机，**RTX 4090 × 1 卡** | ✅ 卡选对了（`docs/15` §0.1 的结论） |
| 状态 | 运行中，**CPU 0% / 内存 0%** | 🔴 **正在空转烧钱**：¥1.88/时 × 什么都没干 |
| 系统盘 | 0.17% | — |
| 数据盘 | **0.00%** | ❓ 容量与"可扩容上限"**未知** —— 这是全流程最硬的闸门 |
| 计费 | 按量计费，**余额不足 24 小时** | 🟠 先充值，否则跑到一半被关机 |
| 释放 | 关机 15 天后释放 | ✅ 够用，但别拖过 15 天 |

一句话：**卡对、盘未知、钱要补、当下在漏钱。**

---

## 1. 前 10 分钟：三件事，按这个顺序做

### ① 立刻关机 → 切「无卡模式」开机

现在开着 4090 却什么都没跑。而接下来的 **90% 工作量完全不需要 GPU**：
传代码、装环境、下 ~90GB 数据、数据体检。

| 模式 | 价格 | 干什么 |
|---|---|---|
| 有卡（4090） | ¥1.88/时 | **只在跑训练/评测时开** |
| 无卡模式 | ¥0.1/时 | 传文件、pip、下数据、体检、打包 |

差 **19 倍**。操作：控制台 → 【关机】→ 等状态变「已关机」→【开机】→
计费方式选 **无卡模式**。

> 两个已知限制：部分区域的无卡模式只有 2核4G；**同一账号同时只能开一个无卡实例**。

### ② 充值

「余额不足 24 小时」= 按 4090 的价，余额撑不到一天。建议先充到
**覆盖一次完整实验的额度：¥200**（75–120 卡时 ≈ ¥150–200，外加数据盘月费）。
不用一次充很多，按量计费可以随时补。

### ③ 查数据盘（唯一可能推翻整台机器的事实）

控制台 → 该实例【更多】/【实例详情】→ 看 **数据盘容量** 与 **可扩容上限**。

| 路线 | 峰值占用 | 能出什么 |
|---|---|---|
| 只下 test / val / CASIA（不下 train） | ≈ **25 GB** | 泛化表、定位实验；**出不了正式训练指标** |
| train **只解 4 类**（推荐） | ≈ **89 GB** | 完整三阶段训练（74.9 下载 + ~14 解压，解压后自动删卷） |
| train 全 20 类 | ≈ **145 GB** | 同上，但**多花 56GB 换来零收益**（见下） |

**需要 ≥150GB，推荐 200–250GB。** 扩容约 ¥0.0065/日/GB → 多 200GB ≈ ¥1.3/天。

> ★ **为什么"只解 4 类"不亏**：`configs/default.yaml` 写着 `max_samples: 40000` ——
> **无论盘上是 4 类（14.4 万张）还是 20 类（72 万张），训练实际只取 4 万张。**
> 代价只是真实图像的**对象类别多样性**下降，论文里注明训练集范围即可
> （`docs/10` §5 已定的止损口径）。
>
> ⚠️ **若扩容上限显示 0GB**：这台机器永远扩不了 → 现在就释放、换一台可扩容的。
> 这比显卡型号更是硬否决项。
>
> 省钱技巧：**按默认 50GB 开机，真要下数据时再扩容**（数据盘按当天最高容量计费）。

---

## 2. 无卡模式阶段（总花费约 ¥5，1–2 小时，大部分是下载时间）

### 2.1 传代码（4.5 MB 的包，不是整个仓库）

本地已经打好了：`outputs/upload/vibnet_code.tar.gz`（4.5 MB；只含
`src scripts configs docs app tools requirements.txt README.md`，**不含任何数据与权重**）。

二选一上传：

**A. JupyterLab 拖拽（最省事）** — 控制台点【JupyterLab】→ 进 `/root/autodl-tmp/` → 把 tar.gz 拖进左侧文件面板。

**B. scp（本地 Git Bash）** — 端口/主机从控制台的 SSH 登录信息里看：
```bash
scp -P <端口> outputs/upload/vibnet_code.tar.gz root@<host>:/root/autodl-tmp/
```

云端解包（JupyterLab 的 Terminal 或 SSH 都行）：
```bash
cd /root/autodl-tmp
tar -xzf vibnet_code.tar.gz
ls -d vibnet-forgery-detector     # 必须在！包内自带顶层目录，解出来就该是它
cd vibnet-forgery-detector && ls scripts/cloud_autodl.sh   # 有输出 = 位置对了
```

> ⚠️ **这一步是最高频的翻车点（见 `docs/08` D35）。** 代码包**自带顶层目录**，
> 所以 **不要**再 `mkdir -p` + `-C`，直接在 `/root/autodl-tmp` 下解包即可。
> 一旦解错位置，脚本会报「项目目录不存在」，而文件其实都在 —— 症状很迷惑。
> 真解错了也别慌，`doctor` 会**自动检测并把修复命令原样打给你**（`mv` 那一行）。

> ⚠️ **必须放在 `/root/autodl-tmp`（数据盘）**。系统盘只有 30GB，
> 而每个 checkpoint 362MB、"保留最近 3 份"加 `_last`/`_best` 就要 1.5GB+。

### 2.2 一条命令跑完所有准备

本轮新增了 `scripts/cloud_autodl.sh`，把下面这些步骤都封装好了：

```bash
bash scripts/cloud_autodl.sh status    # 只读：我现在走到哪了 / 下一步跑哪个 stage（可随时跑）
bash scripts/cloud_autodl.sh doctor    # 体检：GPU / torch 是否 CUDA 版 / 磁盘 / 依赖 / 解包位置 / 旧副本
bash scripts/cloud_autodl.sh install   # 装云端依赖子集 + torch/numpy/transformers 互斥自检与自动修复
bash scripts/cloud_autodl.sh clip      # 取真 CLIP-ViT-B/16 权重（会先查版本互斥，避免白试两个源）
bash scripts/cloud_autodl.sh data      # 下 val / test / CASIA / train 4 类
bash scripts/cloud_autodl.sh verify    # 数据体检 + 全量自检 + 数据能否被吃进去 + 骨干真伪
```

每个 stage 都可以单独重跑，不会互相破坏。

**可覆盖的环境变量**（不用改脚本）：

| 变量 | 默认 | 用途 |
|---|---|---|
| `CONFIG` | `default` | 换成 `configs/lite.yaml` 等实验配置 |
| `ALLOW_BACKBONE_FALLBACK` | `0` | `=1` 才允许带着兜底骨干跑（仅用于验证代码链路，**产出的指标不可进论文**） |
| `CLASSES` | `car cat chair horse` | train 只解这几个类 |
| `JOBS` | `8` | DataLoader 进程数 |
| `AUTOSHUTDOWN` | `1` | 训练结束后自动关机 |

### 2.3 五个必须看懂的 CHECK 点

| 位置 | 看到什么 | 含义与处理 |
|---|---|---|
| `doctor` §2 | `torch.version.cuda=None` | 镜像是 CPU 版 torch → **必须**按提示重装 cu121 版（约 2.5GB，无卡模式放心下） |
| `doctor` §2 | 看不到 GPU | **无卡模式下这是正常的**，不代表有问题 |
| `doctor` §3 | 可用空间 <90GB | train 那一步会自动跳过；先扩容数据盘再回头跑 `data` |
| `doctor` §3 | 表格里是 `overlay 30G /` | 那是**系统盘**。要看的是它上一行「数据盘挂载点」和「数据分区可用：250 GB」→ 数据盘够，别去释放机器（见 `docs/08` D37） |
| `install` | `transformers 用不了 torch` | 镜像 torch 与新版 transformers 版本互斥 → **自动降级**到能用的版本；不必换源、不必挂代理（见 `docs/08` D38） |
| `clip` | 两个源都失败 | **先看它上面第 0 步的结论**：若是版本互斥，脚本会直接拦下并给修法 —— 那不是网络问题。确属网络问题才走**路线 B**（本地 `scp` 传 `pretrained/vit_b16_imagenet.pt`（328MB）上去，再把 `backbone` 改成 `tiny_vit` + `backbone_weights`）。**论文里必须如实写"骨干替换"** |
| `verify` 最后两项 | 真跑了 2 个 batch + 骨干真伪 | 「2 个 batch 过了」= 数据能被吃进模型；「骨干 = 真 CLIP」= 论文前提成立。**任一没过都别开机烧卡** |

> ⚠️ **最重要的一条**：`probe` 与 `train` 现在会先过一道**骨干闸门**
> （`python scripts/check_backbone.py --require-clip`）。CLIP 权重没取到时它会**直接终止**，
> 而不是让 `allow_fallback` 把骨干静默换成随机初始化再照常跑完 —— 那种"跑通了但没有意义"
> 的结果最贵：指标会产出，但论文前提已经没了（见 `docs/08` D12 / D38）。

> `data` 阶段如果不知道类名怎么写，先零成本列一遍：
> `python scripts/fetch_datasets.py --list-train --root /root/autodl-tmp/data`

### 2.4 中断几天后回来：一条命令定位"我现在在哪"

上云不是一次做完的事，中间会隔天。隔天再打开时最怕的不是报错，而是**记不清走到哪了、
手里这份代码是新版还是旧版**。所以有一个只读的 stage：

```bash
bash scripts/cloud_autodl.sh status    # 只读，不改任何东西；无卡模式下也能跑
```

它按顺序回答四件事，最后直接告诉你下一步该跑哪个 stage：

| 小节 | 回答的问题 |
|---|---|
| 0. 位置与磁盘 | 项目在哪、落在哪块盘、还剩多少 |
| 1. 数据 | 压缩包下到哪了、解压产物有多大 |
| 2. 权重 | 真 CLIP 到手没有（决定指标能不能进论文） |
| 3. 依赖 | torch 与 transformers 是不是互相认识（`clip` 假失败的头号根因） |
| 4. 骨干真伪 | 走的是真 CLIP / 路线 B / 随机初始化 |
| 5. 结论 | 下一步跑 `install` / `clip` / `data` / `verify` 里的哪一个 |

> 手里这份包若是旧版（没有 `status`），粘这一段等价的自查块即可：
>
> ```bash
> cd /root/autodl-tmp/vibnet-forgery-detector 2>/dev/null || echo "[X] 项目目录不存在"
> echo "--- A 位置"; pwd; ls
> echo "--- B 上级散装旧副本（干净时应无输出）"
> ( cd /root/autodl-tmp && ls -d src scripts configs docs app tools requirements.txt README.md 2>/dev/null )
> echo "--- C 版本指纹"
> ls scripts/check_backbone.py 2>/dev/null || echo "缺 check_backbone.py -> 旧版包"
> df -h /root/autodl-tmp | tail -1
> echo "--- D 数据"
> du -sh /root/autodl-tmp/data/_downloads 2>/dev/null
> ls -la /root/autodl-tmp/data/_downloads/ 2>/dev/null | tail -12
> du -sh /root/autodl-tmp/data/Datasets/* 2>/dev/null      # 小文件多，等几十秒
> echo "--- E 权重"; ls -la pretrained/ 2>/dev/null
> echo "--- F 版本互斥"
> python - <<'EOF'
> import torch, transformers
> print("torch", torch.__version__, "cuda", torch.version.cuda)
> print("transformers", transformers.__version__,
>       "is_torch_available", transformers.is_torch_available())
> from transformers import CLIPVisionModel
> print("CLIPVisionModel OK")
> EOF
> ```

> ⚠️ **看到 `/root/autodl-tmp` 顶层散着 `src/ scripts/ configs/ docs/ app/ tools/` 就顺手归档掉**：
> 那是**旧包被摊平解包**的残留。从那个目录直接跑 `python scripts/xxx.py` 会用**旧代码**
> （没有骨干闸门、旧版参数），而且**不会有任何报错**。归档不是删除，可随时还原：
>
> ```bash
> mkdir -p /root/autodl-tmp/_stale_code
> mv /root/autodl-tmp/{src,scripts,configs,docs,app,tools,requirements.txt,README.md} \
>    /root/autodl-tmp/_stale_code/ 2>/dev/null
> ```
> `doctor` 的第 4 节现在也会自动检测并原样打印这条命令（见 `docs/08` D39）。

---

## 3. 有卡模式阶段（这时才开始烧钱）

控制台【关机】→【开机】→ 这次不选无卡模式，确认挂着 4090。

```bash
nvidia-smi                                   # 确认卡在
cd /root/autodl-tmp/vibnet-forgery-detector

bash scripts/cloud_autodl.sh probe           # 短跑 100 步 → 实测速度 + 外推卡时与费用
bash scripts/cloud_autodl.sh train           # 正式三阶段训练（后台 + 结束自动关机）
```

**`probe` 会打印三个数，就是要看的全部：**

```
训练集        : 46,120 张  ->  约 2,883 步/epoch（batch=16）
单步耗时      : 0.0xx s
单 epoch      : x.x min
70 轮 + 20%开销: xx.x 小时
按 ¥1.88/时   : 约 ¥xxx（不含数据盘月费）
```

> 若"70 轮总小时数" 远超 150 小时，**先别开跑**，按提示查三件事：
> `num_workers`、是否真吃到 GPU、数据是否在数据盘（系统盘 IO 会拖死 dataloader）。
>
> 训练集 46120 张 = ForenSynths 40000 + CASIAv2 6000 + COVERAGE 120，是正常的。

**`train` 阶段的行为**：
- 用 `nohup` 起在后台，**SSH 断了也不停**
- 日志：`outputs/gpu_train_log.txt`；进度：`tail -f` 或 `python scripts/check_status.py`
- **训练结束自动执行 `/usr/bin/shutdown`**（`AUTOSHUTDOWN=1`，脚本内置）
- 中途要接着跑：
  ```bash
  python scripts/train.py --config configs/default.yaml --device cuda --amp \
      --workers 8 --data-root /root/autodl-tmp/data/Datasets --tag vibnet \
      --resume checkpoints/vibnet_last.pt
  ```

跑完（或要释放实例前）：
```bash
bash scripts/cloud_autodl.sh pack     # 只打包 best/last 权重 + 全部 json/txt/png
```

---

## 4. 花钱的三条铁律

1. **无卡模式做完所有准备** —— 这一步省掉 90% 以上的费用。
2. **一开机就连着跑完** —— AutoDL 按量计费**关机后不保留这张卡**，
   第二天开机原主机可能已无空卡；数据盘在，但卡没了就得换机器。
3. **让训练结束自动关机** —— 脚本已内置；别依赖"我等下记得关"。

**费用总账（4090 @ ¥1.88/时）**

| 项 | 量 | 费用 |
|---|---|---|
| 三阶段正式训练 | 23–35 卡时 | ¥43–66 |
| 消融 4 项 | 16–28 卡时 | ¥30–53 |
| 定位（CASIA v2） | 3–5 卡时 | ¥6–9 |
| 泛化 + 鲁棒性 | 2–4 卡时 | ¥4–8 |
| 数据盘 200GB | 1 个月 | 约 ¥39 |
| **无卡模式（准备阶段）** | 约 20 小时 | **¥2** |
| **合计** | **75–120 卡时** | **约 ¥125–180** |

---

## 5. 故障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `torch.version.cuda=None` | 镜像自带 CPU 版 torch | 按 `doctor` 的提示重装 cu121 |
| **`clip` 报"两个源都失败"，日志里是 `ImportError: CLIPVisionModel requires the PyTorch library`** | **与网络无关**：新版 transformers 要求 `torch>=2.5`，而镜像是 `2.3.0+cu121` | `install` 会自动降级 transformers；手动修：`python -m pip install 'transformers==4.44.2'`，再 `python -c "from transformers import CLIPVisionModel"` 验证 |
| `probe`/`train` 被"骨干不是预训练 CLIP"拦下 | CLIP 权重没取到（`clip` 没成功） | 回去跑 `clip`。要强行跑通链路：`ALLOW_BACKBONE_FALLBACK=1 bash scripts/cloud_autodl.sh probe`（指标不可进论文） |
| `probe` 报"看不到可用的 CUDA" | 实例还在无卡模式 | 切【有卡模式】开机再跑；CPU 上测出来的速度外推费用会错两个数量级 |
| `找不到 ForenSynths` | `--data-root` 传成了 `<data>` | 新版**会自动补一级并打印提示**；写全 `<data>/Datasets` 更稳妥 |
| `CUDA out of memory` | batch 偏大 | `--batch-size 8`（24GB 卡正常不需要） |
| dataloader 卡死 | `num_workers` 过大 / 数据在系统盘 | `--workers 4`；项目必须放 `autodl-tmp` |
| 训练变慢几十倍 | 在 CPU 上开了 `amp` | 已修：现在 CPU 上强制关闭并打印说明（见 `docs/08` C19） |
| 下载卡住不知死活 | — | `python scripts/check_status.py`（进程 / 日志新鲜度 / 进度推进 三证据交叉） |
| **从 `/root/autodl-tmp` 里跑 `python scripts/xxx.py` 行为不对（闸门不生效、参数对不上）** | **那个目录下还散落着旧包摊平的 `src/ scripts/ ...` 副本，你用到了旧代码** | `bash scripts/cloud_autodl.sh doctor` 会检测并打印一条**整行可复制**的归档命令；或手动 `mv` 到 `/root/autodl-tmp/_stale_code/`（见 `docs/08` D39） |
| `doctor` 第 5 节报"体检未通过" | 上面有带 `[!]` 的硬闸门项（通常是磁盘不足） | 按提示修；`doctor` 现在**会把整份报告打完再给结论**，不再中途 `exit`，所以别只看最后一节 |
| 关机后再开机没卡 | 按量计费不保留机型 | 换机器；用 `/root/autodl-fs`（同地区 20GB 免费）中转 checkpoint |

**分不清自己走到哪一步时，第一条永远是**：`bash scripts/cloud_autodl.sh status`（只读，见 §2.4）。

---

## 6. 本轮随代码一起上云的改动（已随包提交）

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `scripts/cloud_autodl.sh` | **新增**，**9 个** stage 的一键脚本（含只读的 `status`）；`install` 加 torch/numpy/transformers 互斥自检与**自动修复**；`clip` 第 0 步先查版本互斥；`probe`/`train` 加**骨干闸门**与 CUDA 闸门；`doctor` 修 `df` 落盘显示、加**摊平解包检测**与**散装旧副本检测**；打包**自带顶层目录** | 把"上云该干什么"从文档变成可执行，并让每个必然踩的坑自带修复命令 |
| `scripts/check_backbone.py` | **新增**：把模型真正装配出来，判定骨干是「真 CLIP / 路线 B / 随机初始化」三态，退出码 `0/2/3` | `allow_fallback` 写死为 True，CLIP 缺失会**静默**退化成随机初始化骨干（见 `docs/08` D12 / D38） |
| `scripts/train.py` | 新增 `--amp` / `--no-amp` / `--data-root` / `--out-dir` | 云端必须开 bf16；数据在数据盘不用改 yaml；probe 不污染正式 run |
| `src/engine/trainer.py` | 新增 `resolve_amp()`；bf16 不再建 GradScaler；CPU 上强制关 amp | 见 `docs/08` C19 |
| `configs/default.yaml` | 新增 `amp_dtype: "bfloat16"` | 让 dtype 显式可配 |
| `requirements.txt` | `transformers >=4.35,<5` | 大版本升级不保证 CLIP 加载路径与权重键名不变（见 `docs/08` D36 / D38） |
| `scripts/smoke_test.py` | 新增 `test_amp_policy()`（6 条）、`test_check_backbone_tool()`、`test_cloud_autodl_cli()` | AMP、骨干闸门、上云脚本都是"只有花钱时才第一次跑"的路径，必须在本机钉死。上云脚本那条钉住了「9 个 stage 齐全 / `bash -n` 通过 / doctor 打印的归档命令必须是**单行可复制**」—— 最后一条正是实测抓到的 bug |
| `outputs/upload/vibnet_code.tar.gz` | **新增**，4.6MB 上传包，**自带顶层目录** | 避免把 43GB 的 `data/` 一起传；顶层目录避免解包摊平（见 `docs/08` D35） |

自检：**37/37 通过**（原 34 项 + AMP 策略 1 项 + 骨干检查脚本 1 项 + 上云脚本 CLI 1 项）。

---

## 7. 一句话

> **先在无卡模式里把"代码、环境、数据、体检"四件事做完并与 ¥5 结账，
> 再开 4090 跑 `probe` 把那句"要花多少钱"变成实测数字，最后 `train` 让它自己关机。**
