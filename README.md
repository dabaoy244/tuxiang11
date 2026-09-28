# 空频双分支轻量 VIB-Net —— AI 图像篡改检测

> 广西大学大学生创新训练项目 · 申报书《基于空频双分支轻量 VIB-Net 的 AI 图像篡改检测模型研究与系统开发》
> 本仓库是该申报书的**完整可运行工程实现**：从空频特征提取、分层 VIB、梯度隔离双支路多任务，
> 到 ONNX/OpenVINO 端侧部署与 Windows 离线桌面检测工具。

> ### 📊 最新正式训练结果（2026-09-28，云端 4090 单卡全量重跑）
>
> **详见 → [`README_RUN_20260928.md`](README_RUN_20260928.md)**
>
> | 指标 | 实测 | 目标 | 判定 |
> |---|---|---|---|
> | CASIAv2 真伪检测 ACC / AUC | **93.82% / 98.47%** | — | — |
> | 篡改定位 mIoU（★`miou_tampered_only`） | **26.68%** | ≥56% | ❌ 未达标 |
> | 模型体积（FP32） | 362.1 MB | ≤120MB | ❌ 未达标 |
> | ForenSynths 检测 ACC | **无数据**（未单独评测） | ≥91% | ⚠️ 测不了 |
> | CPU ≥8 张/秒 | 未真测（JSON 判定用了 GPU 数字） | ≥8 | ⚠️ 不可引用 |
>
> **一句话**：训练链路完整跑通、分类可用，但定位指标与体积两项硬指标未达标；
> 且定位的差距是**结构性**的（stage3 全程平台期，补轮数无效），下一步要改的是分辨率与监督信号，不是轮数。


---

## 一、这个仓库能干什么

| 申报书章节 | 对应代码 | 状态 |
|---|---|---|
| 1.1(1) 空域分支（CLIP-ViT-B/16 + 3 层深度可分离卷积） | `src/models/spatial_branch.py` | ✅ 公式 (1)(2)(3) 已核对 |
| 1.1(2) 频域分支（2D-DFT 幅相联合 + 可学习掩码） | `src/models/freq_branch.py` | ✅ 公式 (4)~(12) 已核对 |
| 1.1(3) CS-CAM 通道-空间交叉注意力融合 | `src/models/cs_cam.py` | ✅ 公式 (13)~(17) 已核对 |
| 1.2 分层 VIB（β 退火 / KL 裁剪 / 梯度裁剪） | `src/models/vib.py` | ✅ 公式 (20)~(26) 已核对 |
| 1.3(1) 梯度隔离梯度停止层 | `src/models/gradient_stop.py` | ✅ 含"该层在兄弟头结构下是恒等映射"的说明 |
| 1.3(3) Mobile-UNetv2 定位头 + Canny 边缘监督 | `src/models/mobile_unetv2.py` | ✅ 公式 (28)(29) 已核对 |
| 1.4 不确定性加权多阶段联合损失 | `src/losses/multi_task.py` | ✅ 公式 (30)(31) 已核对 |
| 三阶段分任务训练 | `src/engine/trainer.py`、`scripts/train.py` | ✅ |
| 3.1 ONNX 导出 + 简化 + OpenVINO 加速 | `src/deploy/export_onnx.py`、`optimize_openvino.py` | ✅ |
| 3.1 模型轻量化（INT8 量化） | `src/deploy/quantize_int8.py` | ✅ 实测 362.1MB → 92.9MB |
| 3.2 Windows 离线桌面检测工具（五大模块） | `app/` | ✅ |
| 3.3 性能验证（鲁棒性 / 对比 / 消融） | `src/evaluation/` | ✅ |

**实测：** `python scripts/smoke_test.py --full` 全流程 29/29 项自检通过，
并完成"合成数据 → 三阶段训练 → 评测 → ONNX 导出 → INT8 量化 → OpenVINO 转换 + 性能测试"的端到端运行。

**两项硬指标已实测达标**（详见 `docs/09_指标可达性与骨干选型.md`）：

| 指标 | 配置 | 实测 | 判定 |
|---|---|---|---|
| 模型体积 ≤ 120MB | `configs/lite.yaml` + ONNX 简化 | **98.7 MB** | ✅ |
| CPU ≥ 8 张/秒（224×224） | 同上（ONNX Runtime，本机 7 核） | **9.46 ~ 11.22 张/秒** | ✅ |

---

## 二、5 分钟跑起来

```bash
cd vibnet-forgery-detector

# 0) 安装依赖（CPU 版，开发调试够用）
pip install -r requirements.txt

# 1) 生成离线演示数据集（程序化合成，无网络即可）
python -m src.data.synth --root data/demo --train 240 --val 60 --test 60

# 2) 端到端自检：核对申报书每一条公式的张量维度、梯度、参数量、模型体积
python scripts/smoke_test.py

# 3) 快速训练一轮（把 70 轮压缩到 ~4 轮，CPU 几分钟）
python scripts/train.py --demo --epochs-scale 0.05

# 4) 评测 + 鲁棒性 + 消融
python -m src.evaluation.evaluate   --demo --ckpt checkpoints/vibnet_best.pt
python -m src.evaluation.robustness --demo --ckpt checkpoints/vibnet_best.pt --max-samples 60
python -m src.evaluation.ablation   --demo --mode eval

# 4b) 查看后台长任务（下载/训练）到底还在不在跑 —— 进程 + 日志时间 + 进度三重证据
python scripts/check_status.py
python scripts/check_status.py --watch --interval 30

# 5) 导出部署模型并测速
python -m src.deploy.export_onnx --ckpt checkpoints/vibnet_best.pt --out deploy/vibnet.onnx --simplify
python -m src.deploy.optimize_openvino --onnx deploy/vibnet_simplified.onnx --bench
python -m src.deploy.optimize_openvino --size-only --params-m 33.84   # 体积核算

# 5b) 指标验收（体积 ≤120MB + CPU ≥8 张/秒）—— 用 lite 档
python -m src.deploy.export_onnx --ckpt "" --config configs/lite.yaml --out deploy/lite.onnx
python scripts/benchmark_cpu.py --reps 5 --runs 20     # 输出 outputs/cpu_benchmark.json
python scripts/measure_backbone_budget.py              # 四个骨干档位的体积/延迟对比

# 5c) 体积优先时可叠加 INT8 量化（注意：本机实测 INT8 会更慢，见 docs/09）
python -m src.deploy.quantize_int8 --onnx deploy/lite.onnx --out deploy/lite_int8.onnx --bench

# 6) 启动 Windows 离线桌面检测工具
python app/main.py
```

> 演示数据集是**程序化合成**的，只用来验证"流程通不通"。
> 论文/结题材料里的指标必须用 ForenSynths / CASIA v2 / COVERAGE 真实数据集重跑。

---

## 三、换到真实数据集正式训练

> 数据集**不需要向作者申请**——ForenSynths 官方已托管 HuggingFace 公开下载（共 90 GB）。
> 详见 `docs/02_数据集构建.md` 与 `docs/10` §1.2。

```bash
# 0) 下载数据集（支持断点续传 / 多镜像自动切换 / 选择性解压）
#    先下 1.6 GB 的小文件，立刻就能开始真实数据训练
python scripts/fetch_datasets.py --split val progan_test
#    跨生成器泛化测试集（18.7 GB，建议挂夜）
python scripts/fetch_datasets.py --split test
#    正式训练集（70.4 GB；只想用 4 类可加 --classes car cat chair horse）
python scripts/fetch_datasets.py --split train --classes car cat chair horse
#    慢的话换国内镜像；--mirror auto（默认）会在主站被阻断时自动切 hf-mirror 并指数退避重试
python scripts/fetch_datasets.py --split val --mirror hf-mirror

# 0b) 篡改定位数据集 CASIA v2.0（3.28 GB）—— mIoU≥56% 这条指标全靠它，最容易被漏
python scripts/fetch_tamper_datasets.py --workers 16
#    下载完自动整理成 {train,val,test}/{image,mask}/，真实图补全黑掩码

# 0c) 后台长任务到底还在不在跑？（进程 + 日志新鲜度 + 进度推进，三重证据）
python scripts/check_status.py
python scripts/check_status.py --watch --interval 30

# 1) 检查 / 规范化数据集
python scripts/prepare_datasets.py --root data/Datasets --check
python scripts/prepare_datasets.py --root data/Datasets --organize --mode link
python scripts/prepare_datasets.py --root data/Datasets --manifest

# 2) 准备骨干权重（二选一，见 docs/15 §5）
#    A 路线：真 CLIP 权重（论文可写"CLIP 语义先验"）
export VIB_NET_ALLOW_DOWNLOAD=1     # Windows: set VIB_NET_ALLOW_DOWNLOAD=1
#    B 路线：CLIP 拿不到时的兜底（ImageNet-1K 预训练 ViT-B/16，共 346 MB）
python scripts/fetch_pretrained_backbone.py

# 3) ★ 先做受控对照实验：证明"预训练权重决定能不能训起来"
#    两臂仅在 backbone_weights 上不同（另两处 diff 是实验标签与落盘路径，不影响训练）
python scripts/run_ablation.py

# 4) 正式训练（三阶段 20+20+30 = 70 轮，建议单卡 GPU）
python scripts/train.py --config configs/default.yaml --device cuda --workers 8

# 5) 完整评测
python -m src.evaluation.evaluate   --ckpt checkpoints/vibnet_best.pt --split test
#    测定位 mIoU 必须限定数据集：混合 test_sets 里 ForenSynths 排在前面，
#    不加 --dataset 时 CASIAv2 可能一批都轮不到，定位指标会静默为空
python -m src.evaluation.evaluate   --ckpt checkpoints/vibnet_best.pt --split test --dataset CASIAv2
python -m src.evaluation.robustness --ckpt checkpoints/vibnet_best.pt
python -m src.evaluation.ablation   --mode train      # 每项独立重训（--mode eval 不能出论文数据）

# 5b) 跨生成器泛化（论文核心表）—— 按生成器分层采样，必须用它而不是 evaluate.py
#     ★ --scores-out 把逐样本分数存盘：以后改指标定义可免推理重算（推理是最贵的一步）
python scripts/eval_cross_generator.py --ckpt checkpoints/vibnet_best.pt \
    --per-class 300 --out outputs/cross_gen \
    --scores-out outputs/cross_gen_scores/full.npz

# 5b') 改过度量实现后，从已存分数秒级重算（复用同一段聚合逻辑，口径不会分叉）
python scripts/recompute_cross_gen.py \
    --scores outputs/cross_gen_scores/full.npz --out outputs/cross_gen

# 5c) ★ 指标口径回归测试（改动度量实现后必跑；已并入 smoke_test 自动执行）
#     定位：三种 mIoU 口径 + 空掩码跳过语义
#     分类：AUC 的并列秩平均 + AP 的分组积分 + 三个入口必须逐位一致
python scripts/test_metric_definitions.py
```

> **报 mIoU 前必读**：`evaluate.py` 会同时打印三种口径 ——
> ① 池化 `miou`（按像素加权，最宽松）② **仅篡改图 `miou_tampered_only`（文献标准，★论文报这个）**
> ③ 全部非空图 `miou_per_sample`（含误报惩罚，最严格）。
> 三者能差一倍以上，**报哪个必须写明**，别挑最高的那个。详见 `docs/09` §4.4。

> **报 AUC/AP 前必读**：分数出现大量并列时（退化模型常整批输出 0.0 或 1.0），
> 必须用**秩平均校正**的 AUC，否则会把退化臂的分数**系统性压低**、反而放大对照优势。
>
> AP 用 **Σ (Rₙ−Rₙ₋₁)·Pₙ 且按不同分数值分组**（并列共用一个阈值）：
> 完美可分 → 1.0、**分数全同 → 正类占比**。逐正例求和会在并列时偏低
> （50 正 50 负全同分给 0.3118，正确值 0.5）。
>
> ★ 这两个指标**全仓库只有 `src/engine/metrics.py` 一处实现**——
> 曾经主评测与跨生成器各有一份、口径不同，两张表的 `ap` 不可比。
> 边界不变量与"三处入口必须逐位一致"由 `scripts/test_metric_definitions.py` 守住。

---

## 四、目录结构

```
vibnet-forgery-detector/
├── configs/
│   ├── default.yaml                # b16 精度优先档（全量超参）
│   └── lite.yaml                   # s16 达标档：98.7MB / 9.5~11.2 张每秒
├── src/
│   ├── models/                     # 模型：与申报书公式一一对应（含 BACKBONE_PRESETS）
│   ├── data/                       # 数据：三套公开数据集 + 离线演示数据合成器
│   ├── losses/                     # 损失：Z-score 归一化 + 同方差不确定性加权
│   ├── engine/                     # 引擎：三阶段训练 + 指标
│   ├── evaluation/                 # 评测：主评测 / 鲁棒性 / 消融
│   └── deploy/                     # 部署：ONNX 导出 / 简化 / INT8 量化 / OpenVINO / 推理封装
├── app/                            # PyQt5 桌面工具（输入/检测/展示/导出/设置 五大模块）
├── scripts/                        # 入口脚本：训练、自检、数据集准备、指标验收基准
│                                   #   ├── check_status.py             # 后台长任务三证据存活判定（进程/日志/进度）
│                                   #   ├── fetch_datasets.py           # ForenSynths 下载：多镜像自动切换 + 断点续传
│                                   #   ├── fetch_tamper_datasets.py    # CASIA v2.0 下载 + 自动整理成定位训练布局
│                                   #   ├── fetch_pretrained_backbone.py# ImageNet ViT-B/16 下载 + 键名转换（CLIP 兜底）
│                                   #   ├── run_ablation.py             # ★ 骨干权重受控对照实验（训练+评测+报告）
│                                   #   ├── finalize_ablation.py        # 等实验结束 → 重出报告 → 提升默认权重（幂等）
│                                   #   ├── make_ablation_configs.py    # 同一模板生成两臂（除标签/路径外只有骨干权重不同）
│                                   #   ├── eval_cross_generator.py     # 按生成器分层采样的跨模型泛化评测（--scores-out 存逐样本分数）
│                                   #   ├── recompute_cross_gen.py      # ★ 从已存分数免推理重算跨生成器指标
│                                   #   ├── test_metric_definitions.py  # ★ 指标口径回归测试（三种 mIoU + AUC 并列校正 + AP 分组 + 跨入口一致）
│                                   #   ├── make_ui_screenshots.py      # 离屏渲染界面截图（拆两进程避开 Qt×torch 崩溃）
│                                   #   ├── make_copyright_docs.py      # 软著：源程序文档 + 软件说明书 PDF
│                                   #   └── plot_training_curves.py     # 训练曲线绘图（无 matplotlib 依赖）
├── docs/                           # 实施指南（★ 先看 docs/00、docs/07、docs/09）
│   └── copyright/                  # 软著申报材料（源程序 60 页 + 说明书 10 页 + 10 张界面截图）
├── configs/                        # default.yaml（b16 高精度）/ lite.yaml（s16 达标档）
│                                   #   / lite_realval.yaml（无 GPU 时的真实数据可训练性验证）
└── data/                           # 数据集与演示数据（不入库）
```

---

## 五、文档导航

| 文档 | 内容 | 什么时候看 |
|---|---|---|
| **`docs/00_实施路线图.md`** | 分 8 个阶段的完整实施步骤、交付物、验收标准、时间安排 | **先看这个** |
| `docs/01_环境搭建.md` | 软硬件环境、依赖版本、常见环境坑 | 动手第一天 |
| `docs/02_数据集构建.md` | 三套数据集的获取、规范化、划分、manifest | 阶段一 |
| `docs/03_模型实现与公式对照.md` | 每个公式 → 代码位置的对照表、实现难点 | 阶段三 |
| `docs/04_训练策略.md` | 三阶段训练、损失权重、调参经验、失败模式 | 阶段四 |
| `docs/05_评测与实验设计.md` | 指标口径、鲁棒性、对比、消融的实验设计 | 阶段六 |
| `docs/06_部署与桌面工具.md` | ONNX/OpenVINO 优化、PyQt5 工具、打包发布 | 阶段五 |
| **`docs/07_申报书待修正问题清单.md`** | **申报书中 7 处技术上站不住/需补充的地方及修正建议** | **答辩前必看** |
| `docs/08_风险与坑清单.md` | 已知工程坑与规避方法 | 随时查 |
| **`docs/09_指标可达性与骨干选型.md`** | **实测数据：CLIP-ViT-B/16 与 ≤120MB 指标的硬冲突、四个骨干档位对比、INT8 反而变慢的原因、验收口径** | **定指标 / 答辩前必看** |
| **`docs/10_剩余工作与单人执行路线图.md`** | **从现在到结题还要做什么：四条主线任务清单、算力成本预算、重排后的单人时间表、止损规则** | **★ 当前最该看这个** |
| **`docs/11_申报书修订对照表.md`** | **申报书逐条修订建议 + 可直接粘贴的改后文字 + 中期检查《研究内容调整说明》模板 + 三条改指标铁律** | **中期检查前** |
| **`docs/12_论文框架与图表清单.md`** | **论文题目候选、摘要骨架、章节框架、14 图 6 表逐项状态标注、实验矩阵与省算力做法、可直接引用的实测数字** | **写论文时** |
| **`docs/13_中期检查材料包与答辩口径.md`** | **中期要交什么、数据附件清单（含测试环境口径）、进度对照表、指标口径调整方案、7 组答辩问答预案、提交前自检清单** | **中期检查前** |
| **`docs/14_对照实验_预训练权重消融.md`** | **受控实验：随机初始化 vs ImageNet 预训练（唯一变量=骨干权重）。判据是 `tn`/`tp` 是否同时为正，不是 ACC。为"缺预训练权重导致判决退化"提供受控证据，并如实标注强度边界（两臂最佳 AUC 0.6428 vs 0.6809，仅差 +0.038，故只称必要条件）** | **答辩被问"为什么训得动"时** |
| **`docs/15_云端GPU完整训练与评测手册.md`** | **★ 上云前必读：本地已验证的结论清单、要同步什么、数据/环境/训练/定位/泛化/消融/导出全流程照抄命令、诚实口径清单、故障速查** | **★ 要跑真指标前** |
| **`docs/18_AutoDL上云执行清单.md`** | **★ 实例已开好之后照着做：前 10 分钟三件事、数据盘闸门核算、无卡模式五步准备、有卡模式 probe→train、花钱三铁律、故障速查。配套 `scripts/cloud_autodl.sh` 九个 stage（含只读 `status`，随时定位"我走到哪了"）、torch/numpy/transformers 互斥自动修复、骨干闸门（烧卡前拦住"静默兜底骨干"）、散装旧副本检测** | **★ AutoDL 实例开好那一刻** |
| **`docs/19_云服务器怎么用_从零上手.md`** | **★ 从没连过远程服务器也能看懂：三种入口怎么点（JupyterLab / SSH / AutoPanel）、三块盘各放什么、密码怎么粘、文件怎么传回来、前台 vs nohup 后台、怎么看任务还活着、关机 vs 释放、新手最常卡的 9 个场景** | **★ 第一次登服务器前** |

---

## 六、关键提醒（先说结论）

1. **模型体积 ≤120MB 这条指标，用 FP32 的 CLIP-ViT-B/16 达不到 —— 这是申报书里的一处硬冲突。**
   实测：CLIP-ViT-B/16 结构骨干 85.81M 参数，全模型 90.53M → **FP32 362.1 MB**，超标 3 倍。
   两条出路（均已实测）：
   - **换轻量档骨干**：`configs/lite.yaml`（s16 档）→ **98.7 MB + 9.46~11.22 张/秒**，两项指标同时达标；
   - **保留 b16 + INT8 量化**：**92.9 MB** 达标，但 CPU 只有 2.86 张/秒，速度指标仍不达标。

   而且**动态 INT8 会让推理变慢**（lite 档 89.2 ms → 298.6 ms，慢 3.3 倍），
   因为 batch=1 时 `DynamicQuantizeLinear` 的开销超过 INT8 矩阵乘省下的时间，
   且本机 CPU 无 VNNI 指令。**量化省体积，不省时间。** 详见 `docs/09`。
2. **只判真伪、不做定位**的数据集（ForenSynths）**没有像素掩码**，混合批次必须用
   `mask_valid` 过滤，否则定位支路会被"全零掩码"教坏。
3. **"梯度停止层"在双兄弟头结构下前向是恒等映射**，按申报书原始描述实现等于空操作；
   本仓库提供了可真正观测到指标差异的严格解耦模式（`gs_stop_cls` / `gs_stop_loc`）。
   详见 `docs/07` 第 2 条。
4. **CLIP 冻结策略要与批次大小匹配**：batch_size 太小时 BN/LayerNorm 统计不稳，
   建议 batch≥16 且开启梯度累积。
5. **KL 裁剪 [0,10] 不能简单用 `torch.clamp`**，否则 KL 常态超限时梯度归零、
   VIB 被静默关闭。本仓库默认"前向硬裁剪 + 反向恒等"（可配置）。

---

## 七、许可与声明

- 本项目为高校大学生创新训练项目科研成果，代码仅供学习与科研使用。
- 使用的 ForenSynths / CASIA v2 / COVERAGE 数据集版权归各自原作者所有，请遵守其 License。
- 桌面工具输出的检测结论仅作辅助参考，**不作为司法鉴定依据**。
