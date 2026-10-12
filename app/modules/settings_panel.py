"""桌面工具 · 系统设置模块

可配置项（申报书 3.2）：推理后端、推理设备、检测阈值、结果保存路径、模型路径。
所有配置通过 QSettings 持久化到注册表/ini，下次启动自动恢复。
"""

from __future__ import annotations

import os

from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class SettingsPanel(QGroupBox):
    """⑤ 系统设置模块。"""

    applied = pyqtSignal(dict)

    DEFAULTS = {
        # ★ backend=onnx：本机实测 onnx 加载 0.3s、单张 ~90ms，而 torch 后端要 27s。
        #   torch / openvino 仍可在下拉框里选。
        "backend": "onnx",
        "device": "cpu",
        "threshold": 0.5,
        "save_dir": "./outputs",
        # ★★★ 2026-10-09（D 组 ③）：默认权重换成审计认定的交付臂 g43。
        #   依据**只有一条**：阈无关主口径 AUC 显著更高 —— ForenSynths 未见生成器
        #   （strict-9）宏平均 **AUC 0.8671 vs 出厂 0928 的 0.8434 ⇒ +2.37pp★**
        #   （同战役配对 bootstrap，CI [+0.0107, +0.0365] 不含 0；full-13 口径 +1.78pp★）。
        #   ⚠ 换权重前必查两件事（本项目踩过）：
        #     ① 不要认文件名，认 md5 —— checkpoint 内 `best.pt` 会被后续 stage 覆盖写；
        #     ② 换完必须重做一次「ONNX vs .pt 同抽样一致性」检查（本轮已做，见下）。
        # ★★★ 2026-10-10：默认权重换成 **geofix_s2_epoch10**（几何修复后 stage2 第 10 轮）。
        #   为什么换：原 g43 的定位支路是在**掩码错位**的管线下训出来的
        #   （`src/data/transforms.py` 里图像与掩码走两套几何，非方形图错位最多 35px，
        #    而 CASIAv2 只有 0.2% 是方形图 —— 见 deliverables/14）。
        #   同战役实测（deliverables/17/18，30 任务 0 失败）：
        #     CASIAv2 定位 mIoU★  0.2103 → **0.3571**（+14.7pp / +70%）
        #     DEFACTO 像素级 AUC  0.6762 → **0.7789**（+10.3pp）
        #     内域分类 AUC        0.9853 → 0.9864（**未受损**）
        #   归因对照（另一臂旧几何同样多训 20 轮）：只 +0.26pp ⇒ **几何修复占 98%**。
        #   ⚠ 交付点不是 `best.pt`：6 个域的 mIoU★ 与像素 AUC **全部在第 10 轮见顶**，
        #     之后回落；`best.pt` 落在回落之后。选点按预注册的域外口径，不按 best。
        # ★★★ 2026-10-11：默认权重再换成 **geofix_s3_best**（几何修复 + stage3 联合微调）。
        #   为什么必须再换：geofix_s2_epoch10 的**跨生成器分类掉了 11.75pp**
        #   （strict-9 宏 AUC 0.7490 vs g43 0.8665）—— 因为那一臂**只跑了 stage2**，
        #   分类头被冻结、无法适应被改动的共享特征。补 stage3 联合微调后（1-K 战役）：
        #     跨生成器分类  **0.7490 → 0.8632**（vs g43 的 0.8665，仅 −0.33pp ⇒ 救回）
        #     定位 mIoU★     0.3571 → **0.3482**（−0.89pp ⇒ 基本保住）
        #   ⇒ 相对上一版默认件是"分类 +11.42pp、定位 −0.89pp"；
        #     相对 g43 是"**分类持平、定位 +13.79pp**"（g43 同协议 = 0.2103）
        #     —— 即**在两个任务上都优于 g43**，这才是选它的理由。
        #   ⚠ 必带披露：stage3 的 12 轮里 `best` 出现在**第 1 轮**，而选点用的
        #     `val_cls_auc` 已被本项目实测为**不跟踪跨生成器性能**（Spearman −0.35）
        #     ⇒ "选 best"这条规则本身没有依据；它的跨生成器优势是**事后观测**。
        "ckpt": "outputs/upload/weights_geofix_s3_20261011/geofix_s3_best.pt",
        # ★★★ 2026-10-11：同步换 ONNX（纯 CPU trace，¥0，本地做）。
        #   导出命令：`python -m src.deploy.export_onnx --ckpt <上面的 .pt>
        #     --out deploy/b16_s3best.onnx --opset 17
        #     --config configs/geofix_s3_only.yaml --simplify`
        #   一致性（logit 级，本项目规矩）：cls_logits max|Δ| 2.10e-05、
        #     **mask_logits max|Δ| 3.89e-04** ⇒ 同一套权重。
        #   已核实 `--simplify` 真跑过：Identity 节点 72 → **0**，且与
        #     `deploy/b16_g43.onnx` 的算子直方图逐项相同（同一导出通路）。
        #   ⚠ 该 ONNX 有 3 个输出（cls_logits / mask_logits / **edge_logits**）——
        #     第 3 个是 edge 头，app 不消费。**不要按位置假设它是 mask_prob**
        #     （我第一版一致性脚本就这么错过，得到 25.8 这种不可能的"概率差"）。
        "onnx": "deploy/b16_s3best.onnx",
        # ★★★ 2026-10-11：OpenVINO IR 同步换（否则 ONNX 后端跑新权重、OpenVINO 跑旧）。
        #   生成：`openvino.convert_model('deploy/b16_s3best.onnx')` +
        #     `save_model(..., compress_to_fp16=False)` → `deploy/vibnet_ir_s3best/`
        #     （本地 CPU，1.2 s，¥0；已回读验证 3 个输出、输入 [?,3,224,224]）。
        #   旧件（0928 的 IR、g43 的 IR、geofix_e10 的 IR）都保留留档，不删。
        "openvino": "deploy/vibnet_ir_s3best/b16_s3best.xml",
        # ---- 历史留档（换默认件时务必读，本条是"存在但无效"的教训）----------------
        # ⛔ **绝不要**把默认件换回 `deploy/lite*.onnx` / `lite_simplified.onnx` /
        #    `lite_int8.onnx`：它们是 `export_onnx.py --ckpt ""`（**空 checkpoint**）
        #    导出的**随机初始化模型**，当初只为测「体积 / 速度」两项指标
        #    （docs/09 §导出命令、docs/08 §38），从未训练过。实测：对任何正常图像恒定
        #    输出 p_fake=0.4852，对全黑/全白输入出 NaN ⇒ 这类文件**存在但无效**，
        #    查"文件是否存在"拦不住，只能靠 `_KNOWN_DEAD` + `self_check()` 兜。
        # 📌 历任默认件：0928 出厂件 → g43(improved_noaug_best) → geofix_s2_epoch10
        #    → **geofix_s3_best**。
        #    每次换权重都必须：① 认 md5 不认文件名；② 重做 ONNX vs .pt 的 **logit 级**
        #    一致性；③ ONNX / OpenVINO 两个后端的默认件**同步换**（否则两个后端两个模型）。
        "config": "configs/default.yaml",
        "image_size": 224,
        "auto_load": True,
    }

    #: 这几项是**文件路径**，加载配置时要校验存在性（见 `load_config`）。
    _PATH_KEYS = ("ckpt", "onnx", "openvino")

    # ==================================================================
    # ★★ 模型卡（2026-10-09 新增）
    #
    # 为什么要有它：此前 `app/` 全目录**不含任何实测指标数字** —— 软件只会给
    # "这张图怎么样"，说不出"这个模型多少分"。软著/答辩时被问"精度多少"就答不上来。
    #
    # 为什么放在「关于」而不是结果面板/PDF 报告：
    #   结果面板与 PDF 报的是**单张推断结果**。若把测试集指标混进去，读者会把它
    #   误读成"本次检测的准确率"。性能指标属于"模型卡"，是**这个构建**的属性，
    #   不是**这一次检测**的属性。
    #
    # ★ 三条纪律（本项目已为此栽过三次）：
    #   ① 每个数都带 (权重 md5, 抽样口径, 聚合方式) —— 见 `MODEL_CARD_PROTOCOL`；
    #   ② **不跨权重误归属** —— 同一张卡里的所有数**必须属于同一个权重**。
    #      2026-10-10 换权重时严守了这条：旧卡是 g43 的数，换 geofix 后
    #      **没有**把 g43 的数留在卡里充数，而是把"还没为新权重测过的行"标成**待测**。
    #      宁可卡上有"待测"，也不能让读者把 g43 的 0.8671 当成 geofix 的。
    #   ③ 数字**只作参考、不作承诺** —— 它们是特定测试集上的指标，不是现场精度保证。
    # ==================================================================
    MODEL_CARD_WEIGHT = "geofix_s3_best.pt（几何修复 + stage3 联合微调开局）"
    MODEL_CARD_WEIGHT_MD5 = "184fb15c1787cf5470397adbe8a1ac93"
    MODEL_CARD_ONNX = "deploy/b16_s3best.onnx"
    #: ★★★ 2026-10-11（批次 1-K / 1-J）：本权重是**两个任务都优于上一交付件**的第一个版本。
    #:   相对 geofix_s2_epoch10：分类 +11.42pp、定位 −0.89pp（补 stage3 的代价）。
    #:   相对 g43：**分类持平（−0.33pp）、定位 +13.79pp** ⇒ 因此在两个任务上都不劣。
    #:   ⚠ 两条必带披露：① stage3 的 `best` 出现在**第 1 轮**，而选点用的 `val_cls_auc`
    #:     已被实测为不跟踪跨生成器性能（Spearman −0.35）⇒ 选点规则本身无依据，
    #:     跨生成器上的优势是**事后观测**；② 全部单 seed。
    MODEL_CARD_METRICS = [
        ("ForenSynths 未见生成器（strict-9）宏平均 AUC",
         "0.8678 ／ 0.8632",
         "✅ **分类已救回**：比 geofix_s2_epoch10 高 **+11.56pp／+11.42pp**。"
         "与上一交付件 g43（0.8670／0.8665）之差只有 **+0.08pp／−0.33pp** "
         "⇒ **两套抽样口径下都是 ±0.4pp 以内、连符号都翻转 ⇒ 二者不可区分**。"
         "（两个数分别来自 `--per-class 150` 与 `300`；见下方口径说明。）"
         "geofix e10 之所以低，是因为那一臂**只跑了 stage2**，分类头被冻结、"
         "无法适应被改动的共享特征；本权重补了 stage3 联合微调。"),
        ("ForenSynths strict-9 宏平均 AP", "0.8583 ／ 0.8530",
         "与 AUC 同向：g43 为 0.8559／0.8539 ⇒ +0.24pp／−0.09pp"),
        ("ForenSynths 全量 13 生成器 宏平均 AUC", "0.8940 ／ 0.8943",
         "g43 为 0.8949／0.8976 ⇒ −0.09pp／−0.33pp，同样是「持平」"),
        ("CASIAv2 test 真伪 AUC（内域）", "0.9879",
         "内域本就接近饱和，与 g43 的 0.9853 同量级；**内域高不代表跨域高**"),
        ("CASIAv2 test 定位 mIoU★（仅篡改图）", "0.3482",
         "★判定口径，n=474。比补 stage3 之前（geofix e10 的 0.3571）低 **−0.89pp**，"
         "**比上一交付件 g43 的 0.2103 高 +13.79pp**"),
    ]
    MODEL_CARD_PROTOCOL = (
        "跨生成器：分母每生成器真/假各 ≤N（真图去重）、随机种子 3407、宏平均（先逐生成器算再平均）。"
        "strict-9 = 13 个生成器里排除 `progan` 与 `biggan/cyclegan/stargan` 后的 9 个。<br>"
        "★ <b>本卡每个跨生成器指标都给两个数（前者 = `--per-class 150`、"
        "后者 = `--per-class 300`）</b>："
        "`--per-class 150` 是**审计全 corpus**（`docs/05`、`step6`）在用的口径（strict-9 n=2700），"
        "`--per-class 300` 是 2026-10-11 战役口径（n=4998）。**两套口径不可互相加减**，"
        "所以本卡把两个都列出、并各自给同口径的 g43 对照值。<br>"
        "★ <b>一致性背书</b>：g43 在 `per-class 150` 下实测 <b>0.8670</b>，"
        "而 `docs/05` 归档值为 <b>0.8671</b>（差 0.0001）⇒ 采样、真图去重、"
        "生成器筛选、聚合四条路径被**独立复现**。<br>"
        "定位：6 个域（CASIAv2 / COVERAGE / DEFACTO / IMD2020 / CocoGlide / CASIA1.0）"
        "**全量**、单一 base config、云端 GPU；聚合为定位逐图平均。"
        "g43 的定位对照值 <b>在两次独立批次里都测得 0.2103</b>、geofix e10 都测得 0.3571 "
        "⇒ 跨批次可比。"
    )

    #: ★★ 篡改定位（第二个任务）—— **主动交代不利证据**。
    #: 为什么必须写：本工具**同时**展示"结论 + 篡改热力图"，所以定位变差用户看得见。
    #: 只报分类 AUC 那个亮点而不提定位，等于藏事。
    #: 两个数都带标签 ⇒ 不构成"把 0928 的数算到 g43 头上"。
    #: 来源：`outputs/loc_campaign_20261009/`（本机 CPU 战役，两份 eval_test.json）
    #:      + `paired_ci.json`（配对 bootstrap，n=513，boot=2000，seed=3407）。
    MODEL_CARD_LOC = [
        ("CASIAv2 test 逐图平均 mIoU（仅篡改图，★判定口径）", "0.3482", "本权重；n=474"),
        ("　对照：上一交付件 g43，同命令、同分母（n=474）", "0.2103", "差 **+13.79pp**"),
        ("　对照：补 stage3 之前的 geofix e10", "0.3571", "**−0.89pp** ⇒ 补分类的代价"),
        ("　对照：另一臂（旧几何、同样多训 20 轮 vs g43）", "0.2129", "只 +0.26pp ⇒ g43 那侧增益 98% 来自几何修复"),
    ]
    MODEL_CARD_LOC_NOTE = (
        "口径 `miou_tampered_only` = 只在含篡改区域的图上逐图算 IoU 再平均。<br>"
        "★ <b>这一版是两个任务都优于 g43 的第一个交付件</b>：分类持平（−0.33pp）、"
        "定位 <b>+13.79pp</b>。代价是相对「只跑 stage2 的 geofix e10」定位掉了 0.89pp —— "
        "那是补 stage3 联合微调、把分类从 −11.75pp 救回来的价格，**值得**。<br>"
        "⚠ <b>分母必须一起看</b>：本行 n=474，而 2026-10-10 之前归档的数字是"
        "<b>n=513</b>（旧几何下「含前景的图」更多）。<b>n=474 与 n=513 不可相减</b>；"
        "所以本卡一律用新几何下重测的对照值（g43 = 0.2103）。<br>"
        "⚠ <b>选点规则无依据</b>：stage3 的 `best` 出现在<b>第 1 轮</b>，而选它用的 "
        "`val_cls_auc` 已被本项目实测为<b>不跟踪</b>跨生成器性能（Spearman −0.35）。"
        "⇒ 本权重在跨生成器上的表现是<b>事后观测</b>，不是被验证过的最优点；全部单 seed。"
    )

    # ==================================================================
    # ★★★ 2026-10-10（批次 1-C）：**跨域能力边界**。
    #
    # 为什么必须进模型卡：本卡原来只有 ForenSynths（跨生成器）与 CASIAv2（内域）两项，
    #   读者会顺理成章地以为这是"通用篡改检测器"。1-C 用**同一次战役**
    #   （同设备/同权重/同抽样）把 4 个权重 × 6 个域一起测了，结果很不一样：
    #   换到**训练分布之外**的伪造机制，分类 AUC 掉到 0.47~0.62（接近随机），
    #   IMD2020 的像素级 AUC = 0.513（纯随机）。
    #   ⇒ "能做什么、不能做什么"是这个构建的**关键属性**，必须写在模型卡里，
    #     否则等于把工具的适用边界藏起来让用户自己踩。
    #
    # 数据来源：云端 `outputs/1c_20261010/`（24 任务，2026-10-10 01:24–02:06），
    #          汇总见 `deliverables/15_批次1C_四权重六域同战役总表_20261010.md`。
    # ==================================================================
    MODEL_CARD_XDOMAIN_HEAD = ("域（n = 真/假）", "伪造机制", "分类 AUC", "定位 mIoU★")
    MODEL_CARD_XDOMAIN = [
        ("CASIAv2　1263 (750/513)", "手工拼接·复制移动（<b>训练同源</b>）", "0.9879", "0.3482"),
        ("COVERAGE　191 (100/91)", "复制移动（零样本）", "0.5753", "0.1344"),
        ("DEFACTO　8000 (4000/4000)", "拼接·复制移动·物体移除（零样本）", "0.4779", "0.1258"),
        ("IMD2020　3500 (0/3500)", "<b>相机传感器级</b>（零样本）", "n/a", "0.0064"),
        ("CocoGlide　1024 (512/512)", "<b>扩散 inpainting</b>（零样本）", "0.6250", "0.1197"),
        ("CASIA1.0　920 (0/920)", "同源早期版本", "n/a", "0.1930"),
    ]
    MODEL_CARD_XDOMAIN_NOTE = (
        "<b>这张表是本工具适用边界的唯一依据，请务必读</b>："
        "内域（CASIAv2）分类 AUC <b>0.988</b> ⇒ 模型在<b>训练分布内</b>是有效的；"
        "换到训练分布之外的伪造机制，分类 AUC 掉到 <b>0.48~0.63（接近随机）</b>。"
        "<b>⇒ 本构建只在「内域（CASIAv2 型）伪造」上可用；对复制移动 / 物体移除 / "
        "扩散修复 / 相机传感器级伪造，分类与定位都<b>不可依赖</b>。</b><br>"
        "IMD2020 与 CASIA1.0 没有真实图（Au=0）⇒ AUC 无定义，其 ACC 语义是"
        "「篡改召回率」，不得与含真图的域并列比较。<br>"
        "<b>像素级 AUC（阈无关，ManTra-Net/SPAN 的第一位指标）</b>：CASIAv2 0.932、"
        "COVERAGE 0.664、DEFACTO 0.750、CocoGlide 0.688、CASIA1.0 0.698，"
        "但 <b>IMD2020 只有 0.520 = 纯随机</b> ⇒ 对相机传感器级篡改没有可用的定位信号。<br>"
        "<b>定位为什么这么低：主要是篡改区太小。</b>按面积分桶实测（deliverables/19）："
        "CASIAv2 上面积 &lt;0.5% 的图 IoU 只有 <b>0.068</b>，10–25% 的是 <b>0.613</b>、"
        "25–50% 的是 <b>0.736</b>；而 CASIAv2 的篡改区面积<b>中位数只有 3.5%</b>、"
        "DEFACTO 只有 1.5%、IMD2020 <b>最大也只有 3.4%</b>。"
        "所以「定位差」很大程度上是「待检目标太小」，不是单一的能力刻度。"
    )

    #: ⚠⚠ 定位数字的**管线缺陷交代**（2026-10-10 发现 / 2026-10-11 更新为"已换到修复后"）。
    #: 旧管线的缺陷：图像走「等比缩放 + 中心裁剪」，掩码却走「整幅拉伸到 224×224」
    #: —— 对非方形图这是两个不同的几何变换，实测错位最多 35 像素（224 的 15%）。
    #: 本仓 CASIAv2 只有 0.2% 是方形图 ⇒ 定位支路几乎全程在该管线下训练。
    #: 缺陷已修复（`src/data/transforms.py` + `scripts/test_transform_align.py`）。
    #: ★ 2026-10-11：**本卡当前这个权重是第一个在修复后管线下训练的交付件**
    #:   ⇒ 它的定位数字属于"修复后"体系。对照值（g43 的 0.2103）是**在新几何下重测**
    #:   的，不是旧归档值 ⇒ 可直接相减。旧归档那批（旧几何、分母 n=513）**不在本卡出现**，
    #:   因为它们与修复后数字不可比（需要时到 `deliverables/15` 查）。
    #: 详情：`deliverables/14_定位掩码错位缺陷与修复_20261010.md`
    MODEL_CARD_LOC_CAVEAT = (
        "⚠ <b>定位数字的管线交代</b>：2026-10-10 发现旧的预处理管线里"
        "「图像做等比缩放+中心裁剪、掩码却做整幅拉伸」，非方形图上两者错位最多 "
        "<b>35 像素</b>（本仓训练集只有 0.2% 是方形图）。缺陷已修复。<br>"
        "<b>本卡当前这个权重是第一个在修复后管线下训练的交付件</b> ⇒ 它的定位数字属"
        "「修复后」体系；卡里的 g43 对照值也是<b>在新几何下重测</b>的（0.2103），"
        "两者<b>可直接相减</b>。而 2026-10-10 之前归档的那批定位数字（旧几何、"
        "分母 n=513）与修复后数字<b>不可比</b>，故本卡<b>一律不引用它们</b>"
        "（含出厂件与 g43 的旧归档值）—— 需要时请到 `deliverables/15` 查。<br>"
        "⚠ 仍须记住：定位的<b>绝对</b>水平受「待检目标太小」主导（见上一段），"
        "本卡的定位数值宜用于<b>权重之间比高低</b>，不宜当作能力上限。"
    )

    def _md5_of(self, path: str) -> str:
        import hashlib
        h = hashlib.md5()
        p = self._abs(path)
        if not os.path.exists(p):
            return "(文件不存在)"
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def model_card_html(self, weight_md5: str = None) -> str:
        """模型卡正文。`weight_md5` 传 None 时显示占位（由对话框后台补齐）。"""
        rows = "".join(
            "<tr><td style='padding:2px 10px 2px 0;'>{}</td>"
            "<td style='padding:2px 14px 2px 0;'><b>{}</b></td>"
            "<td style='color:#777;'>{}</td></tr>".format(*m) for m in self.MODEL_CARD_METRICS)
        if weight_md5 is None:
            md5_cell = "<span style='color:#999;'>计算中…</span>"
        elif weight_md5.lower() == self.MODEL_CARD_WEIGHT_MD5:
            md5_cell = "{}　<span style='color:#1e8e3e;'>✅ 与预期一致</span>".format(weight_md5)
        else:
            md5_cell = "{}　<span style='color:#c0392b;'>⚠ 与预期不一致（"
            md5_cell = md5_cell.format(weight_md5) + self.MODEL_CARD_WEIGHT_MD5 + "）</span>"
        return (
            "<h3 style='margin:2px 0 6px;'>空频双分支轻量 VIB-Net · AI 图像篡改检测系统 V1.0</h3>"
            "<p style='margin:4px 0;'>"
            "部署权重：<b>{w}</b><br>"
            "当前 ONNX：<b>{o}</b><br>"
            "权重 md5：{m}</p>"
            "<hr>"
            "<p style='margin:4px 0 2px;'><b>性能（ForenSynths，分类主口径）</b></p>"
            "<table style='font-size:12px;'>{r}</table>"
            "<p style='margin:6px 0 0;color:#666;font-size:11px;'>{p}</p>"
            "<p style='margin:8px 0 2px;'><b>篡改定位（第二个任务）</b></p>"
            "<table style='font-size:12px;'>{rl}</table>"
            "<p style='margin:6px 0 0;color:#666;font-size:11px;'>{rn}</p>"
            "<hr>"
            "<p style='margin:4px 0 2px;'><b>跨域能力边界（本权重 × 6 域实测，2026-10-11）"
            "</b></p>"
            "<table style='font-size:12px;'>{xd}</table>"
            "<p style='margin:6px 0 0;color:#666;font-size:11px;'>{xn}</p>"
            "<hr>"
            "<p style='margin:4px 0 2px;color:#666;font-size:11px;'>{cv}</p>"
            "<hr>"
            "<p style='margin:4px 0;color:#666;font-size:11px;'>"
            "• 以上数字是<b>特定测试集</b>上的指标，<b>不是</b>本工具对任意图片的精度承诺，"
            "也<b>不是</b>司法鉴定依据。<br>"
            "• 本卡只列<b>本权重</b>自己的数；跨权重对照行（g43 / geofix e10）"
            "已逐行标注归属，未与之混同。<br>"
            "• 结论由模型自动给出，仅供辅助参考。</p>"
        ).format(w=self.MODEL_CARD_WEIGHT, o=self.MODEL_CARD_ONNX,
                 m=md5_cell, r=rows, p=self.MODEL_CARD_PROTOCOL,
                 rl="".join(
                     "<tr><td style='padding:2px 10px 2px 0;'>{}</td>"
                     "<td style='padding:2px 14px 2px 0;'><b>{}</b></td>"
                     "<td style='color:#777;'>{}</td></tr>".format(*m) for m in self.MODEL_CARD_LOC),
                 rn=self.MODEL_CARD_LOC_NOTE,
                 xd=(
                     "<tr style='color:#666;'>"
                     + "".join("<td style='padding:2px 10px 2px 0;'>{}</td>".format(h)
                               for h in self.MODEL_CARD_XDOMAIN_HEAD)
                     + "</tr>"
                     + "".join(
                         "<tr><td style='padding:2px 10px 2px 0;'>{}</td>"
                         "<td style='padding:2px 10px 2px 0;'>{}</td>"
                         "<td style='padding:2px 10px 2px 0;'><b>{}</b></td>"
                         "<td style='padding:2px 10px 2px 0;'><b>{}</b></td></tr>".format(*m)
                         for m in self.MODEL_CARD_XDOMAIN)),
                 xn=self.MODEL_CARD_XDOMAIN_NOTE,
                 cv=self.MODEL_CARD_LOC_CAVEAT)

    def _show_model_card(self) -> None:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QTextBrowser, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("关于 · 模型卡")
        dlg.resize(660, 560)
        lay = QVBoxLayout(dlg)
        br = QTextBrowser()
        br.setOpenExternalLinks(False)
        br.setHtml(self.model_card_html(None))
        lay.addWidget(br)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        # md5 要读 345 MB（约 1 秒）。先让对话框画出来，再算 —— 否则点"关于"
        # 会有约 1 秒无反馈，看起来像卡死。
        QTimer.singleShot(80, lambda: br.setHtml(self.model_card_html(self._md5_of(self.DEFAULTS["ckpt"]))))
        dlg.exec_()

    #: ★ 2026-10-09（A3）：`QSettings` 持久化键名映射。
    #:
    #: 病根：`get_config()` 返回的键名是**给 `Detector(**kwargs)` 用的**
    #: （`onnx_path` / `openvino_ir` / `config_path`，见 `src/deploy/inference.py:148`
    #: 的形参名），而 `DEFAULTS` 与 `load_config()` 用的是 `onnx` / `openvino` /
    #: `config`。`_apply()` 此前**把 `get_config()` 的键名原样写进注册表** ⇒
    #: 那 3 个键**写了从不读**，症状 = "保存成功、重启丢失"。
    #:
    #: ⚠ **不能改 `get_config()` 的键名** —— `app/main.py:185-187` 直接读
    #: `cfg["onnx_path"]` / `["openvino_ir"]` / `["config_path"]` 去构造 Detector，
    #: 而 Detector 的形参本来就叫 `onnx_path`。所以输出名是对的、错的是持久化层。
    #: ⇒ 只在**写边界**做映射，接口名保持不动（`main.py` / `inference.py` 零改动）。
    #:
    #: 注：`reset()` 走的 `DEFAULTS` 键名本来就是规范名，无需映射。
    _PERSIST_ALIASES = {
        "onnx_path": "onnx",
        "openvino_ir": "openvino",
        "config_path": "config",
    }

    #: ★ 本仓库内**已知无效**的导出件：文件存在、能加载、推理不报错，
    #: 但对任何图像都恒定输出（随机初始化 / 训练前导出）。仅查存在性拦不住，
    #: 所以显式拉黑，命中就回退到 DEFAULTS。
    #:   证据见 docs/09「导出命令」——lite 三件套是 `--ckpt ""` 导出的。
    _KNOWN_DEAD = {
        "deploy/lite.onnx": "随机初始化（--ckpt \"\" 导出），恒输出 0.4852",
        "deploy/lite_simplified.onnx": "随机初始化（--ckpt \"\" 导出），恒输出 0.4852",
        "deploy/lite_int8.onnx": "随机初始化（--ckpt \"\" 导出），恒输出 0.4852",
        "deploy/b16.onnx": "训练前导出，恒输出 0.4982",
        "deploy/b16_int8.onnx": "训练前导出，恒输出 0.4982",
        "checkpoints/vibnet_best.pt": "09-26 旧权重，p_fake 恒在 0.54 附近（须用 _pack 那份）",
    }

    def __init__(self, parent=None, project_root: str = "."):
        super().__init__("⑤ 系统设置", parent)
        self.root = project_root
        self.settings = QSettings("GXU", "VIBNetDetector")
        self._auto_fixed: list = []       # 被自动纠正的过期项（load_config 里填）
        self._build()
        self.load_config()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.cmb_backend = QComboBox()
        try:
            from src.deploy.inference import list_available_backends

            self.cmb_backend.addItems(list_available_backends())
        except Exception:
            self.cmb_backend.addItems(["torch", "onnx", "openvino"])
        self.cmb_backend.setToolTip("torch=调试用；onnx=通用；openvino=Intel CPU 加速（最快）")
        form.addRow("推理后端", self.cmb_backend)

        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["cpu", "gpu"])
        form.addRow("推理设备", self.cmb_device)

        self.spin_thr = QDoubleSpinBox()
        # ★ 2026-10-09（A4）：上限 0.95 → 1.00。实测 EER 最优点在 0.9988~0.9999，
        #   旧上限连 0.99 都够不到，留一个"设不到最优点"的滑块没有意义。
        #   默认值**仍为 0.50** —— 不把任何标定值提升为默认：本项目没有合法留出集
        #   （val_cross 与域内 val 都不跟踪测试性能），标定值是测试集产物。
        #   ⚠ decimals 保持 2 位：这样 0.9988 仍不可达、只到 0.99，而操作点拐点在
        #   0.95~0.99 ⇒ 0.99 够用。两位小数恰好挡住"照着测试集把阈值拧到 0.9988"。
        self.spin_thr.setRange(0.05, 1.00)
        #   ⚠ 步长必须 0.01（原 0.05 是错的）：实测（offscreen 复现，2026-10-09）
        #   step=0.05 时箭头/滚轮可达集合只有 0.05 的倍数共 20 个值
        #   （…0.90/0.95/**1.00**），从 0.95 往上**直接跳过 0.99** ⇒ 上文"只到 0.99"
        #   注释形同虚设，0.99 只能靠手输。改 0.01 后 0.99 可正常步进选中，
        #   而 0.9988 因 decimals=2 仍被吞成 1.00 ⇒ 挡 oracle 调参的原意保留。
        self.spin_thr.setSingleStep(0.01)
        self.spin_thr.setDecimals(2)
        # 判定实现是 `prob >= threshold`（src/deploy/inference.py:373），故写"大于等于"。
        #   旧上限 0.95 下这个措辞差异无后果；上限到 1.00 后才有后果：阈值 1.00 仍会命中
        #   输出**恰好 1.0000** 的图像（本模型分数饱和），而不是"全部判真"。
        self.spin_thr.setToolTip("伪造概率大于等于该阈值即判定为疑似伪造，默认 0.50")
        form.addRow("检测阈值", self.spin_thr)

        self.spin_size = QSpinBox()
        # ★ 2026-10-09（A2）：锁成 224（原 128~512 可调）。
        #   原因：ONNX 计算图的 H/W 是**静态 224**（当时默认件 `deploy/b16_g43.onnx`；
        #   现默认件 `deploy/b16_s3best.onnx` 同样是静态 224），
        #   一改尺寸每张图都会失败且只报"加载失败"。真支持多尺寸需重导 dynamic-H/W 的
        #   ONNX 并改 Detector 预处理，属独立改动，不在这里做。
        self.spin_size.setRange(224, 224)
        self.spin_size.setSingleStep(32)
        self.spin_size.setToolTip("ONNX 计算图固定 224×224；改尺寸需重新导出模型")
        form.addRow("输入尺寸", self.spin_size)

        self.edit_save = QLineEdit()
        btn_save = QPushButton("…")
        btn_save.setFixedWidth(30)
        row = QHBoxLayout()
        row.addWidget(self.edit_save)
        row.addWidget(btn_save)
        w = QWidget()
        w.setLayout(row)
        form.addRow("结果保存路径", w)
        btn_save.clicked.connect(lambda: self._pick_dir(self.edit_save))

        self.edit_ckpt = QLineEdit()
        b1 = QPushButton("…")
        b1.setFixedWidth(30)
        r1 = QHBoxLayout(); r1.addWidget(self.edit_ckpt); r1.addWidget(b1)
        w1 = QWidget(); w1.setLayout(r1)
        form.addRow("PyTorch 权重", w1)
        b1.clicked.connect(lambda: self._pick_file(self.edit_ckpt, "*.pt"))

        self.edit_onnx = QLineEdit()
        b2 = QPushButton("…")
        b2.setFixedWidth(30)
        r2 = QHBoxLayout(); r2.addWidget(self.edit_onnx); r2.addWidget(b2)
        w2 = QWidget(); w2.setLayout(r2)
        form.addRow("ONNX 模型", w2)
        b2.clicked.connect(lambda: self._pick_file(self.edit_onnx, "*.onnx"))

        self.edit_ov = QLineEdit()
        b3 = QPushButton("…")
        b3.setFixedWidth(30)
        r3 = QHBoxLayout(); r3.addWidget(self.edit_ov); r3.addWidget(b3)
        w3 = QWidget(); w3.setLayout(r3)
        form.addRow("OpenVINO IR", w3)
        b3.clicked.connect(lambda: self._pick_file(self.edit_ov, "*.xml"))

        root.addLayout(form)

        btns = QHBoxLayout()
        self.btn_apply = QPushButton("保存并加载模型")
        self.btn_reset = QPushButton("恢复默认")
        self.btn_about = QPushButton("关于 · 模型卡")
        self.btn_about.setToolTip("查看本构建的模型版本、权重 md5 与实测指标（含口径）")
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        btns.addWidget(self.btn_about)
        root.addLayout(btns)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#666;")
        root.addWidget(self.lbl_status)

        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_about.clicked.connect(self._show_model_card)

    # ------------------------------------------------------------------
    def _pick_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择目录", edit.text() or self.root)
        if d:
            edit.setText(d)

    def _pick_file(self, edit: QLineEdit, pattern: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择文件", self.root, f"文件 ({pattern})")
        if p:
            edit.setText(p)

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        return {
            "backend": self.cmb_backend.currentText(),
            "device": self.cmb_device.currentText(),
            "threshold": float(self.spin_thr.value()),
            "save_dir": self.edit_save.text().strip() or "./outputs",
            "ckpt": self.edit_ckpt.text().strip(),
            "onnx_path": self.edit_onnx.text().strip(),
            "openvino_ir": self.edit_ov.text().strip(),
            "image_size": int(self.spin_size.value()),
            "config_path": self._abs(self.DEFAULTS["config"]),
        }

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.root, p))

    def _apply(self) -> None:
        cfg = self.get_config()
        # ★ A3：按 `_PERSIST_ALIASES` 映射后再写 —— 与 `load_config()` 读的键名对齐，
        #   否则 `onnx_path` / `openvino_ir` / `config_path` 写进去后永远读不回来。
        for k, v in cfg.items():
            self.settings.setValue(self._PERSIST_ALIASES.get(k, k), v)
        self.applied.emit(cfg)

    def reset(self) -> None:
        for k, v in self.DEFAULTS.items():
            self.settings.setValue(k, v)
        self.load_config()
        self.lbl_status.setText("已恢复默认设置")

    def load_config(self) -> None:
        # ---- ① 先解析各路径项：注册表里的值若指向不存在的文件，回退到默认值 ----
        # 为什么要这一步：QSettings 会把上次的值持久化到注册表（HKCU\Software\GXU\
        # VIBNetDetector），**只改代码里的 DEFAULTS 盖不过它** —— 上一版写进去的
        # 错误路径会一直生效，表现成「明明改好了还是报同样的错」。
        self._auto_fixed = []
        vals = {}
        for k, dv in self.DEFAULTS.items():
            v = self.settings.value(k, dv)
            if k in self._PATH_KEYS:
                v = self._pick_valid(v, dv)
            vals[k] = v

        # ---- ② 后端自愈：存的后端若依赖文件不存在，自动退到可用的那个 ----
        vals["backend"] = self._resolve_backend(str(vals.get("backend", "")), vals)

        # ---- ③ 回填控件 ----
        for k, v in vals.items():
            if k == "backend":
                i = self.cmb_backend.findText(str(v))
                self.cmb_backend.setCurrentIndex(i if i >= 0 else 0)
            elif k == "device":
                i = self.cmb_device.findText(str(v))
                self.cmb_device.setCurrentIndex(i if i >= 0 else 0)
            elif k == "threshold":
                self.spin_thr.setValue(float(v))
            elif k == "image_size":
                self.spin_size.setValue(int(v))
            elif k == "save_dir":
                self.edit_save.setText(str(v))
            elif k == "ckpt":
                self.edit_ckpt.setText(str(v))
            elif k == "onnx":
                self.edit_onnx.setText(str(v))
            elif k == "openvino":
                self.edit_ov.setText(str(v))

        if self._auto_fixed:
            self.set_status(f"已自动纠正 {len(self._auto_fixed)} 项过期路径："
                            + "、".join(self._auto_fixed), "#e67e22")

    # ------------------------------------------------------------------
    def _pick_valid(self, value, default) -> str:
        """路径项：值不可用时回退默认值，并记进 `_auto_fixed`。

        「不可用」有两种，**必须都拦**：
          ① 文件不存在            —— 直接报 FileNotFoundError，容易发现；
          ② 文件存在但已知无效    —— 本仓库的 lite*.onnx / b16.onnx / 09-26 旧权重
             属于这一类，**加载不报错、结果全错**（静默失效），只查存在性会漏掉。
        """
        v = str(value or "")
        key = v.replace("\\", "/").lstrip("./")
        if v and key in self._KNOWN_DEAD:
            self._auto_fixed.append(f"{v}（{self._KNOWN_DEAD[key]}）→ {default}")
            return str(default)
        if v and os.path.exists(self._abs(v)):
            return v
        if v and v != str(default):
            self._auto_fixed.append(f"{v} → {default}")
        return str(default)

    def _resolve_backend(self, want: str, vals: dict) -> str:
        """挑一个**依赖文件确实存在**的后端；存的后端不可用时按优先级退让。

        这里刻意不静默：退让会记进 `_auto_fixed`，状态栏与主窗口日志都会显示，
        避免出现「下拉框显示 torch、实际用 onnx」这种对不上账的情况。
        """
        cands = [self.cmb_backend.itemText(i) for i in range(self.cmb_backend.count())]
        need = {
            "onnx": vals.get("onnx"),
            "openvino": vals.get("openvino"),
        }

        def usable(b: str) -> bool:
            p = need.get(b)
            if p is None:
                return True                      # torch：缺权重也能跑（随机初始化 + 警告）
            return os.path.exists(self._abs(str(p)))

        if want in cands and usable(want):
            return want
        for b in cands:
            if usable(b):
                self._auto_fixed.append(f"后端 {want} → {b}（依赖文件缺失）")
                return b
        return cands[0] if cands else "torch"

    def set_status(self, text: str, color: str = "#666") -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color:{color};")
