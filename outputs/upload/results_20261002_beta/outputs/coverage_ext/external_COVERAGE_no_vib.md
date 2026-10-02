# 外部域独立基准行：COVERAGE（零样本，未参与训练）

- 权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_vib/ckpt/ablation_no_vib_best.pt`　设备：`cpu`
- 样本：100 真实 + 91 篡改 = **191** 张（定位池 191 张，其中 GT 真有前景 91 张）
- 口径来源：`configs/_auto_no_vib.yaml` 的模型/评测段；仅 `data.root` 换成`data/external/COVERAGE`
- ⚠ 复制-移动（copy-move）篡改，官方 100 对；与训练域（ForenSynths 生成图 / CASIAv2 拼接·复制移动）**不同源**，全程未参与训练 ⇒ 这是**零样本跨数据集**结果。

| 指标 | 值 |
|---|---|
| 真伪 ACC | **0.5340** |
| 真伪 AUC | **0.5570** |
| AP | 0.5543 |
| F1 | 0.5189 |
| 真实图召回（真判真） | 0.5400 |
| 篡改图召回（假判假） | 0.5275 |
| 定位 mIoU（池化，宽松） | 0.0821 |
| 定位 mIoU（仅篡改图，★文献标准） | **0.0658** |
| 定位 mIoU（全部非空图，严格） | 0.0468 |
| Dice | 0.1517 |
| Pixel Acc | 0.9247 |

> 复现命令：`python scripts/eval_external_domain.py --domain COVERAGE --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_vib/ckpt/ablation_no_vib_best.pt --base-config configs/_auto_no_vib.yaml --device cpu --cpu-threads 8 --expect 191 --out outputs/coverage_ext --tag no_vib`

> ⚠️ **口径红旗**：定位池含 100 张全零 GT 的真实图（掩码文件存在但无前景）。只有 miou_tampered_only 跨数据集可比；其余三个含真实图像素，受误报影响，不可与 CASIAv2 的同名指标并列。
> 报数请只报 `miou_tampered_only`（= 上表★行），并注明它是91 张篡改图上的逐图 IoU 平均。
