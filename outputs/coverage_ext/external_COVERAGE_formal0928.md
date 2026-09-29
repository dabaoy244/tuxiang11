# 外部域独立基准行：COVERAGE（零样本，未参与训练）

- 权重：`checkpoints/vibnet_best.pt`　设备：`cpu`
- 样本：100 真实 + 91 篡改 = **191** 张（定位池 191 张，其中 GT 真有前景 91 张）
- 口径来源：`configs/default_crossval.yaml` 的模型/评测段；仅 `data.root` 换成`data/external/COVERAGE`
- ⚠ 复制-移动（copy-move）篡改，官方 100 对；与训练域（ForenSynths 生成图 / CASIAv2 拼接·复制移动）**不同源**，全程未参与训练 ⇒ 这是**零样本跨数据集**结果。

| 指标 | 值 |
|---|---|
| 真伪 ACC | **0.5026** |
| 真伪 AUC | **0.5746** |
| AP | 0.5403 |
| F1 | 0.6332 |
| 真实图召回（真判真） | 0.1400 |
| 篡改图召回（假判假） | 0.9011 |
| 定位 mIoU（池化，宽松） | 0.1333 |
| 定位 mIoU（仅篡改图，★文献标准） | **0.1598** |
| 定位 mIoU（全部非空图，严格） | 0.0836 |
| Dice | 0.2352 |
| Pixel Acc | 0.8696 |

> 复现命令：`python scripts/eval_external_domain.py --domain COVERAGE --ckpt checkpoints/vibnet_best.pt --device cpu --cpu-threads 8 --expect 191 --out outputs/coverage_ext --tag ckpt0928`

> ⚠️ **口径红旗**：定位池含 100 张全零 GT 的真实图（掩码文件存在但无前景）。只有 miou_tampered_only 跨数据集可比；其余三个含真实图像素，受误报影响，不可与 CASIAv2 的同名指标并列。
> 报数请只报 `miou_tampered_only`（= 上表★行），并注明它是91 张篡改图上的逐图 IoU 平均。
