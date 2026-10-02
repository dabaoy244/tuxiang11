# 跨生成器泛化评测（ablation_full_best.pt）

- 配置：`configs/default_crossval.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/full/ckpt/ablation_full_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 255 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6333 | 0.3733 | 0.8933 | 0.8563 | 0.8965 |
| deepfake | 150/150 | 0.6167 | 0.3533 | 0.8800 | 0.7885 | 0.7819 |
| gaugan | 150/150 | 0.6533 | 0.3067 | 1.0000 | 0.9852 | 0.9857 |
| imle | 150/150 | 0.6367 | 0.3800 | 0.8933 | 0.8524 | 0.8845 |
| san | 150/150 | 0.5300 | 0.7133 | 0.3467 | 0.5603 | 0.5708 |
| seeingdark | 150/150 | 0.6567 | 0.4133 | 0.9000 | 0.8271 | 0.8519 |
| stylegan | 150/150 | 0.9733 | 1.0000 | 0.9467 | 0.9975 | 0.9980 |
| stylegan2 | 150/150 | 0.8967 | 0.9867 | 0.8067 | 0.9802 | 0.9821 |
| whichfaceisreal | 150/150 | 0.7033 | 0.7533 | 0.6533 | 0.8070 | 0.8150 |
| **宏平均** | — | **0.7000** | **0.5867** | **0.8133** | **0.8505** | **0.8629** |
| **微平均（合并全部样本）** | 1350/1350 | **0.7000** | 0.5867 | 0.8133 | 0.8272 | 0.8341 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/default_crossval.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/full/ckpt/ablation_full_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/full/strict --tag full_strict`