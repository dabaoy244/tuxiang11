# 跨生成器泛化评测（ablation_full_best.pt）

- 配置：`configs/_auto_beta_full.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/full/ckpt/ablation_full_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 254 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6900 | 0.3800 | 1.0000 | 0.9764 | 0.9755 |
| deepfake | 150/150 | 0.5867 | 0.4267 | 0.7467 | 0.7117 | 0.7531 |
| gaugan | 150/150 | 0.6400 | 0.2800 | 1.0000 | 0.9849 | 0.9846 |
| imle | 150/150 | 0.6333 | 0.2733 | 0.9933 | 0.9680 | 0.9753 |
| san | 150/150 | 0.4800 | 0.6600 | 0.3000 | 0.5166 | 0.5432 |
| seeingdark | 150/150 | 0.6667 | 0.3533 | 0.9800 | 0.8834 | 0.8881 |
| stylegan | 150/150 | 0.9867 | 1.0000 | 0.9733 | 0.9976 | 0.9980 |
| stylegan2 | 150/150 | 0.9167 | 0.9867 | 0.8467 | 0.9732 | 0.9774 |
| whichfaceisreal | 150/150 | 0.6400 | 0.5933 | 0.6867 | 0.7630 | 0.7927 |
| **宏平均** | — | **0.6933** | **0.5504** | **0.8363** | **0.8638** | **0.8764** |
| **微平均（合并全部样本）** | 1350/1350 | **0.6933** | 0.5504 | 0.8363 | 0.8452 | 0.8716 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_beta_full.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/full/ckpt/ablation_full_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/full_beta/strict --tag full_beta_strict`