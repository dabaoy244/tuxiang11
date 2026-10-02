# 跨生成器泛化评测（ablation_no_vib_best.pt）

- 配置：`configs/_auto_no_vib.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_vib/ckpt/ablation_no_vib_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 281 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7800 | 0.5733 | 0.9867 | 0.9563 | 0.9547 |
| crn | 150/150 | 0.6600 | 0.6800 | 0.6400 | 0.7176 | 0.7532 |
| cyclegan | 150/150 | 0.5567 | 0.1133 | 1.0000 | 0.8980 | 0.8341 |
| deepfake | 150/150 | 0.6700 | 0.5467 | 0.7933 | 0.7474 | 0.7115 |
| gaugan | 150/150 | 0.6433 | 0.2867 | 1.0000 | 0.9427 | 0.9257 |
| imle | 150/150 | 0.7267 | 0.6533 | 0.8000 | 0.8410 | 0.8544 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5400 | 0.7467 | 0.3333 | 0.5681 | 0.5662 |
| seeingdark | 150/150 | 0.7133 | 0.4600 | 0.9667 | 0.9195 | 0.8997 |
| stargan | 150/150 | 0.8000 | 0.6000 | 1.0000 | 0.9957 | 0.9918 |
| stylegan | 150/150 | 0.9967 | 1.0000 | 0.9933 | 1.0000 | 1.0000 |
| stylegan2 | 150/150 | 0.9067 | 0.9933 | 0.8200 | 0.9323 | 0.9560 |
| whichfaceisreal | 150/150 | 0.6933 | 0.6333 | 0.7533 | 0.7931 | 0.8012 |
| **宏平均** | — | **0.7451** | **0.6374** | **0.8528** | **0.8701** | **0.8653** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7451** | 0.6374 | 0.8528 | 0.8610 | 0.8618 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_vib.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_vib/ckpt/ablation_no_vib_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_vib --tag no_vib --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_vib/scores.npz`