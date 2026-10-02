# 跨生成器泛化评测（ablation_no_vib_best.pt）

- 配置：`configs/_auto_beta_no_vib.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/no_vib/ckpt/ablation_no_vib_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 275 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7500 | 0.5067 | 0.9933 | 0.9642 | 0.9609 |
| crn | 150/150 | 0.6433 | 0.3467 | 0.9400 | 0.9000 | 0.9132 |
| cyclegan | 150/150 | 0.5333 | 0.0667 | 1.0000 | 0.8774 | 0.8110 |
| deepfake | 150/150 | 0.5900 | 0.5800 | 0.6000 | 0.6637 | 0.6948 |
| gaugan | 150/150 | 0.6233 | 0.2467 | 1.0000 | 0.9456 | 0.9232 |
| imle | 150/150 | 0.6600 | 0.3467 | 0.9733 | 0.8743 | 0.8817 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5033 | 0.7000 | 0.3067 | 0.4977 | 0.5152 |
| seeingdark | 150/150 | 0.6300 | 0.2600 | 1.0000 | 0.9355 | 0.9208 |
| stargan | 150/150 | 0.8933 | 0.7867 | 1.0000 | 1.0000 | 1.0000 |
| stylegan | 150/150 | 0.9900 | 1.0000 | 0.9800 | 0.9989 | 0.9990 |
| stylegan2 | 150/150 | 0.8867 | 0.9933 | 0.7800 | 0.8982 | 0.9338 |
| whichfaceisreal | 150/150 | 0.7367 | 0.6133 | 0.8600 | 0.8404 | 0.8464 |
| **宏平均** | — | **0.7262** | **0.5728** | **0.8795** | **0.8766** | **0.8769** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7262** | 0.5728 | 0.8795 | 0.8655 | 0.8735 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_beta_no_vib.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/no_vib/ckpt/ablation_no_vib_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_vib_beta --tag no_vib_beta --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_vib_beta/scores.npz`