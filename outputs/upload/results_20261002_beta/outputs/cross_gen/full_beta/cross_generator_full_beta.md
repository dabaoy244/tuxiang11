# 跨生成器泛化评测（ablation_full_best.pt）

- 配置：`configs/_auto_beta_full.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/full/ckpt/ablation_full_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 276 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7467 | 0.4933 | 1.0000 | 0.9742 | 0.9739 |
| crn | 150/150 | 0.6633 | 0.3400 | 0.9867 | 0.9688 | 0.9733 |
| cyclegan | 150/150 | 0.5267 | 0.0533 | 1.0000 | 0.9547 | 0.9463 |
| deepfake | 150/150 | 0.5933 | 0.3733 | 0.8133 | 0.7076 | 0.7479 |
| gaugan | 150/150 | 0.6233 | 0.2467 | 1.0000 | 0.9758 | 0.9759 |
| imle | 150/150 | 0.6367 | 0.2867 | 0.9867 | 0.9643 | 0.9743 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.4900 | 0.6867 | 0.2933 | 0.5183 | 0.5537 |
| seeingdark | 150/150 | 0.6833 | 0.3867 | 0.9800 | 0.8983 | 0.8952 |
| stargan | 150/150 | 0.8933 | 0.7867 | 1.0000 | 0.9993 | 0.9994 |
| stylegan | 150/150 | 0.9900 | 1.0000 | 0.9800 | 0.9977 | 0.9981 |
| stylegan2 | 150/150 | 0.9067 | 0.9933 | 0.8200 | 0.9729 | 0.9786 |
| whichfaceisreal | 150/150 | 0.6033 | 0.5333 | 0.6733 | 0.7288 | 0.7668 |
| **宏平均** | — | **0.7197** | **0.5523** | **0.8872** | **0.8970** | **0.9064** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7197** | 0.5523 | 0.8872 | 0.8844 | 0.9055 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_beta_full.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/full/ckpt/ablation_full_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/full_beta --tag full_beta --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/full_beta/scores.npz`