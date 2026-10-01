# 跨生成器泛化评测（ablation_no_cross_attention_best.pt）

- 配置：`configs/_auto_no_cross_attention.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_cross_attention/ckpt/ablation_no_cross_attention_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 274 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7767 | 0.5667 | 0.9867 | 0.9731 | 0.9699 |
| crn | 150/150 | 0.6167 | 0.2800 | 0.9533 | 0.7961 | 0.8001 |
| cyclegan | 150/150 | 0.5267 | 0.0600 | 0.9933 | 0.9315 | 0.9079 |
| deepfake | 150/150 | 0.6000 | 0.5133 | 0.6867 | 0.7082 | 0.7164 |
| gaugan | 150/150 | 0.6733 | 0.3467 | 1.0000 | 0.9651 | 0.9610 |
| imle | 150/150 | 0.6400 | 0.2933 | 0.9867 | 0.9524 | 0.9500 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5067 | 0.7067 | 0.3067 | 0.5885 | 0.5650 |
| seeingdark | 150/150 | 0.6933 | 0.5200 | 0.8667 | 0.8365 | 0.8249 |
| stargan | 150/150 | 0.9067 | 0.8200 | 0.9933 | 0.9982 | 0.9986 |
| stylegan | 150/150 | 0.9733 | 1.0000 | 0.9467 | 0.9991 | 0.9991 |
| stylegan2 | 150/150 | 0.8400 | 0.9933 | 0.6867 | 0.9748 | 0.9771 |
| whichfaceisreal | 150/150 | 0.7300 | 0.6800 | 0.7800 | 0.8374 | 0.8532 |
| **宏平均** | — | **0.7295** | **0.5985** | **0.8605** | **0.8893** | **0.8864** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7295** | 0.5985 | 0.8605 | 0.8692 | 0.8782 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_cross_attention.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_cross_attention/ckpt/ablation_no_cross_attention_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_cross_attention --tag no_cross_attention --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_cross_attention/scores.npz`