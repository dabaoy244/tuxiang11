# 跨生成器泛化评测（ablation_full_best.pt）

- 配置：`configs/default_crossval.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/full/ckpt/ablation_full_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 282 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7567 | 0.5200 | 0.9933 | 0.9724 | 0.9759 |
| crn | 150/150 | 0.6767 | 0.4333 | 0.9200 | 0.8742 | 0.9050 |
| cyclegan | 150/150 | 0.5367 | 0.0733 | 1.0000 | 0.9367 | 0.9152 |
| deepfake | 150/150 | 0.6200 | 0.3200 | 0.9200 | 0.8239 | 0.8260 |
| gaugan | 150/150 | 0.6200 | 0.2400 | 1.0000 | 0.9628 | 0.9615 |
| imle | 150/150 | 0.6067 | 0.3133 | 0.9000 | 0.8293 | 0.8677 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5567 | 0.7467 | 0.3667 | 0.5846 | 0.5873 |
| seeingdark | 150/150 | 0.6700 | 0.4467 | 0.8933 | 0.8366 | 0.8566 |
| stargan | 150/150 | 0.9100 | 0.8200 | 1.0000 | 0.9991 | 0.9992 |
| stylegan | 150/150 | 0.9900 | 1.0000 | 0.9800 | 0.9991 | 0.9992 |
| stylegan2 | 150/150 | 0.8933 | 1.0000 | 0.7867 | 0.9803 | 0.9840 |
| whichfaceisreal | 150/150 | 0.6700 | 0.7267 | 0.6133 | 0.7592 | 0.7793 |
| **宏平均** | — | **0.7313** | **0.5877** | **0.8749** | **0.8891** | **0.8967** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7313** | 0.5877 | 0.8749 | 0.8721 | 0.8835 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/default_crossval.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/full/ckpt/ablation_full_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/full --tag full --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/full/scores.npz`