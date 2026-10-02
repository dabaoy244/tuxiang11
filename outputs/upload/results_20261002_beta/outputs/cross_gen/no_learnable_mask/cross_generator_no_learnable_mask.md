# 跨生成器泛化评测（ablation_no_learnable_mask_best.pt）

- 配置：`configs/_auto_no_learnable_mask.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_learnable_mask/ckpt/ablation_no_learnable_mask_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 277 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7200 | 0.4400 | 1.0000 | 0.9828 | 0.9822 |
| crn | 150/150 | 0.6833 | 0.4067 | 0.9600 | 0.9072 | 0.9239 |
| cyclegan | 150/150 | 0.5300 | 0.0600 | 1.0000 | 0.9066 | 0.8966 |
| deepfake | 150/150 | 0.6533 | 0.5667 | 0.7400 | 0.7590 | 0.7842 |
| gaugan | 150/150 | 0.6033 | 0.2067 | 1.0000 | 0.9659 | 0.9621 |
| imle | 150/150 | 0.6067 | 0.3067 | 0.9067 | 0.8541 | 0.8941 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5267 | 0.7600 | 0.2933 | 0.5764 | 0.5707 |
| seeingdark | 150/150 | 0.6267 | 0.3400 | 0.9133 | 0.8622 | 0.8868 |
| stargan | 150/150 | 0.9400 | 0.8800 | 1.0000 | 0.9993 | 0.9994 |
| stylegan | 150/150 | 0.9600 | 1.0000 | 0.9200 | 0.9966 | 0.9971 |
| stylegan2 | 150/150 | 0.8067 | 0.9933 | 0.6200 | 0.9824 | 0.9825 |
| whichfaceisreal | 150/150 | 0.6700 | 0.5733 | 0.7667 | 0.8278 | 0.8462 |
| **宏平均** | — | **0.7174** | **0.5795** | **0.8554** | **0.8939** | **0.9020** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7174** | 0.5795 | 0.8554 | 0.8670 | 0.8819 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_learnable_mask.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_learnable_mask/ckpt/ablation_no_learnable_mask_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_learnable_mask --tag no_learnable_mask --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_learnable_mask/scores.npz`