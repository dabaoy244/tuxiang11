# 跨生成器泛化评测（ablation_no_learnable_mask_best.pt）

- 配置：`configs/_auto_no_learnable_mask.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_learnable_mask/ckpt/ablation_no_learnable_mask_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 257 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6900 | 0.4200 | 0.9600 | 0.8959 | 0.9131 |
| deepfake | 150/150 | 0.6567 | 0.6067 | 0.7067 | 0.7548 | 0.7731 |
| gaugan | 150/150 | 0.6367 | 0.2733 | 1.0000 | 0.9745 | 0.9712 |
| imle | 150/150 | 0.6333 | 0.3800 | 0.8867 | 0.8678 | 0.9041 |
| san | 150/150 | 0.5200 | 0.7467 | 0.2933 | 0.5768 | 0.5779 |
| seeingdark | 150/150 | 0.6133 | 0.3267 | 0.9000 | 0.8420 | 0.8744 |
| stylegan | 150/150 | 0.9333 | 1.0000 | 0.8667 | 0.9968 | 0.9971 |
| stylegan2 | 150/150 | 0.8467 | 0.9867 | 0.7067 | 0.9858 | 0.9860 |
| whichfaceisreal | 150/150 | 0.6633 | 0.5667 | 0.7600 | 0.8120 | 0.8342 |
| **宏平均** | — | **0.6881** | **0.5896** | **0.7867** | **0.8563** | **0.8701** |
| **微平均（合并全部样本）** | 1350/1350 | **0.6881** | 0.5896 | 0.7867 | 0.8211 | 0.8333 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_learnable_mask.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_learnable_mask/ckpt/ablation_no_learnable_mask_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_learnable_mask/strict --tag no_learnable_mask_strict`