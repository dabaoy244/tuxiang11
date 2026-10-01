# 跨生成器泛化评测（ablation_no_phase_best.pt）

- 配置：`configs/_auto_no_phase.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_phase/ckpt/ablation_no_phase_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 280 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7267 | 0.4600 | 0.9933 | 0.9731 | 0.9742 |
| crn | 150/150 | 0.5933 | 0.2267 | 0.9600 | 0.8784 | 0.8891 |
| cyclegan | 150/150 | 0.5233 | 0.0467 | 1.0000 | 0.9693 | 0.9742 |
| deepfake | 150/150 | 0.6133 | 0.4000 | 0.8267 | 0.7532 | 0.8025 |
| gaugan | 150/150 | 0.6367 | 0.2733 | 1.0000 | 0.9785 | 0.9767 |
| imle | 150/150 | 0.6267 | 0.2533 | 1.0000 | 0.9112 | 0.9148 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5000 | 0.7333 | 0.2667 | 0.5325 | 0.5316 |
| seeingdark | 150/150 | 0.6133 | 0.2333 | 0.9933 | 0.9314 | 0.9297 |
| stargan | 150/150 | 0.8767 | 0.7533 | 1.0000 | 1.0000 | 1.0000 |
| stylegan | 150/150 | 0.9900 | 1.0000 | 0.9800 | 1.0000 | 1.0000 |
| stylegan2 | 150/150 | 0.8733 | 0.9933 | 0.7533 | 0.9500 | 0.9621 |
| whichfaceisreal | 150/150 | 0.6467 | 0.3400 | 0.9533 | 0.8798 | 0.8829 |
| **宏平均** | — | **0.7092** | **0.5164** | **0.9021** | **0.9044** | **0.9106** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7092** | 0.5164 | 0.9021 | 0.8883 | 0.9096 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_phase.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_phase/ckpt/ablation_no_phase_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_phase --tag no_phase --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_phase/scores.npz`