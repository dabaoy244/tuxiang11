# 跨生成器泛化评测（ablation_no_grad_stop_best.pt）

- 配置：`configs/_auto_no_grad_stop.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_grad_stop/ckpt/ablation_no_grad_stop_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 277 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7400 | 0.4800 | 1.0000 | 0.9716 | 0.9763 |
| crn | 150/150 | 0.6500 | 0.5400 | 0.7600 | 0.7754 | 0.8177 |
| cyclegan | 150/150 | 0.5300 | 0.0600 | 1.0000 | 0.8960 | 0.8588 |
| deepfake | 150/150 | 0.5867 | 0.2267 | 0.9467 | 0.7585 | 0.7580 |
| gaugan | 150/150 | 0.6167 | 0.2333 | 1.0000 | 0.9739 | 0.9647 |
| imle | 150/150 | 0.6033 | 0.4200 | 0.7867 | 0.7480 | 0.8092 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5100 | 0.7667 | 0.2533 | 0.5407 | 0.5532 |
| seeingdark | 150/150 | 0.7100 | 0.4600 | 0.9600 | 0.9097 | 0.9128 |
| stargan | 150/150 | 0.8200 | 0.6400 | 1.0000 | 0.9966 | 0.9970 |
| stylegan | 150/150 | 0.9800 | 1.0000 | 0.9600 | 0.9988 | 0.9989 |
| stylegan2 | 150/150 | 0.8800 | 0.9933 | 0.7667 | 0.9898 | 0.9905 |
| whichfaceisreal | 150/150 | 0.6167 | 0.5267 | 0.7067 | 0.7225 | 0.7615 |
| **宏平均** | — | **0.7110** | **0.5651** | **0.8569** | **0.8678** | **0.8768** |
| **微平均（合并全部样本）** | 1950/1950 | **0.7110** | 0.5651 | 0.8569 | 0.8476 | 0.8498 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_grad_stop.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_grad_stop/ckpt/ablation_no_grad_stop_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_grad_stop --tag no_grad_stop --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_grad_stop/scores.npz`