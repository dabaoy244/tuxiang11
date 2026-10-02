# 跨生成器泛化评测（ablation_no_edge_best.pt）

- 配置：`configs/_auto_no_edge.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_edge/ckpt/ablation_no_edge_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- 推理：3900 张，用时 281 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 150/150 | 0.7167 | 0.4333 | 1.0000 | 0.9817 | 0.9806 |
| crn | 150/150 | 0.6067 | 0.2867 | 0.9267 | 0.8094 | 0.8472 |
| cyclegan | 150/150 | 0.5133 | 0.0267 | 1.0000 | 0.8793 | 0.8320 |
| deepfake | 150/150 | 0.6567 | 0.4667 | 0.8467 | 0.7621 | 0.7312 |
| gaugan | 150/150 | 0.5733 | 0.1467 | 1.0000 | 0.9744 | 0.9698 |
| imle | 150/150 | 0.5600 | 0.2067 | 0.9133 | 0.7260 | 0.7541 |
| progan | 150/150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| san | 150/150 | 0.5233 | 0.6600 | 0.3867 | 0.5700 | 0.5790 |
| seeingdark | 150/150 | 0.5700 | 0.1400 | 1.0000 | 0.8892 | 0.8929 |
| stargan | 150/150 | 0.8133 | 0.6267 | 1.0000 | 0.9922 | 0.9891 |
| stylegan | 150/150 | 0.9800 | 1.0000 | 0.9600 | 0.9962 | 0.9969 |
| stylegan2 | 150/150 | 0.8833 | 1.0000 | 0.7667 | 0.9740 | 0.9791 |
| whichfaceisreal | 150/150 | 0.6167 | 0.4333 | 0.8000 | 0.7483 | 0.7613 |
| **宏平均** | — | **0.6933** | **0.4944** | **0.8923** | **0.8695** | **0.8702** |
| **微平均（合并全部样本）** | 1950/1950 | **0.6933** | 0.4944 | 0.8923 | 0.8552 | 0.8550 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_edge.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_edge/ckpt/ablation_no_edge_best.pt --per-class 150 --size 224 --device cuda --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_edge --tag no_edge --scores-out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_edge/scores.npz`