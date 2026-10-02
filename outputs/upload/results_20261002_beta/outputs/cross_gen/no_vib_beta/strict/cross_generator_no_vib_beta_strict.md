# 跨生成器泛化评测（ablation_no_vib_best.pt）

- 配置：`configs/_auto_beta_no_vib.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/no_vib/ckpt/ablation_no_vib_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 257 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6700 | 0.3600 | 0.9800 | 0.9214 | 0.9278 |
| deepfake | 150/150 | 0.6067 | 0.5933 | 0.6200 | 0.6944 | 0.7118 |
| gaugan | 150/150 | 0.6600 | 0.3200 | 1.0000 | 0.9666 | 0.9622 |
| imle | 150/150 | 0.7067 | 0.4267 | 0.9867 | 0.9005 | 0.8854 |
| san | 150/150 | 0.4967 | 0.7000 | 0.2933 | 0.5005 | 0.5299 |
| seeingdark | 150/150 | 0.6200 | 0.2467 | 0.9933 | 0.9304 | 0.9198 |
| stylegan | 150/150 | 0.9867 | 1.0000 | 0.9733 | 0.9938 | 0.9967 |
| stylegan2 | 150/150 | 0.8933 | 0.9933 | 0.7933 | 0.9114 | 0.9373 |
| whichfaceisreal | 150/150 | 0.7667 | 0.7333 | 0.8000 | 0.8486 | 0.8402 |
| **宏平均** | — | **0.7119** | **0.5970** | **0.8267** | **0.8519** | **0.8568** |
| **微平均（合并全部样本）** | 1350/1350 | **0.7119** | 0.5970 | 0.8267 | 0.8357 | 0.8551 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_beta_no_vib.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation_betafull/no_vib/ckpt/ablation_no_vib_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_vib_beta/strict --tag no_vib_beta_strict`