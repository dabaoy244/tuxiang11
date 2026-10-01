# 跨生成器泛化评测（ablation_no_phase_best.pt）

- 配置：`configs/_auto_no_phase.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_phase/ckpt/ablation_no_phase_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 255 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6067 | 0.2533 | 0.9600 | 0.8708 | 0.8857 |
| deepfake | 150/150 | 0.6300 | 0.4467 | 0.8133 | 0.7777 | 0.8347 |
| gaugan | 150/150 | 0.6167 | 0.2333 | 1.0000 | 0.9907 | 0.9913 |
| imle | 150/150 | 0.6633 | 0.3267 | 1.0000 | 0.9397 | 0.9380 |
| san | 150/150 | 0.5000 | 0.7200 | 0.2800 | 0.5606 | 0.5475 |
| seeingdark | 150/150 | 0.6000 | 0.2067 | 0.9933 | 0.9288 | 0.9291 |
| stylegan | 150/150 | 0.9800 | 1.0000 | 0.9600 | 0.9984 | 0.9985 |
| stylegan2 | 150/150 | 0.8967 | 0.9867 | 0.8067 | 0.9641 | 0.9705 |
| whichfaceisreal | 150/150 | 0.7100 | 0.4600 | 0.9600 | 0.8925 | 0.9033 |
| **宏平均** | — | **0.6893** | **0.5148** | **0.8637** | **0.8804** | **0.8887** |
| **微平均（合并全部样本）** | 1350/1350 | **0.6893** | 0.5148 | 0.8637 | 0.8498 | 0.8736 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_phase.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_phase/ckpt/ablation_no_phase_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_phase/strict --tag no_phase_strict`