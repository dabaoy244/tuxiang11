# 跨生成器泛化评测（ablation_no_vib_best.pt）

- 配置：`configs/_auto_no_vib.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_vib/ckpt/ablation_no_vib_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 257 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6600 | 0.6600 | 0.6600 | 0.7356 | 0.7547 |
| deepfake | 150/150 | 0.6500 | 0.5667 | 0.7333 | 0.7260 | 0.6944 |
| gaugan | 150/150 | 0.6900 | 0.3867 | 0.9933 | 0.9610 | 0.9500 |
| imle | 150/150 | 0.7700 | 0.7200 | 0.8200 | 0.8578 | 0.8555 |
| san | 150/150 | 0.5200 | 0.7533 | 0.2867 | 0.5472 | 0.5641 |
| seeingdark | 150/150 | 0.7233 | 0.4733 | 0.9733 | 0.9161 | 0.8982 |
| stylegan | 150/150 | 0.9933 | 1.0000 | 0.9867 | 1.0000 | 1.0000 |
| stylegan2 | 150/150 | 0.9167 | 0.9933 | 0.8400 | 0.9582 | 0.9650 |
| whichfaceisreal | 150/150 | 0.7533 | 0.7067 | 0.8000 | 0.8289 | 0.8218 |
| **宏平均** | — | **0.7419** | **0.6956** | **0.7881** | **0.8368** | **0.8337** |
| **微平均（合并全部样本）** | 1350/1350 | **0.7419** | 0.6956 | 0.7881 | 0.8284 | 0.8265 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_vib.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_vib/ckpt/ablation_no_vib_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_vib/strict --tag no_vib_strict`