# 跨生成器泛化评测（ablation_no_edge_best.pt）

- 配置：`configs/_auto_no_edge.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_edge/ckpt/ablation_no_edge_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 257 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.5700 | 0.2333 | 0.9067 | 0.7418 | 0.7861 |
| deepfake | 150/150 | 0.6567 | 0.5133 | 0.8000 | 0.7416 | 0.7232 |
| gaugan | 150/150 | 0.6100 | 0.2200 | 1.0000 | 0.9760 | 0.9751 |
| imle | 150/150 | 0.5800 | 0.2200 | 0.9400 | 0.8082 | 0.8226 |
| san | 150/150 | 0.5300 | 0.6600 | 0.4000 | 0.5525 | 0.5797 |
| seeingdark | 150/150 | 0.5767 | 0.1533 | 1.0000 | 0.8876 | 0.8925 |
| stylegan | 150/150 | 0.9533 | 0.9933 | 0.9133 | 0.9915 | 0.9931 |
| stylegan2 | 150/150 | 0.8833 | 0.9867 | 0.7800 | 0.9733 | 0.9770 |
| whichfaceisreal | 150/150 | 0.6667 | 0.5267 | 0.8067 | 0.7547 | 0.7665 |
| **宏平均** | — | **0.6696** | **0.5007** | **0.8385** | **0.8252** | **0.8351** |
| **微平均（合并全部样本）** | 1350/1350 | **0.6696** | 0.5007 | 0.8385 | 0.8041 | 0.7992 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_edge.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_edge/ckpt/ablation_no_edge_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_edge/strict --tag no_edge_strict`