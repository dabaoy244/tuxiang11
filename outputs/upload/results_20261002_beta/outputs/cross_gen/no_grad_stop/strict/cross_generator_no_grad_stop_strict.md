# 跨生成器泛化评测（ablation_no_grad_stop_best.pt）

- 配置：`configs/_auto_no_grad_stop.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_grad_stop/ckpt/ablation_no_grad_stop_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 255 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6167 | 0.4667 | 0.7667 | 0.7698 | 0.8264 |
| deepfake | 150/150 | 0.5800 | 0.2667 | 0.8933 | 0.7171 | 0.7002 |
| gaugan | 150/150 | 0.6367 | 0.2733 | 1.0000 | 0.9766 | 0.9772 |
| imle | 150/150 | 0.6367 | 0.4867 | 0.7867 | 0.7837 | 0.8266 |
| san | 150/150 | 0.5100 | 0.7667 | 0.2533 | 0.5390 | 0.5698 |
| seeingdark | 150/150 | 0.6867 | 0.4200 | 0.9533 | 0.9028 | 0.9114 |
| stylegan | 150/150 | 0.9700 | 1.0000 | 0.9400 | 0.9986 | 0.9986 |
| stylegan2 | 150/150 | 0.8667 | 0.9933 | 0.7400 | 0.9867 | 0.9859 |
| whichfaceisreal | 150/150 | 0.6533 | 0.5867 | 0.7200 | 0.7505 | 0.7582 |
| **宏平均** | — | **0.6841** | **0.5844** | **0.7837** | **0.8250** | **0.8394** |
| **微平均（合并全部样本）** | 1350/1350 | **0.6841** | 0.5844 | 0.7837 | 0.7979 | 0.7812 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_grad_stop.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_grad_stop/ckpt/ablation_no_grad_stop_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_grad_stop/strict --tag no_grad_stop_strict`