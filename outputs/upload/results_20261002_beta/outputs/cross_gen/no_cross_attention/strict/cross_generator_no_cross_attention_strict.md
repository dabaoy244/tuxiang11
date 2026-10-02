# 跨生成器泛化评测（ablation_no_cross_attention_best.pt）

- 配置：`configs/_auto_no_cross_attention.yaml`　权重：`/root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_cross_attention/ckpt/ablation_no_cross_attention_best.pt`
- 采样：每生成器每类 ≤150 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`全部`　exclude=`progan,biggan,cyclegan,stargan`（本表只覆盖筛选后的生成器，勿与全量 13 生成器表混比）
- 推理：2700 张，用时 255 秒，batch=8

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| crn | 150/150 | 0.6167 | 0.3133 | 0.9200 | 0.8150 | 0.8179 |
| deepfake | 150/150 | 0.6333 | 0.5733 | 0.6933 | 0.7224 | 0.7154 |
| gaugan | 150/150 | 0.7267 | 0.4533 | 1.0000 | 0.9768 | 0.9654 |
| imle | 150/150 | 0.6667 | 0.3400 | 0.9933 | 0.9226 | 0.9145 |
| san | 150/150 | 0.4833 | 0.6733 | 0.2933 | 0.5816 | 0.5709 |
| seeingdark | 150/150 | 0.6733 | 0.4867 | 0.8600 | 0.8230 | 0.8161 |
| stylegan | 150/150 | 0.9467 | 1.0000 | 0.8933 | 0.9937 | 0.9948 |
| stylegan2 | 150/150 | 0.8533 | 0.9800 | 0.7267 | 0.9812 | 0.9808 |
| whichfaceisreal | 150/150 | 0.7500 | 0.7133 | 0.7867 | 0.8355 | 0.8501 |
| **宏平均** | — | **0.7056** | **0.6148** | **0.7963** | **0.8502** | **0.8473** |
| **微平均（合并全部样本）** | 1350/1350 | **0.7056** | 0.6148 | 0.7963 | 0.8205 | 0.8251 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

> 复现命令：`python scripts/eval_cross_generator.py --config configs/_auto_no_cross_attention.yaml --ckpt /root/autodl-tmp/vibnet-forgery-detector/outputs/ablation/no_cross_attention/ckpt/ablation_no_cross_attention_best.pt --per-class 150 --size 224 --device cuda --exclude-generators progan,biggan,cyclegan,stargan --out /root/autodl-tmp/vibnet-forgery-detector/outputs/cross_gen/no_cross_attention/strict --tag no_cross_attention_strict`