# 跨生成器泛化评测（vibnet_best.pt）

- 配置：`configs/default.yaml`（仅决定模型结构；**本表覆盖哪些生成器由下一行决定**）　权重：`outputs/upload/results_20260928/_pack/vibnet_best.pt`
- 采样：每生成器每类 ≤400 张；真图去重：是；随机种子 3407
- **生成器筛选**：include=`biggan,cyclegan,stargan`　exclude=`无`
  - ⚠ 本表**只覆盖 3 个留出生成器**（val_cross），**不可**与全量 13 生成器的跨生成器表（`outputs/cross_gen_20260928/`）混比，两者分母不同。
  - 该次运行早于脚本开始记录筛选字段，此处的筛选口径由产物反推（3 行表 + `scores_valcross.npz` 的 `per_gen_meta`），原始命令行未留存。
- 推理：2400 张，用时 843 秒，batch=16

| 生成器 | 真/假样本数 | ACC | 真图ACC | 假图ACC | AUC | AP |
|---|---|---|---|---|---|---|
| biggan | 400/400 | 0.6800 | 0.3625 | 0.9975 | 0.9268 | 0.9127 |
| cyclegan | 400/400 | 0.5150 | 0.0300 | 1.0000 | 0.8739 | 0.8300 |
| stargan | 400/400 | 0.7612 | 0.5225 | 1.0000 | 0.9974 | 0.9973 |
| **宏平均** | — | **0.6521** | **0.3050** | **0.9992** | **0.9327** | **0.9133** |
| **微平均（合并全部样本）** | 1200/1200 | **0.6521** | 0.3050 | 0.9992 | 0.9137 | 0.8765 |

> 注：宏平均 = 先算各生成器的指标再取平均（每个生成器权重相同）；微平均 = 把所有样本合在一起算。
> 两者差异大说明模型在不同生成器上表现严重不均。

## 附：等效复现命令

该次运行的原始命令行未留存；按产物记录（`configs/default.yaml`、`per_class=400`、`seed=3407`、`batch=16`、`size=224`、`include=biggan,cyclegan,stargan`）反推的等效命令为：

```bash
python scripts/eval_cross_generator.py \
  --config configs/default.yaml \
  --ckpt outputs/upload/results_20260928/_pack/vibnet_best.pt \
  --generators biggan,cyclegan,stargan \
  --per-class 400 --size 224 --batch-size 16 --seed 3407 \
  --out outputs/val_cross_20260929 --tag 20260929 \
  --scores-out outputs/val_cross_20260929/scores_valcross.npz
```

> 注：`--config` 建议改用 `configs/default_crossval.yaml`（把留出生成器写成配置的一部分，
> 口径不必再靠命令行传参维持）。本次未改，是为了保持与已产出结果完全一致。