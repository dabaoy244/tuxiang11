| 消融项 | 说明 | ACC | F1 | AUC | mIoU | Dice | ΔACC | ΔmIoU | 归档 |
|---|---|---|---|---|---|---|---|---|---|
| full | 完整模型 | 0.7929 | 0.8113 | 0.9207 | 0.4290 | 0.6004 | +0.0000 | +0.0000 | full_20260930_053039 |
| no_vib | model.vib.use_vib -> False | 0.8234 | 0.8283 | 0.9064 | 0.4160 | 0.5876 | -0.0305 | +0.0129 | no_vib_20260930_142204 |
| no_cross_attention | model.fusion.use_cross_attention -> False | 0.7853 | 0.8025 | 0.9011 | 0.4291 | 0.6005 | +0.0076 | -0.0002 | no_cross_attention_20260930_195318 |
| no_edge | model.localization.use_edge_head -> False | 0.7672 | 0.7936 | 0.8999 | 0.4219 | 0.5934 | +0.0257 | +0.0071 | no_edge_20261001_011455 |
| no_learnable_mask | model.freq.use_learnable_mask -> False | 0.7709 | 0.7880 | 0.9031 | 0.4362 | 0.6074 | +0.0220 | -0.0072 | no_learnable_mask_20261001_060534 |
| no_phase | model.freq.use_phase -> False | 0.7831 | 0.8092 | 0.9241 | 0.4394 | 0.6105 | +0.0098 | -0.0104 | no_phase_20261001_105902 |
| no_grad_stop | model.localization.use_grad_stop -> False | 0.7778 | 0.7909 | 0.8936 | 0.4544 | 0.6249 | +0.0151 | -0.0255 | no_grad_stop_20261001_160507 |
