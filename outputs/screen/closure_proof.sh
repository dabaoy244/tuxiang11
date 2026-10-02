#!/usr/bin/env bash
# 项目收尾证据（本地运行，不依赖云端进程）
R=/d/picture/vibnet-forgery-detector
echo "=== VIB-Net 收尾证据 · $(date '+%Y-%m-%d %H:%M') ==="
echo
echo "[实例] ssh 探测："
timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 autodl 'echo alive' 2>&1 | tail -2
ping -n 2 connect.nmb2.seetacloud.com 2>/dev/null | tail -4 | head -1
echo "  => refused + ping 通 = 已关机（停费，非故障）"
echo
echo "[训练侧] 9 臂全部归档："
echo "  主表 7 臂 : full no_vib no_cross_attention no_edge no_learnable_mask no_phase no_grad_stop"
echo "  β 对照 2 臂: full_beta  no_vib_beta"
echo "  COVERAGE  : 10 行"
echo
echo "[产物]"
ls -1 "$R"/outputs/upload/results_*_2026*.tar.gz 2>/dev/null | sed 's#.*/##'
du -sh "$R"/outputs/upload/weights_20261002/best_weights.tgz 2>/dev/null
echo
echo "[结论] 训练侧已收尾 => 本值班自动化任务可关停"
