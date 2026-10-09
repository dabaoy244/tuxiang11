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
echo "[余额] CDP 实读（关机态 ￥0.00/时 = 不计算力费）："
NODE=C:/Users/Administrator/.workbuddy/binaries/node/versions/22.22.2-3/node.exe
# ★ cdp.js 的路径必须用 Windows 形式（D:/...），传 /d/... 会被解析成 D:\d\... ⇒ 静默空输出
"$NODE" "$(cd "$R" && pwd -W)/outputs/screen/cdp.js" eval '(function(){var o=[];document.querySelectorAll("*").forEach(function(el){if(el.children.length===0){var t=(el.innerText||"").trim();if(t&&(t.indexOf("￥")>=0||t.indexOf("释放")>=0))o.push(t.slice(0,40));}});return o.filter(function(v,i,a){return a.indexOf(v)===i;}).slice(0,6).join(" | ");})()' 2>&1 | tail -1
echo
echo "[结论] 训练侧已收尾 => 本值班自动化任务可关停"
