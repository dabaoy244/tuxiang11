#!/usr/bin/env bash
# =============================================================================
# cloud_locate_repo.sh —— 云端「仓库到底在哪」自证脚本
#
# 为什么需要它：容器里数据盘是持久卷（AutoDL 上是 /root/autodl-tmp，**没有 /autodl-tmp**），
# 但解包路径、实例是否换过、数据盘是否挂上，都不是我们能猜的。猜路径的代价是
# 「cd 失败 → 后面的命令就地继续执行 → 静默加载旧代码 / 空跑」。所以：不假设，直接问机器。
# 实测：连续猜错两条路径（各花一轮沟通）后，第一次跑本脚本就定位到正确目录。
#
# 用法：整块粘贴到终端即可（也可以 bash scripts/cloud_locate_repo.sh）
# 成本：0 卡时，约 5~15 秒
# =============================================================================
set +e

echo "== 0) 我在哪 / 哪个容器 =="
hostname; pwd; date

echo
echo "== 1) 数据盘挂没挂上 =="
ls -ld /autodl-tmp /root/autodl-tmp 2>&1
df -h /autodl-tmp 2>&1 | tail -3

echo
echo "== 2) 数据盘根目录长什么样 =="
ls -la /autodl-tmp 2>&1 | head -40

echo
echo "== 3) 历史上真实用过的路径（最有用的一条） =="
echo "--- 3.1 带 vibnet 的历史命令 ---"
grep -aE "vibnet" ~/.bash_history 2>/dev/null | tail -30
echo "--- 3.2 带 autodl-tmp 的历史命令 ---"
grep -aE "cd +[^ ]*autodl-tmp" ~/.bash_history 2>/dev/null | tail -20

echo
echo "== 3.5 先试最可能的几个候选（0 成本，秒出） =="
for d in /root/autodl-tmp/vibnet-forgery-detector \
         /autodl-tmp/vibnet-forgery-detector \
         /autodl-tmp/autodl-tmp/vibnet-forgery-detector \
         /root/autodl-tmp/autodl-tmp/vibnet-forgery-detector \
         /root/vibnet-forgery-detector \
         /workspace/vibnet-forgery-detector; do
  if [ -d "$d" ]; then echo "  FOUND  $d"; else echo "  no     $d"; fi
done

echo
echo "== 4) 全盘定位（不看历史，直接找实物） =="
echo "--- 4.1 仓库目录 ---"
find / -maxdepth 6 -type d -name "vibnet-forgery-detector" 2>/dev/null
echo "--- 4.2 仓库里的 train.py ---"
find / -maxdepth 7 -path "*vibnet-forgery-detector*" -name "train.py" 2>/dev/null
echo "--- 4.3 我们打的交付包 ---"
find / -maxdepth 5 -name "vibnet_code.tar.gz" -o -maxdepth 5 -name "official_vibnet_baseline.tar.gz" 2>/dev/null
echo "--- 4.4 探针脚本与其日志（证明上次在哪跑的） ---"
find / -maxdepth 7 \( -name "bench_upsample.py" -o -name "bench_prof.py" -o -name "pA.log" -o -name "pB.log" -o -name "pC.log" \) 2>/dev/null | head -30

echo
echo "== 5) 数据集在不在 =="
find / -maxdepth 4 -type d \( -iname "CASIA*" -o -iname "Datasets" -o -iname "*tamper*" \) 2>/dev/null | head -20

echo
echo "== 6) 一句话进入仓库并自检（把 <上面找到的路径> 换掉） =="
echo '    cd <PATH> && pwd && md5sum scripts/train.py'
echo "    期望 md5 = ea8ad14f94884a0fee27f2f78abe806d（18:13 交付包内的 train.py）"
echo
echo "== 7) 懒人版：不用知道路径 =="
echo '    cd "$(find / -maxdepth 6 -type d -name vibnet-forgery-detector 2>/dev/null | head -1)" && pwd && md5sum scripts/train.py'
