#!/bin/bash
# =============================================================================
# 云端脚本同步器 —— 防止「本地旧版覆盖云端」造成静默回退（白训练）
#
# 背景（2026-10-05 实证）：
#   云端 `/root/autodl-tmp` 是**非 git 仓库**，脚本改在云端就地生效再 scp 回本地，
#   而本地 `scripts/cloud/` 长期没跟上 ⇒ 本地停在 v3.2、云端已是 v3.3
#   （`PAT_CHAIN` 差一个 `lite_chain`）。若在本地旧版上改完直接 scp 上去，
#   **静默覆盖掉云端的修复** ⇒ 守夜认不出 lite_chain ⇒ 链在等待期被判「全部结束」
#   ⇒ 15 min 后关机 ⇒ **正在等待或刚拉起的训练被砍（白训练）**。
#
# 本脚本把「先取回 → 再 diff → 才上传」固化成一条命令：push 默认**拒绝**覆盖有差异的同名文件。
#
# 用法：
#   bash scripts/cloud/sync_cloud_scripts.sh check        只读比对，列出差异（不发任何写操作）
#   bash scripts/cloud/sync_cloud_scripts.sh pull         云端 → 本地（覆盖前自动备份到 outputs/）
#   bash scripts/cloud/sync_cloud_scripts.sh push         本地 → 云端；有同名差异时**必须加 --force**
#   bash scripts/cloud/sync_cloud_scripts.sh push --force 明知有差异仍覆盖（会打印被覆盖项的 diff 摘要）
#
# 退出码：check/push 发现「同名但内容不同」→ 2（便于 CI/钩子拦截）
# =============================================================================
set -u

# --- PATH 自愈：Git Bash 下 PATH 可能整个空掉（dirname/ssh 全 127）---
export PATH="/c/Users/Administrator/.workbuddy/binaries/PortableGit/versions/1.2.0/mingw64/bin:/c/Users/Administrator/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:/c/Windows/System32:/c/Windows/System32/OpenSSH:/usr/bin:/bin:$PATH"

HOST=${SYNC_HOST:-autodl}
REMOTE=${SYNC_REMOTE:-/root/autodl-tmp}
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$LOCAL_DIR/../.." && pwd)"
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$ROOT_DIR/outputs/cloud_logs_$(date +%Y%m%d)/sync_backup_$TS"
TMP=$(mktemp -d 2>/dev/null || echo "/tmp/sync_$$")
mkdir -p "$TMP"
# ⚠ Git Bash 的 mktemp 返回 Windows 风格路径（C:\Users\...），反斜杠传给 awk -v / scp 会被吃掉。
#   统一转成 MSYS 风格（/c/Users/...）——这是本脚本踩过的真坑，别删。
TMP=$(cd "$TMP" && pwd)

MODE=${1:-check}
FORCE=${2:-}

remote_md5(){
  timeout 60 ssh -o ConnectTimeout=10 "$HOST" "cd $REMOTE && md5sum *.sh 2>/dev/null" || return 1
}
local_md5(){
  ( cd "$LOCAL_DIR" && md5sum *.sh 2>/dev/null )
}

remote_md5 > "$TMP/remote.txt" || { echo "[x] 无法读取云端清单（实例是否已关机？）"; exit 3; }
local_md5  > "$TMP/local.txt"

report(){
  # 输出四类差异到 stdout，并按类型写入 $TMP/diff.txt（用 shell 重定向，不用 awk -v 传路径）
  awk '
    FNR==NR { sub(/^\*/,"",$2); r[$2]=$1; next }
            { sub(/^\*/,"",$2); l[$2]=$1 }
    END {
      for (f in l) {
        if (f in r) { if (l[f]!=r[f]) printf "DIFF\t%s\n", f }
        else printf "ONLY_LOCAL\t%s\n", f
      }
      for (f in r) if (!(f in l)) printf "ONLY_REMOTE\t%s\n", f
    }
  ' "$TMP/remote.txt" "$TMP/local.txt" > "$TMP/diff.txt"

  echo "== 同名但内容不同（push 会覆盖云端）=="
  if [ "$(diff_count)" -gt 0 ]; then
    awk -F'\t' '$1=="DIFF"{print "   [!] "$2}' "$TMP/diff.txt"
  else
    echo "   （无）"
  fi
  echo "== 仅云端有（本地缺失，pull 会补齐）=="
  if grep -q '^ONLY_REMOTE' "$TMP/diff.txt" 2>/dev/null; then
    awk -F'\t' '$1=="ONLY_REMOTE"{print "   - "$2}' "$TMP/diff.txt"
  else
    echo "   （无）"
  fi
  echo "== 仅本地有（云端没有，push 会新增）=="
  if grep -q '^ONLY_LOCAL' "$TMP/diff.txt" 2>/dev/null; then
    awk -F'\t' '$1=="ONLY_LOCAL"{print "   + "$2}' "$TMP/diff.txt"
  else
    echo "   （无）"
  fi
  echo "== 汇总：同名差异 $(diff_count) 个 =="
}

# ⚠ 不要用 `grep -c PATTERN file || echo 0`：无匹配时 grep 输出 "0" 且返回 1，
#   `||` 会再补一个 0 ⇒ 得到 "0\n0"，`[ -gt ]` 直接报 integer expected。用 awk 计数。
diff_count(){ awk '$1=="DIFF"{c++} END{print c+0}' "$TMP/diff.txt" 2>/dev/null; }

case "$MODE" in
  check)
    echo "== 比对 $HOST:$REMOTE/*.sh 与 $LOCAL_DIR/*.sh =="
    report
    [ "$(diff_count)" -gt 0 ] && exit 2
    exit 0
    ;;
  pull)
    echo "== pull：云端 → 本地（$BACKUP_DIR 为被覆盖文件的备份）=="
    mkdir -p "$TMP/dl" "$BACKUP_DIR"
    if timeout 120 scp -o ConnectTimeout=10 "$HOST:$REMOTE/*.sh" "$TMP/dl/" >/dev/null 2>&1; then :; else
      # 通配符在部分 scp 实现下不展开，退回 tar
      timeout 60 ssh -o ConnectTimeout=10 "$HOST" "cd $REMOTE && tar czf /tmp/_sync.tgz *.sh" \
        && timeout 120 scp -o ConnectTimeout=10 "$HOST:/tmp/_sync.tgz" "$TMP/dl/" \
        && tar xzf "$TMP/dl/_sync.tgz" -C "$TMP/dl" \
        && timeout 30 ssh -o ConnectTimeout=10 "$HOST" 'rm -f /tmp/_sync.tgz'
    fi
    n=0
    for f in "$TMP/dl"/*.sh; do
      [ -f "$f" ] || continue
      b=$(basename "$f")
      if [ -f "$LOCAL_DIR/$b" ]; then
        if cmp -s "$f" "$LOCAL_DIR/$b"; then continue; fi
        cp "$LOCAL_DIR/$b" "$BACKUP_DIR/$b"
        echo "   ~ 覆盖 $b（旧版已备份）"
      else
        echo "   + 新增 $b"
      fi
      cp "$f" "$LOCAL_DIR/$b"; chmod +x "$LOCAL_DIR/$b" 2>/dev/null
      n=$((n+1))
    done
    echo "== pull 完成：$n 个文件有变化，备份在 $BACKUP_DIR =="
    ;;
  push)
    echo "== push：本地 → 云端 =="
    report
    if [ "$(diff_count)" -gt 0 ] && [ "$FORCE" != "--force" ]; then
      echo ""
      echo "[x] 拒绝上传：存在同名内容差异（上图 [!] 项）。"
      echo "    → 若云端那份才是最新的，先执行 pull；确认要用本地版覆盖再加 --force。"
      exit 2
    fi
    for f in "$LOCAL_DIR"/*.sh; do
      [ -f "$f" ] || continue
      timeout 60 scp -o ConnectTimeout=10 "$f" "$HOST:$REMOTE/" >/dev/null 2>&1 \
        && echo "   ^ 上传 $(basename "$f")" || echo "   [x] 失败 $(basename "$f")"
    done
    echo "== push 完成；云端复核 md5：=="
    remote_md5 | sed 's/^/   /'
    ;;
  *)
    echo "用法：$0 {check|pull|push [--force]}"; exit 1;;
esac
