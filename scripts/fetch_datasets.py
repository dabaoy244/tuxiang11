#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ForenSynths（CNNDetection）官方数据集的下载与解压。

数据来源（官方，托管在 HuggingFace，**无需向作者发邮件申请**）：
    https://huggingface.co/datasets/sywang/CNNDetection

官方仓库 scripts 原文（dataset/{train,val,test}/download_*set.sh）：
    val    -> progan_val.zip                       ~0.79 GB
    test   -> CNN_synth_testset.zip                ~18.7 GB   (13 种生成器, 约 9 万张)
              progan_testset.zip                   ~0.80 GB   (仅 ProGAN 部分)
    train  -> progan_train.7z.001 ... .007         ~70.4 GB   (20 类, 约 72 万张)
              7 卷合并解压得到 progan_train.zip，再解压得 train/

总体积约 90 GB；解压后另需约 85~100 GB。**请预留至少 200 GB 磁盘。**

用法
----
    # ① 先跑通真实数据（只下 1.6 GB，推荐第一步）
    python scripts/fetch_datasets.py --split val test

    # ② 全量（90 GB，建议挂一晚上）
    python scripts/fetch_datasets.py --split all

    # ③ 训练集只取 4 个类，省磁盘（需 7z 或 py7zr）
    python scripts/fetch_datasets.py --split train --classes car cat chair horse

    # ④ 换国内镜像（huggingface.co 慢时用）
    python scripts/fetch_datasets.py --split val --mirror hf-mirror

    # ⑤ 只校验本地已有文件，不下载
    python scripts/fetch_datasets.py --split all --check-only
"""

from __future__ import annotations

import argparse
import bisect
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- 源与文件表
HF_REPO = "sywang/CNNDetection"

ENDPOINTS = {
    "hf": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}

SPLIT_FILES = {
    "val": ["progan_val.zip"],
    "progan_test": ["progan_testset.zip"],
    "test": ["CNN_synth_testset.zip"],
    "train": [f"progan_train.7z.{i:03d}" for i in range(1, 8)],
}

#: 各文件解压后的目标子目录（相对 data/Datasets）。
#:
#: ⚠ 官方各压缩包的**内部层级并不一致**，实测：
#:     progan_val.zip      -> <class>/{0_real,1_fake}
#:     progan_testset.zip  -> progan/<class>/{0_real,1_fake}
#:     CNN_synth_testset   -> <generator>/[<class>/]{0_real,1_fake}
#:   所以不能直接解压到同一个目录（会互相混层，且 val 会被 test 覆盖）。
#:   统一做法：先解到 staging，再用 _move_children 归位到 train/val/test。
EXTRACT_TARGET = {
    "progan_val.zip": "ForenSynths/val",
    "progan_testset.zip": "ForenSynths/test",
    "CNN_synth_testset.zip": "ForenSynths/test",
    "progan_train.zip": "ForenSynths/train",
}

#: 训练集默认只解这几类（与 `cloud_autodl.sh` 的 `CLASSES` 保持一致）
DEFAULT_TRAIN_CLASSES = ("car", "cat", "chair", "horse")


def _is_nonfirst_volume(name: str) -> bool:
    """`progan_train.7z.002` ~ `.007` 这类**非首卷**分片 → True。

    ⚠ 为什么必须显式跳过：`ok` 里 7 个卷是**各自独立**的条目。首卷分支处理完会把
    `.001~.007` 全删掉（省 70 GB），而循环接着拿 `.002` 去 `zipfile.ZipFile()` ——
    打开一个刚被删掉的文件 → `FileNotFoundError`。
    2026-09-24 上云实测：**144024 项全部解压成功、4 个类已归位之后**，脚本在收尾处崩掉。
    """
    if ".7z." not in name:
        return False
    tail = name.rsplit(".7z.", 1)[1]
    return tail.isdigit() and tail != "001"


def target_rel_for(fname: str) -> str:
    """`_downloads/` 里的文件名 → 该解压归位到 `Datasets/` 下的哪个子目录。

    ⚠ 训练集在磁盘上叫 `progan_train.7z.001`（分卷首卷），而 `EXTRACT_TARGET` 的键
    写的是**内层归档名** `progan_train.zip`。直接 `EXTRACT_TARGET.get(f)` 查不到、
    会**静默落到默认值 "ForenSynths"**，于是 train 被放成 `ForenSynths/<类>/`，
    而不是 `ForenSynths/train/<类>/`。而 `GenSynthsDataset(split="train")` 只认
    `<base>/train`（见 src/data/datasets.py 的注释：不能拿 `<base>` 当回退，
    否则会把 val/test 混进训练集）——结果 `split="train"` **一条样本都扫不到**。
    2026-09-24 实测踩到，且发生在开 GPU 之前。
    """
    if fname.endswith(".7z.001"):
        return EXTRACT_TARGET["progan_train.zip"]
    return EXTRACT_TARGET.get(fname, "ForenSynths")


def _dir_has_files(d: str) -> bool:
    try:
        return any(not n.startswith(".") for n in os.listdir(d))
    except OSError:
        return False


def train_already_extracted(dl_dir: str, ds_dir: str,
                            classes: list[str] | None) -> bool:
    """train 分卷已被删、但解压产物确实在盘上 → True（用于**阻止重下 70.4 GB**）。

    `download()` 的"已存在"判据只看 `_downloads/`，看不见"已经解压好"这个事实；
    而解压成功后会主动删掉分卷来省磁盘。两者叠加的后果是：**照脚本自己的提示
    "再跑一次"= 白下 70 GB**。这里把"已解压"这一事实补进判据。
    """
    if os.path.exists(os.path.join(dl_dir, "progan_train.7z.001")):
        return False                     # 分卷还在 → 走正常路径（还能补解其它类）
    want = list(classes) if classes else list(DEFAULT_TRAIN_CLASSES)
    root = os.path.join(ds_dir, "ForenSynths")
    for c in want:
        for sub in ("0_real", "1_fake"):
            # 两种位置都认：修正后的 ForenSynths/train/<类>/，以及历史误放的 ForenSynths/<类>/
            if not (_dir_has_files(os.path.join(root, "train", c, sub))
                    or _dir_has_files(os.path.join(root, c, sub))):
                return False
    return True

#: 官方给出的体积（字节），用于校验下载完整性
EXPECTED_SIZE = {
    "progan_val.zip": 830_792_545,
    "progan_testset.zip": 834_215_577,          # 约 0.80 GB，仅作参考
    "CNN_synth_testset.zip": 20_053_000_000,    # 约 18.7 GB，仅作参考
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; vibnet-dataset-fetcher/1.0)"}


# ---------------------------------------------------------------- 工具函数
def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def rel_or_abs(path: str) -> str:
    """尽量给相对路径；跨盘符时 relpath 会直接抛异常，那就退回绝对路径。

    （这个坑是自检抓出来的：临时目录在 C: 而项目在 D:，`os.path.relpath` 抛
    `ValueError: path is on mount 'C:', start on mount 'D:'`，把"空间不足"这个
    本该好好报出来的提示变成了栈回溯。）
    """
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


# ------------------------------------------------- 分卷内 zip 的「零中间产物」直读
class SplitVolumeReader:
    """把分卷文件（`.7z.001` … `.00N`）当成一个**连续、可随机读**的二进制流。

    为什么需要它
    ------------
    官方 `progan_train.7z.001~007` 的实测结构是「**7z(Copy，零压缩) 包着一个 zip**」
    （`Method = Copy`，内层 `progan_train.zip` 74.9 GB）。常规做法是先用 7z 把这
    74.9 GB 解出来，再解这个 zip，于是磁盘峰值 = 74.9（分卷）+ 74.9（中间 zip）
    = **149.8 GB**；本机 D 盘只剩 114 GB，这条路必然在中途把磁盘写满。

    但 zip 只需要「一个可 seek 的只读流」就能被 `zipfile` 打开。所以这里把若干
    分卷在**逻辑上拼成一个文件**，用 `start` / `length` 裁到恰好等于内层 zip 的
    那一段，直接交给 `zipfile` —— **零中间产物**，磁盘峰值只剩解压产物本身。

    只读；带 4 MB 预读缓存，避免逐 4~8 KB 反复 seek 拖慢顺序解压。
    """

    READAHEAD = 1 << 22                     # 4 MB

    def __init__(self, paths: list[str], start: int = 0, length: int | None = None):
        self._paths = [str(p) for p in paths]
        self._sizes = [os.path.getsize(p) for p in self._paths]
        self._starts: list[int] = []
        acc = 0
        for s in self._sizes:
            self._starts.append(acc)
            acc += s
        self._vol_total = acc
        end = acc if length is None else start + length
        if not (0 <= start < end <= acc):
            raise ValueError(f"非法区间 start={start} length={length} 分卷总大小={acc}")
        self._base, self._total = start, end - start
        self._pos = 0
        self._fh = None
        self._fh_idx = -1
        self._buf = b""
        self._buf_off = -1

    # ---- 只读流协议（zipfile 需要 seek/tell/read） -------------------
    def __len__(self) -> int:
        return self._total

    def tell(self) -> int:
        return self._pos

    def seek(self, off: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            p = off
        elif whence == os.SEEK_CUR:
            p = self._pos + off
        elif whence == os.SEEK_END:
            p = self._total + off
        else:
            raise ValueError(f"不支持的 whence={whence}")
        if p < 0:
            raise ValueError("seek 到负偏移")
        self._pos = p
        return p

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._fh is None and self._fh_idx == -1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh, self._fh_idx = None, -1
        self._buf, self._buf_off = b"", -1

    # ---- 实际读字节 --------------------------------------------------
    def _handle(self, vi: int):
        if self._fh_idx != vi:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self._paths[vi], "rb")
            self._fh_idx = vi
        return self._fh

    def _raw(self, logical: int, take: int) -> bytes:
        """从逻辑偏移 logical 处读 take 字节（带预读缓存）。"""
        if self._buf and self._buf_off <= logical < self._buf_off + len(self._buf):
            off = logical - self._buf_off
            if off + take <= len(self._buf):
                return self._buf[off:off + take]
        vi = bisect.bisect_right(self._starts, logical) - 1
        off_in_vol = logical - self._starts[vi]
        f = self._handle(vi)
        f.seek(off_in_vol)
        room = self._sizes[vi] - off_in_vol
        data = f.read(min(max(take, self.READAHEAD), room))
        self._buf, self._buf_off = data, logical
        return data[:take]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._total - self._pos
        n = min(n, self._total - self._pos)
        out = bytearray()
        while n > 0:
            chunk = self._raw(self._base + self._pos, n)
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def locate_inner_zip(volumes: list[str]) -> tuple[int, int] | None:
    """在分卷流里定位内层 zip 的 `[start, length)`，不依赖任何外部清单。

    实测本包是 **ZIP64**：经典 EOCD 里的中央目录偏移被写成哨兵值 `0xFFFFFFFF`
    （72 万条目也必然触发 ZIP64），真正的偏移/大小在 ZIP64 EOCD 记录里。所以这里
    同时支持 ZIP64 与经典两种布局，并且要求两条**互相独立**的证据都成立才返回：
        ① 中央目录必须正好紧接在 ZIP64 EOCD 记录之前（`cd_off + cd_size == d64`）
        ② 在 `cd_off` 处读到的必须是中央目录项签名 `PK\\x01\\x02`
    "看起来像"不算数 —— 定位错了会让 zipfile 读到垃圾，甚至把解压产物理错。
    """
    size = sum(os.path.getsize(p) for p in volumes)
    whole = SplitVolumeReader(volumes)
    try:
        start = whole.read(1 << 20).find(b"PK\x03\x04")
        if start < 0:
            return None

        back = min(1 << 18, size)
        whole.seek(size - back)
        tail = whole.read(back)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0 or len(tail) - eocd < 22:
            return None
        eocd_abs = size - back + eocd
        comment_len = tail[eocd + 20] | (tail[eocd + 21] << 8)

        loc = eocd - 20
        if loc >= 0 and tail[loc:loc + 4] == b"PK\x06\x07":
            # ---- ZIP64：哨兵值背后才是真数
            d64 = int.from_bytes(tail[loc + 8:loc + 16], "little")
            e64 = start + d64
            whole.seek(e64)
            rec = whole.read(56)
            if rec[:4] != b"PK\x06\x06":
                return None
            recsize = int.from_bytes(rec[4:12], "little")
            cd_size = int.from_bytes(rec[40:48], "little")
            cd_off = int.from_bytes(rec[48:56], "little")
            if cd_off + cd_size != d64:
                return None
            length = e64 + 12 + recsize + 20 + 22 + comment_len - start
        else:
            # ---- 经典 ZIP32 布局
            cd_size = int.from_bytes(tail[eocd + 12:eocd + 16], "little")
            cd_off = int.from_bytes(tail[eocd + 16:eocd + 20], "little")
            if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
                return None                 # 需要 ZIP64 却没找到定位器
            if cd_off + cd_size != eocd_abs - start:
                return None
            length = eocd_abs + 22 + comment_len - start

        if length <= 0:
            return None
        whole.seek(start + cd_off)
        if whole.read(4) != b"PK\x01\x02":
            return None
        return start, length
    finally:
        whole.close()


def open_train_zip_direct(dl_dir: str) -> tuple[zipfile.ZipFile, str] | None:
    """直接打开 `progan_train.7z.001~007` 里的内层 zip（不落中间文件）。

    成功返回 (已打开的 ZipFile, 人类可读的说明)；失败返回 None，调用方应退回
    常规的 7z 解压路径。返回的 ZipFile 由调用方负责 close()。
    """
    volumes = [os.path.join(dl_dir, f"progan_train.7z.{i:03d}") for i in range(1, 8)]
    missing = [os.path.basename(v) for v in volumes if not os.path.exists(v)]
    if missing:
        print(f"  [zip ] 分卷不全，缺 {missing}")
        return None
    loc = locate_inner_zip(volumes)
    if loc is None:
        return None
    start, length = loc
    rdr = SplitVolumeReader(volumes, start, length)
    try:
        z = zipfile.ZipFile(rdr)
    except Exception as e:                                          # noqa: BLE001
        rdr.close()
        print(f"  [zip ] 分卷内 zip 打开失败：{type(e).__name__}: {e}")
        return None
    return z, f"{len(volumes)} 卷内层 zip @偏移 {start}，{human(length)}"


def plan_zip(z: zipfile.ZipFile, classes: list[str] | None) -> tuple[list[str], int]:
    """挑出要解压的成员，并**精确**算出解压后总字节数（用于磁盘预检）。

    只遍历一次 `infolist()`：这批训练集有 **72 万条目**，再调一次 `namelist()`
    会多造一份同样大的字符串列表，本机内存本就吃紧。

    ⚠ 两类**文件**之外一概不选，目录条目尤其不能选：
    目录条目形如 `airplane/`，用 `"/<类名>/" in f"/{name}"` 去匹配时它**恒不成立**，
    于是有人写了 `i.is_dir() or ...` 兜着。但那等于"目录永远被选中" —— 结果是
    类名写错也会"成功解压 20 项"（其实是 20 个空目录），进而通过"解压完整"校验、
    把压缩包删掉。**实测就是这样把 74.9 GB 训练集分卷删没的。**
    父目录由 `zipfile.extract()` 自动创建，不需要我们操心。
    """
    infos = z.infolist()
    if classes:
        keys = tuple(f"/{c}/" for c in classes)
        sel = [i for i in infos
               if not i.is_dir() and any(k in f"/{i.filename}" for k in keys)]
    else:
        sel = [i for i in infos if not i.is_dir()]
    return [i.filename for i in sel], sum(i.file_size for i in sel)


#: 解压预检保留的安全余量（GB）：解压过程中还会写移动/临时文件
DISK_MARGIN_GB = 2.0


def preflight_space(need_bytes: float, work_dir: str, target: str,
                    ignore: bool = False) -> bool:
    """解压前算清空间。不够就**拒绝开工**，绝不让磁盘写到一半满。

    为什么必须有：本机 D 盘 114 GB 可用，而"先落 74.9 GB 中间 zip 再解压"
    这条老路峰值 149.8 GB —— 中途 ENOSPC 会留下半截 zip 和一个满盘，
    用户看到的是"跑了几小时最后失败"，且下次还得先手工清理。
    """
    free = shutil.disk_usage(work_dir).free
    need = need_bytes + DISK_MARGIN_GB * 1024 ** 3
    if need <= free or ignore:
        print(f"  [磁盘] 解压约需 {need / 1024 ** 3:.1f} GB，可用 {free / 1024 ** 3:.1f} GB")
        if ignore and need > free:
            print("  [磁盘] ⚠ 你用了 --ignore-disk-check，空间不足也继续，后果自负")
        return True
    print("  " + "!" * 64)
    print(f"  [磁盘] 空间不足：解压约需 {need / 1024 ** 3:.1f} GB，"
          f"可用仅 {free / 1024 ** 3:.1f} GB（差 {(need - free) / 1024 ** 3:.1f} GB）")
    print(f"         目标目录：{rel_or_abs(target)}")
    print("         可选做法：")
    print("           ① 加 --classes car cat chair horse 只解几个类（最省）")
    print("           ② 清出空间后重跑（本脚本支持续传，压缩包不会白下）")
    print("           ③ 确认无妨可加 --ignore-disk-check 强行继续")
    print("  " + "!" * 64)
    return False


def resolve_url(name: str, endpoint: str) -> str:
    return f"{endpoint}/datasets/{HF_REPO}/resolve/main/{name}"


def remote_size(url: str, tries: int = 2) -> int | None:
    """拿远端文件大小（跟随重定向，读 x-linked-size 或 content-length）。"""
    for _ in range(tries):
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                for key in ("x-linked-size", "Content-Length", "content-length"):
                    v = r.headers.get(key)
                    if v:
                        return int(v)
        except Exception:
            time.sleep(2)
    return None


def first_alive(urls: list) -> str | None:
    """返回第一个能连上的 URL（只取前 1KB 探活），都不通则返回 None。"""
    for u in urls:
        req = urllib.request.Request(u, headers={**UA, "Range": "bytes=0-1023"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
                return u
        except Exception:
            continue
    return None


def download(name: str, dest_dir: str, endpoint, retries: int = 12,
             endpoints: list | None = None) -> str | None:
    """下载单个文件，支持 HTTP Range 断点续传 **与多镜像自动切换**。

    为什么要多镜像 + 长退避
    ----------------------
    `huggingface.co` 在这类网络环境下会**间歇性**整体不可达（实测：
    同一 URL 早上能跑 8 MB/s，过一会儿连续 5 次探测全部超时）。
    原来的实现只重试 4 次、最长等 20 秒，一次十几分钟的阻断就能让它彻底失败；
    失败后它跳过该文件去下下一个，于是 7 个分卷**全军覆没**。

    现在的策略：
      * 每次尝试轮换镜像（默认 huggingface.co → hf-mirror.com）；
      * 指数退避 15s → 300s，给网络恢复留时间；
      * 每次失败都重新读取本地大小，断点续传，已下的部分绝不重下；
      * 默认重试 12 次，覆盖约 40 分钟的中断窗口。
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)

    if endpoints is None:
        endpoints = [endpoint] if isinstance(endpoint, str) else list(endpoint)
    urls = [resolve_url(name, ep) for ep in endpoints]

    total = remote_size(urls[0]) or (remote_size(urls[1]) if len(urls) > 1 else None)

    if os.path.exists(dest):
        cur = os.path.getsize(dest)
        if total and cur == total:
            print(f"  [skip] {name} 已完整 ({human(cur)})")
            return dest
        if total and cur > total:
            print(f"  [warn] {name} 本地比远端大，重新下载")
            os.remove(dest)
            cur = 0
        elif cur:
            print(f"  [resume] {name} 从 {human(cur)} 继续")
    else:
        cur = 0

    if total:
        print(f"  [get ] {name}  远端 {human(total)}"
              + (f"  剩余 {human(total - cur)}" if cur else ""))

    for attempt in range(retries):
        ep_i = attempt % len(urls)
        url = urls[ep_i]
        headers = dict(UA)
        if cur:
            headers["Range"] = f"bytes={cur}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r, \
                    open(dest, "ab" if cur else "wb") as f:
                got, last, t0 = cur, time.time(), time.time()
                while True:
                    chunk = r.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last >= 2.0:                      # 每 2 秒刷一次进度
                        speed = (got - cur) / max(now - t0, 1e-6)
                        if total:
                            pct = 100.0 * got / total
                            eta = (total - got) / max(speed, 1.0)
                            print(f"\r    {pct:5.1f}%  {human(got)}/{human(total)}  "
                                  f"{human(speed)}/s  剩余 {eta/60:.1f} 分钟   ",
                                  end="", flush=True)
                        else:
                            print(f"\r    {human(got)}  {human(speed)}/s   ",
                                  end="", flush=True)
                        last = now
            print()
            size = os.path.getsize(dest)
            if total and size != total:
                print(f"  [warn] {name} 大小不符：{size} != {total}，将续传")
                cur = size
                continue
            print(f"  [ok  ] {name}  {human(size)}")
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            cur = os.path.getsize(dest) if os.path.exists(dest) else 0
            short = f"{type(e).__name__}: {str(e)[:90]}"
            if attempt + 1 < retries:
                wait = min(15 * (2 ** min(attempt, 4)), 300)
                nxt = urls[(attempt + 1) % len(urls)]
                nxt_name = nxt.split("/")[2]
                print(f"\n  [retry {attempt + 1}/{retries}] {short}")
                print(f"          …{wait}s 后切到 {nxt_name} 续传"
                      f"（已存 {human(cur)}）", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [retry {attempt + 1}/{retries}] {short}")

    print(f"  [FAIL] {name} 重试 {retries} 次仍失败（已下载的部分保留，重跑本脚本会续传）")
    return None


def find_7z() -> str | None:
    """按优先级寻找 7z 可执行文件（项目内置 > 系统安装 > PATH）。"""
    here = os.path.join(ROOT, "tools")
    candidates = [
        os.path.join(here, "7zr.exe"),          # 项目内置独立版（推荐，免安装）
        os.path.join(here, "7z.exe"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z", "7za", "7z.exe",
    ]
    for c in candidates:
        if os.sep in c or "/" in c:
            if os.path.exists(c):
                return c
        else:
            p = shutil.which(c)
            if p:
                return p
    return None


def ensure_7z() -> str | None:
    """确保有可用的 7z；项目内没有时自动从 7-zip.org 下载独立版 7zr.exe。"""
    exe = find_7z()
    if exe:
        return exe
    url = "https://www.7-zip.org/a/7zr.exe"
    dest_dir = os.path.join(ROOT, "tools")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "7zr.exe")
    print(f"  [7z  ] 未找到 7z，尝试下载独立版 {url}")
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:                                          # noqa: BLE001
        print(f"  [7z  ] 下载失败：{type(e).__name__}: {e}")
        return None
    print(f"  [7z  ] 已就位 {dest}  ({human(os.path.getsize(dest))})")
    return dest


def extract_7z(first_volume: str, out_dir: str) -> bool:
    """用 7z 命令行解压多卷压缩包；没有 7z 时回退到 py7zr。"""
    exe = ensure_7z()
    if exe:
        print(f"  [7z  ] {os.path.basename(exe)} x {os.path.basename(first_volume)}")
        r = subprocess.run([exe, "x", first_volume, f"-o{out_dir}", "-y"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [7z  ] 失败：{r.stderr.strip()[:300]}")
            return False
        return True

    print("  [7z  ] 未找到 7z 可执行文件，改用 py7zr")
    try:
        import py7zr
    except ImportError:
        print("  [7z  ] py7zr 未安装。请执行其一：")
        print("          pip install py7zr")
        print("          或安装 7-Zip 后把 7z.exe 加进 PATH")
        return False
    with py7zr.SevenZipFile(first_volume, "r") as z:
        z.extractall(path=out_dir)
    return True


def extract_zip(archive, out_dir: str, classes: list[str] | None = None,
                members: list[str] | None = None) -> int:
    """解压 zip。`classes` 非空时只解包含这些类的成员（训练集瘦身）。

    `archive` 可以是路径，也可以是**已打开的 ZipFile**（分卷直读时就是后者，
    这样就不必先落一个 74.9 GB 的中间 zip）。
    """
    os.makedirs(out_dir, exist_ok=True)
    own = not isinstance(archive, zipfile.ZipFile)
    z = zipfile.ZipFile(archive) if own else archive
    n = 0
    try:
        if members is None:
            members, _ = plan_zip(z, classes)
        print(f"  [zip ] 解压 {len(members)} 项"
              + (f"（只取类：{', '.join(classes)}）" if classes else ""))
        for i, m in enumerate(members, 1):
            try:
                z.extract(m, out_dir)
                n += 1
            except Exception as e:                                  # noqa: BLE001
                print(f"  [warn] {m}: {type(e).__name__}: {e}")
            if i % 2000 == 0:
                print(f"\r    已解压 {i}/{len(members)}   ", end="", flush=True)
        print()
    finally:
        if own:
            z.close()
    return n


def _move_children(src_dir: str, dst_dir: str,
                   strip: tuple = ("train", "val", "test")) -> int:
    """把 src_dir 下的条目搬进 dst_dir。

    若唯一顶层目录名恰好是 train/val/test，则先剥掉这一层再搬
    （官方压缩包内部有时带一层 split 目录，有时不带，两种都要能吃）。
    目标已存在同名条目时**跳过并告警，绝不覆盖**已有数据。
    """
    os.makedirs(dst_dir, exist_ok=True)
    entries = [e for e in os.listdir(src_dir) if not e.startswith(".")]
    if len(entries) == 1:
        only = os.path.join(src_dir, entries[0])
        if entries[0] in strip and os.path.isdir(only):
            src_dir = only
            entries = [e for e in os.listdir(src_dir) if not e.startswith(".")]
    moved = skipped = 0
    for e in entries:
        s, d = os.path.join(src_dir, e), os.path.join(dst_dir, e)
        if os.path.exists(d):
            print(f"  [skip] {e} 已存在，不覆盖")
            skipped += 1
            continue
        shutil.move(s, d)
        moved += 1
    if skipped:
        print(f"  [warn] {skipped} 项因目标已存在被跳过")
    return moved


# ---------------------------------------------------------------- 自检
def self_test() -> int:
    """对「分卷直读」这条路径做回归自检（合成数据，秒级，不碰真实分卷）。

    为什么必须测：这条路一旦偏移算错，`zipfile` **不会报错** —— 它只会解出一堆
    错位的数据，属于典型的「不报错的静默失效」。真实包又是 ZIP64（经典 EOCD 里
    的偏移是 `0xFFFFFFFF` 哨兵值），靠肉眼根本看不出对错，只能用测试钉死。
    """
    import io
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, extra: str = "") -> None:
        checks.append((name, ok, extra))
        print(f"  {'✅' if ok else '❌'} {name}" + (f"   {extra}" if extra and not ok else ""))

    def real_zip() -> bytes:
        """造一个形如真实训练集的 zip：**既有目录条目、也有文件条目**。

        目录条目是关键 —— 真实包的目录条目就是那次误删事故的起因，
        夹具里必须有，否则这个 bug 在自检里根本暴露不出来。
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for d in ("car", "cat"):
                z.writestr(f"{d}/", b"")
                for sub in ("0_real", "1_fake"):
                    z.writestr(f"{d}/{sub}/", b"")
                    for i in range(3):
                        z.writestr(f"{d}/{sub}/img{i}.bin",
                                   bytes([i]) * 1000 if sub == "0_real" else b"x" * 700)
        return buf.getvalue()

    def nfiles(names) -> int:
        return len([n for n in names if not n.endswith("/")])

    def split_write(raw: bytes, d: str, nvol: int, wrap7z: bool) -> list[str]:
        """把 raw 切成 nvol 份写成 .001/.002…；wrap7z 时在最前面塞 32 字节模拟 7z 头。"""
        head = (b"7z\xbc\xaf'\x1c\x00\x04" + b"\x00" * 24) if wrap7z else b""
        blob = head + raw
        per = (len(blob) + nvol - 1) // nvol
        paths = []
        for k in range(nvol):
            p = os.path.join(d, f"progan_train.7z.{k + 1:03d}")
            with open(p, "wb") as f:
                f.write(blob[k * per:(k + 1) * per])
            paths.append(p)
        return paths

    print("  [自检] 分卷直读：合成 zip 切 3 卷 → 定位 → 逐字节读回")
    with tempfile.TemporaryDirectory() as d:
        raw = real_zip()
        vols = split_write(raw, d, 3, wrap7z=True)
        loc = locate_inner_zip(vols)
        chk("定位到 7z 头之后的 zip（start=32）", loc is not None and loc[0] == 32,
            f"loc={loc}")
        if loc:
            start, length = loc
            chk("长度等于 zip 真实大小", length == len(raw),
                f"{length} vs {len(raw)}")
            rdr = SplitVolumeReader(vols, start, length)
            back = rdr.read()
            rdr.close()
            chk("逐字节读回与原始 zip 相同", back == raw)
            rdr = SplitVolumeReader(vols, start, length)
            with zipfile.ZipFile(rdr) as z:
                names = sorted(z.namelist())
                body = z.read("car/0_real/img2.bin")
            rdr.close()
            chk("zipfile 能打开并正确解出成员", body == bytes([2]) * 1000)
            chk("文件条目完整（12 项）", nfiles(names) == 12, f"{nfiles(names)}")

    print("  [自检] 空间预检：不够时必须拒绝开工（而不是硬写满磁盘）")
    with tempfile.TemporaryDirectory() as d:
        chk("空间充足 → 放行", preflight_space(0, d, d) is True)
        chk("需要 1 PB → 拒绝", preflight_space(1 << 50, d, d) is False)
        chk("强制忽略 → 放行", preflight_space(1 << 50, d, d, ignore=True) is True)

    print("  [自检] 类别筛选：--classes 必须**只**留下指定的类")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.zip")
        with open(p, "wb") as f:
            f.write(real_zip())
        with zipfile.ZipFile(p) as z:
            members, need = plan_zip(z, ["car"])
            chk("只选 car 时不含任何 cat 成员",
                bool(members) and all("/cat/" not in m for m in members),
                f"n={len(members)}")
            chk("car 的 6 个文件都在", nfiles(members) == 6, f"{len(members)}")
            chk("目录条目不进成员表（它们会被 zipfile 自动建出来）",
                all(not m.endswith("/") for m in members))
            chk("体积估算只算被选中的项", 0 < need <= 6 * 1000)
            allm, allneed = plan_zip(z, None)
            chk("不筛选时拿到全部 12 个文件", nfiles(allm) == 12, f"{nfiles(allm)}")

            # ★ 这一条是一次真实事故的回归测试（见 D32）：
            #   类名写错时，若把目录条目也算作"选中"，就会"成功解压 20 个空目录"
            #   并通过完整性校验，然后删掉压缩包 —— 74.9 GB 训练集就是这么没的。
            bogus, bneed = plan_zip(z, ["zzz_nonexistent"])
            chk("★ 类名写错 → 一个成员都不选（否则会误删压缩包）",
                bogus == [] and bneed == 0, f"n={len(bogus)}")
            bogus2, _ = plan_zip(z, ["car", "zzz_nonexistent"])
            chk("★ 部分写错 → 只取对的那部分，不因错名而全空",
                nfiles(bogus2) == 6, f"n={len(bogus2)}")

    print("  [自检] ZIP64 分支：真实包就是 ZIP64，这里用合成结构验证算术")
    with tempfile.TemporaryDirectory() as d:
        cd_off, cd_size = 4096, 512
        body = bytearray(b"\x00" * 32)          # 32 字节 7z 签名头占位
        body += b"PK\x03\x04" + b"\x00" * 1000  # 假装是 zip 的本地头
        # ⚠ cd_off 是**相对 zip 起点**的偏移，而 zip 起点在 7z 头之后（32）。
        #   摆中央目录时必须按绝对位置 32+cd_off 垫，否则夹具本身就是错的。
        body += b"\x00" * (32 + cd_off - len(body))
        body += b"PK\x01\x02" + b"\x00" * (cd_size - 4)
        d64 = cd_off + cd_size
        e64 = 32 + d64
        rec = (b"PK\x06\x06" + (44).to_bytes(8, "little") + b"\x00" * 28
               + cd_size.to_bytes(8, "little") + cd_off.to_bytes(8, "little"))
        locator = (b"PK\x06\x07" + b"\x00" * 4 + d64.to_bytes(8, "little")
                   + b"\x00" * 4)
        eocd = (b"PK\x05\x06" + b"\x00" * 4 + b"\xff" * 4 + b"\xff" * 4
                + b"\x00" * 4 + b"\x00" * 2)
        body += rec + locator + eocd
        paths = []
        per = (len(body) + 2) // 3
        for k in range(3):
            q = os.path.join(d, f"progan_train.7z.{k + 1:03d}")
            with open(q, "wb") as f:
                f.write(bytes(body[k * per:(k + 1) * per]))
            paths.append(q)
        loc = locate_inner_zip(paths)
        want_len = e64 + 12 + 44 + 20 + 22 - 32
        chk("ZIP64 哨兵值布局能定位", loc is not None, f"loc={loc}")
        if loc:
            chk("ZIP64 算出的总长正确", loc[1] == want_len,
                f"{loc[1]} vs {want_len}")
        # 反向对照：真正的 ZIP32 结构里若 cd 偏移被改坏，必须拒绝
        bad = bytearray(body)
        bad[e64 + 48:e64 + 56] = (cd_off + 7).to_bytes(8, "little")
        q = os.path.join(d, "progan_train.7z.001")
        with open(q, "wb") as f:
            f.write(bytes(bad))
        chk("cd 偏移与长度不自洽 → 拒绝（宁可退回 7z 也不解出错位数据）",
            locate_inner_zip([q]) is None)

    print("  [自检] 分卷遍历 / 归位目录：一次真实上云事故的回归（2026-09-24）")
    vols = SPLIT_FILES["train"]
    chk("首卷 .001 不该被判为非首卷", _is_nonfirst_volume(vols[0]) is False)
    chk("后续卷 .002~.007 必须被判为非首卷（否则会去开一个已被删掉的文件）",
        all(_is_nonfirst_volume(v) for v in vols[1:]), f"{vols[1:]}")
    chk("单卷包（val / test / progan_test）不受影响",
        not any(_is_nonfirst_volume(v) for v in
                SPLIT_FILES["val"] + SPLIT_FILES["test"] + SPLIT_FILES["progan_test"]))
    chk("★ 首卷必须归位到 ForenSynths/train"
        "（否则 GenSynthsDataset(split='train') 一条样本都扫不到）",
        target_rel_for(vols[0]) == "ForenSynths/train", f"{target_rel_for(vols[0])}")
    chk("★ 归位目录不得是默认值 ForenSynths（键写内层名、磁盘上却是分卷名，原 bug）",
        target_rel_for(vols[0]) != "ForenSynths")

    print("  [自检] 已解压判据：防止『照脚本提示再跑一次』白下 70.4 GB")
    with tempfile.TemporaryDirectory() as d:
        dl, ds = os.path.join(d, "_downloads"), os.path.join(d, "Datasets")
        os.makedirs(dl)
        os.makedirs(ds)
        chk("什么都没有 → 不算已解压",
            train_already_extracted(dl, ds, ["car"]) is False)
        for sub in ("0_real", "1_fake"):
            q = os.path.join(ds, "ForenSynths", "train", "car", sub)
            os.makedirs(q)
            open(os.path.join(q, "a.png"), "wb").close()
        chk("train/car 两半都有文件 → 已解压",
            train_already_extracted(dl, ds, ["car"]) is True)
        chk("要的类里有一个缺 → 不算已解压（该补解，不该短路）",
            train_already_extracted(dl, ds, ["car", "cat"]) is False)
        for c in ("cat",):
            for sub in ("0_real", "1_fake"):
                q = os.path.join(ds, "ForenSynths", c, sub)
                os.makedirs(q)
                open(os.path.join(q, "a.png"), "wb").close()
        chk("★ 误放在 ForenSynths/<类>/ 也要认（否则会被判成没解压 → 重下 70 GB）",
            train_already_extracted(dl, ds, ["car", "cat"]) is True)
        open(os.path.join(dl, "progan_train.7z.001"), "wb").close()
        chk("分卷仍在 → 必须返回 False（要让它走正常解压路径，可补解其它类）",
            train_already_extracted(dl, ds, ["car"]) is False)

    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  自检结果：{n_ok}/{len(checks)} 通过")
    for name, ok, extra in checks:
        if not ok:
            print(f"    ❌ {name}  {extra}")
    return 0 if n_ok == len(checks) else 1


# ---------------------------------------------------------------- 主流程
def list_train(dl_dir: str) -> int:
    """列出训练集分卷内的类别构成（不解压、不落盘）。

    价值：`--classes` 要写哪些名字，此前只能靠猜。这里直接从分卷里读出内层 zip
    的中央目录（只读几十 KB），给出每个类的条目数与解压后体积，让"要花多少磁盘"
    变成一个可预算的数字，而不是解到一半才发现不够。
    """
    print("=" * 68)
    print("训练集内容盘点（从 7z 分卷直读，不落盘）")
    print("=" * 68)
    direct = open_train_zip_direct(dl_dir)
    if direct is None:
        print("❌ 无法从分卷定位内层 zip。可先确认分卷是否完整（--check-only）。")
        return 1
    z, how = direct
    try:
        groups: dict[str, list] = {}
        for i in z.infolist():
            if i.is_dir():
                continue
            top = i.filename.replace("\\", "/").split("/")[0]
            groups.setdefault(top, []).append(i)
        total_n = sum(len(v) for v in groups.values())
        total_b = sum(i.file_size for v in groups.values() for i in v)
        print(f"内层 zip：{how}")
        print(f"总条目 {total_n}，解压后约 {human(total_b)}\n")
        print(f"  {'类名':<28}{'条目数':>8}{'解压后':>12}")
        print("  " + "-" * 48)
        for k in sorted(groups, key=lambda x: -len(groups[x])):
            b = sum(i.file_size for i in groups[k])
            print(f"  {k:<28}{len(groups[k]):>8}{human(b):>12}")
        print("\n想只解其中几类，就把类名原样传给 --classes，例如：")
        sample = [k for k in sorted(groups) if "/" not in k][:4]
        print(f"  python scripts/fetch_datasets.py --split train "
              f"--classes {' '.join(sample)}")
        print("（脚本会先做空间预检；不足会直接拒绝开工，不会再写满磁盘）")
    finally:
        z.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="下载 ForenSynths(CNNDetection) 官方数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=os.path.join(ROOT, "data"),
                    help="数据根目录，默认 <项目>/data")
    ap.add_argument("--split", nargs="+", default=["val"],
                    choices=list(SPLIT_FILES) + ["all"],
                    help="要下载的划分，可多选，all=全部（约 90GB）")
    ap.add_argument("--mirror", default="auto",
                    choices=list(ENDPOINTS) + ["auto"],
                    help="下载源。auto(默认)=按 hf → hf-mirror 顺序自动切换，"
                         "单个源不通时自动换下一个（推荐，抗间歇性阻断）")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="只解压训练集里这些类，如 car cat chair horse")
    ap.add_argument("--keep-archive", action="store_true",
                    help="解压后保留压缩包（默认删除以省磁盘）")
    ap.add_argument("--check-only", action="store_true",
                    help="只检查本地文件，不下载")
    ap.add_argument("--no-extract", action="store_true", help="只下载不解压")
    ap.add_argument("--ignore-disk-check", action="store_true",
                    help="解压前不做空间预检（默认空间不足会拒绝开工，避免写满磁盘）")
    ap.add_argument("--list-train", action="store_true",
                    help="只列出训练集分卷内的类别与条目数，不解压（零磁盘开销，"
                         "用于确认 --classes 该写哪些名字）")
    ap.add_argument("--self-test", action="store_true",
                    help="对分卷直读/空间预检/类别筛选做回归自检（合成数据，不下载）")
    args = ap.parse_args()

    if args.self_test:
        print("=" * 68)
        print("fetch_datasets 自检")
        print("=" * 68)
        return self_test()

    dl_dir = os.path.join(args.root, "_downloads")
    ds_dir = os.path.join(args.root, "Datasets")
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(ds_dir, exist_ok=True)

    splits = list(SPLIT_FILES) if "all" in args.split else args.split
    todo: list[str] = []
    for s in splits:
        for f in SPLIT_FILES[s]:
            if f not in todo:
                todo.append(f)

    # 镜像顺序：auto 时按"先直连、再镜像"排。实测两者都可能间歇不可达，
    # 所以真正起作用的是 download() 里那套「失败就换下一个 + 指数退避」。
    if args.mirror == "auto":
        ep_list = [ENDPOINTS["hf"], ENDPOINTS["hf-mirror"]]
    else:
        ep_list = [ENDPOINTS[args.mirror]] + \
                  [v for k, v in ENDPOINTS.items() if v != ENDPOINTS[args.mirror]]
    endpoint = ep_list[0]
    print("=" * 68)
    print(f"数据源   : {' → '.join(a.split('//')[-1] for a in ep_list)}"
          f"  (共 {len(ep_list)} 个，失败自动切换)")
    print(f"仓库     : {HF_REPO}")
    print(f"下载目录 : {dl_dir}")
    print(f"数据目录 : {ds_dir}")
    print(f"待下载   : {len(todo)} 个文件")
    print(f"磁盘可用 : {free_gb(args.root):.1f} GB")
    print("=" * 68)

    if args.check_only:
        for f in todo:
            p = os.path.join(dl_dir, f)
            if os.path.exists(p):
                print(f"  [有] {f}  {human(os.path.getsize(p))}")
            else:
                print(f"  [缺] {f}")
        return 0

    # ---- 零开销盘点：直接从分卷里读出内层 zip 的目录，不落任何文件 ----
    if args.list_train:
        return list_train(dl_dir)

    need = 0
    for f in todo:
        p = os.path.join(dl_dir, f)
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if f.endswith(".7z.001"):
            need += 70.4 * 1024 ** 3 - have
        elif f.startswith("CNN_synth"):
            need += 18.7 * 1024 ** 3 - have
        else:
            need += 0.8 * 1024 ** 3 - have
    need_gb = max(need, 0) / 1024 ** 3
    avail = free_gb(args.root)
    if need_gb > avail * 0.9:
        print(f"[警告] 预计还需 {need_gb:.1f} GB，磁盘仅剩 {avail:.1f} GB。")
        print("       建议加 --classes car cat chair horse 分批处理，或清理后重跑。")
        if need_gb > avail:
            print("[中止] 空间不足。")
            return 2
    else:
        print(f"[磁盘] 预计还需 {need_gb:.1f} GB，可用 {avail:.1f} GB，充足。\n")

    ok, fail = [], []
    pre_extracted: set[str] = set()      # 「已解压」的条目：跳过下载，也跳过解压
    for f in todo:
        if f in SPLIT_FILES["train"] and \
                train_already_extracted(dl_dir, ds_dir, args.classes):
            ok.append(f)
            pre_extracted.add(f)
            continue
        p = download(f, dl_dir, endpoint, endpoints=ep_list)
        (ok if p else fail).append(f)

    if pre_extracted:
        print(f"\n  [skip] train 分卷已删、但目标已解压 → 跳过 {len(pre_extracted)} 个分卷，"
              f"不重下 70.4 GB")
        root = os.path.join(ds_dir, "ForenSynths")
        misplaced = [c for c in (args.classes or list(DEFAULT_TRAIN_CLASSES))
                     if _dir_has_files(os.path.join(root, c, "0_real"))
                     and not os.path.isdir(os.path.join(root, "train", c))]
        if misplaced:
            print("         ⚠ 检测到历史误放：train 落在 ForenSynths/<类>/，"
                  "而 split=\"train\" 只认 ForenSynths/train/。先归位：")
            print(f"           cd {root} && mkdir -p train && "
                  f"mv {' '.join(misplaced)} train/")

    # ---- 补跑一遍失败的卷 ----
    # 为什么需要：download() 对单个文件重试 12 次后就放弃、并**继续下一个文件**，
    # 所以一次运行可能留下「某个卷差最后几百 MB」的半成品（实测 progan_train.7z.001
    # 就曾停在 9.73/10.0 GiB）。若直接进解压，7z 分卷不全必然失败，
    # 用户看到的就是「下载跑了几小时，解压却报错」。
    # 这里再扫一遍失败项 —— 有断点续传，代价只是补那点尾巴。
    if fail:
        print("\n" + "=" * 68)
        print(f"补跑 {len(fail)} 个未完成的文件（断点续传）")
        print("=" * 68)
        still = []
        for f in fail:
            print(f"\n  [补跑] {f}")
            p = download(f, dl_dir, endpoint, endpoints=ep_list)
            if p:
                ok.append(f)
                print(f"  [补跑成功] {f}")
            else:
                still.append(f)
        fail = still

    if fail:
        print("\n" + "!" * 68)
        print(f"[警告] 仍有 {len(fail)} 个文件未完成：{fail}")
        print("       这些文件不会参与解压。请再跑一次本脚本（支持续传），")
        print("       或先用已完整的子集训练（例如 --split val）。")
        print("!" * 68)

    if args.no_extract:
        print("\n[--no-extract] 跳过解压。")
        return 0 if not fail else 1

    # ---- 解压：先解到 staging，再规范化搬入目标目录（见 EXTRACT_TARGET 处的说明）
    print("\n" + "=" * 68)
    print("开始解压")
    print("=" * 68)

    # 首卷缺失时，整组多卷包都无法解压（非首卷不能单独打开）。与其让循环拿着
    # `.002` 去 `ZipFile()` 抛一个让人摸不着头脑的 FileNotFoundError，不如先说清楚。
    if any(_is_nonfirst_volume(f) for f in ok) and \
            not any(f.endswith(".7z.001") for f in ok):
        print("  [warn] 训练集分卷缺首卷 .001 —— 整组都无法解压（非首卷不能单独打开）。")
        print("         先补齐首卷： python scripts/fetch_datasets.py --split train")

    for f in ok:
        if f in pre_extracted:
            print(f"  [skip] {f}（已解压过，不重复解压）")
            continue
        if _is_nonfirst_volume(f):
            # 首卷分支已把整组卷一次性解完（并把 .001~.007 删掉）。继续遍历非首卷
            # 只会去打开一个已不存在的文件 —— 见 _is_nonfirst_volume 的注释。
            continue
        p = os.path.join(dl_dir, f)
        target_rel = target_rel_for(f)
        target = os.path.join(ds_dir, target_rel)
        # ⚠ 修复 D31：此前 `--classes` 只对 CNN_synth 生效（`if f.startswith("CNN_synth")`），
        # 而文档却写着"训练集加 --classes car cat chair horse 省磁盘" —— 承诺了却不生效，
        # 用户会以为省了磁盘、结果被解满。现在对所有需要瘦身的包一律生效。
        selective = list(args.classes) if args.classes else None

        opened: list[zipfile.ZipFile] = []
        if f.endswith(".7z.001"):
            direct = open_train_zip_direct(dl_dir)
            if direct is not None:
                z, how = direct
                opened.append(z)
                archives = [(z, "progan_train.zip")]
                print(f"  [zip ] 直读分卷内的 zip：{how}")
                print("         ✅ 不落 74.9 GB 中间文件（磁盘峰值只剩解压产物本身）")
            else:
                print("  [zip ] 分卷内 zip 未定位成功，退回 7z 解压")
                print("         ⚠ 该路径要额外 74.9 GB 存放中间 zip，磁盘会紧张")
                if not extract_7z(p, dl_dir):
                    fail.append(f)
                    continue
                inner = os.path.join(dl_dir, "progan_train.zip")
                if not os.path.exists(inner):
                    print("  [warn] 未找到解压产物 progan_train.zip")
                    continue
                archives = [(inner, os.path.basename(inner))]
        else:
            archives = [(p, os.path.basename(p))]

        aborted = False
        incomplete = False
        for arc, arc_name in archives:
            own = not isinstance(arc, zipfile.ZipFile)
            z = zipfile.ZipFile(arc) if own else arc
            try:
                members, need = plan_zip(z, selective)
                # ⚠ 危险缺口：类名写错时 members 为空，而"解压 0 项"会被当成**成功**，
                # 接着就走到下面的删除分支 —— 于是 74.9 GB 分卷被删、什么都没解出来。
                # 这类"零产物却报成功"正是最该拦住的一种静默失效，这里直接终止。
                if not members:
                    why = (f"--classes 里的类名一个都没命中（{' '.join(selective)}）"
                           if selective else "内层 zip 的目录是空的（文件可能已损坏）")
                    print(f"  ✗ {why}")
                    print("    未改动任何文件，压缩包保留。可先盘点有哪些类：")
                    print("      python scripts/fetch_datasets.py --split train --list-train")
                    fail.append(f)
                    aborted = True
                    break
                if not preflight_space(need, dl_dir, target, args.ignore_disk_check):
                    fail.append(f)
                    aborted = True
                    break
                staging = os.path.join(
                    dl_dir, "_extract", os.path.splitext(arc_name)[0])
                if os.path.isdir(staging):
                    shutil.rmtree(staging)
                os.makedirs(staging, exist_ok=True)
                n = extract_zip(z, staging, selective, members=members)
                # 少解了一项就不算成功：*绝不能*在这时删掉源压缩包，
                # 否则用户要为一个静默的跳过再重下 74.9 GB。
                # `n == 0` 单独判一次：哪怕 members 非空，"一个文件都没出来"也
                # 一律视为失败 —— 这是删包前最后一道闸门。
                if n == 0 or n < len(members):
                    incomplete = True
                    print(f"  [warn] 计划 {len(members)} 项，实际成功 {n} 项，"
                          f"压缩包将保留（不删）")
                moved = _move_children(staging, target)
                print(f"  [put ] {moved} 项归位到 {target_rel}/")
                shutil.rmtree(staging, ignore_errors=True)
            finally:
                if own:
                    z.close()
        for z in opened:
            z.close()
        if aborted:
            continue
        if incomplete:
            fail.append(f)
            continue

        if not args.keep_archive:
            if os.path.exists(p):
                os.remove(p)
            if f.endswith(".7z.001"):
                for i in range(1, 8):
                    v = os.path.join(dl_dir, f"progan_train.7z.{i:03d}")
                    if os.path.exists(v):
                        os.remove(v)
                inner = os.path.join(dl_dir, "progan_train.zip")
                if os.path.exists(inner):
                    os.remove(inner)
                print("  [del ] progan_train.7z.001~007 + progan_train.zip（已解压）")
            else:
                print(f"  [del ] {f}（已解压，省磁盘）")

    print("\n" + "=" * 68)
    print(f"完成：成功 {len(ok)} / 失败 {len(fail)}")
    if fail:
        print("失败文件：" + ", ".join(fail))
    print(f"磁盘剩余：{free_gb(args.root):.1f} GB")
    print("\n下一步：检查目录结构是否正确")
    print(f"  python scripts/prepare_datasets.py --root {args.root}/Datasets --check")
    print("=" * 68)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
