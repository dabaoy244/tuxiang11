"""在**可见的控制台窗口**里跑一条命令，并把输出同时落一份到日志。

为什么需要它（而不是直接 `cmd /k ...`）：
- `cmd /k "title X & ..."` 里的 `title` 内建命令在沙箱下的引号处理不可靠（实测不生效），
  结果窗口标题一直是默认值，按标题找窗口就会扑空。
- 这里改成由 **自己** 调 `SetConsoleTitleW` 设标题，标题一定是我们想要的那个。
- 顺便用 `SetConsoleScreenBufferSize` + `SetConsoleWindowInfo` 把窗口调大，
  否则默认 80x25 一行行折行，截出来的图没法看。

用法（由 vis.py 启动，一般不用手敲）：
    python console_run.py --title "VIB-Net-CLOUD" --tail-lines 500 -- <命令> [参数...]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

_k32 = ctypes.windll.kernel32
STD_OUTPUT_HANDLE = -11


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class SMALL_RECT(ctypes.Structure):
    _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]


def set_title(title: str) -> None:
    _k32.SetConsoleTitleW(title)


ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
ENABLE_PROCESSED_OUTPUT = 0x0001


def enable_vt() -> bool:
    """打开 ANSI 转义支持。

    不开的话，`\\x1b[1;36m` 会被原样打在屏幕上（截图里就是一串 `←[1;36m`），
    既难看又占位置。Windows 10+ 的 conhost 支持 VT，只是默认关着。
    """
    h = _k32.GetStdHandle(STD_OUTPUT_HANDLE)
    if h in (0, -1):
        return False
    mode = wt.DWORD()
    if not _k32.GetConsoleMode(h, ctypes.byref(mode)):
        return False
    return bool(_k32.SetConsoleMode(h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                                    | ENABLE_PROCESSED_OUTPUT))


def set_size(cols: int, rows: int, buffer_lines: int = 4000) -> bool:
    """先放大缓冲区，再缩小可视窗口——顺序反了会被 Windows 拒绝（缓冲区不能小于窗口）。

    缓冲区留 4000 行是为了让输出**滚回去**还在，截图时能一次看到更多历史。
    """
    h = _k32.GetStdHandle(STD_OUTPUT_HANDLE)
    if h in (0, -1):
        return False
    ok = bool(_k32.SetConsoleScreenBufferSize(h, COORD(cols, buffer_lines)))
    rect = SMALL_RECT(0, 0, cols - 1, rows - 1)
    ok = bool(_k32.SetConsoleWindowInfo(h, True, ctypes.byref(rect))) and ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--cols", type=int, default=150)
    ap.add_argument("--rows", type=int, default=44)
    ap.add_argument("--log", default=None, help="同时把输出写到这个文件")
    ap.add_argument("--banner", default=None, help="开跑前打印的一行说明")
    ap.add_argument("--hwnd-file", default=None,
                    help="把自己的控制台窗口句柄写到这里（外部按句柄截图，不用猜标题）")
    ap.add_argument("--hold", type=int, default=0,
                    help="命令跑完后窗口再停留这么多秒，便于截到**最终结果**那一屏")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        print("没有给出要执行的命令", file=sys.stderr)
        return 2

    set_title(a.title)
    vt = enable_vt()
    sized = set_size(a.cols, a.rows)
    set_title(a.title)          # 改尺寸可能重置标题，再设一次

    # ★ 先把自己的窗口句柄交出去，外部才可能 100% 对准这个窗口。
    #   GetConsoleWindow() 拿到的就是这个进程的控制台窗口，不需要任何匹配。
    if a.hwnd_file:
        try:
            h = int(_k32.GetConsoleWindow() or 0)
            os.makedirs(os.path.dirname(os.path.abspath(a.hwnd_file)), exist_ok=True)
            with open(a.hwnd_file, "w", encoding="utf-8") as f:
                f.write(f"{h}\n{a.title}\n")
        except OSError:
            pass

    out = sys.stdout.buffer

    def w(s: str) -> None:
        if not vt:              # 没有 VT 支持就把转义序列剥掉，别把它当正文打出来
            import re
            s = re.sub(r"\x1b\[[0-9;]*m", "", s)
        out.write(s.encode("utf-8", "replace"))
        out.flush()

    w(f"\x1b[1;36m=== {a.title} ===\x1b[0m\n")
    w(f"主机: {os.environ.get('COMPUTERNAME','?')}   时间: " 
      + subprocess.run(["cmd", "/c", "echo", "%DATE% %TIME%"], capture_output=True
                       ).stdout.decode("gbk", "replace").strip() + "\n")
    if a.banner:
        w(f"\x1b[1;33m{a.banner}\x1b[0m\n")
    w(f"窗口尺寸设置: {'成功' if sized else '失败(沿用默认)'}  缓冲=4000 行"
      f"  ANSI颜色: {'开' if vt else '关(已剥除转义码)'}\n")
    w("命令: " + " ".join(cmd) + "\n")
    w("-" * 100 + "\n")

    logf = open(a.log, "ab") if a.log else None
    if logf:
        logf.write(f"\n\n===== {a.title}  {cmd} =====\n".encode("utf-8"))

    # 逐行 tee：既写控制台（用户看得见），又写日志文件（我读得到）。
    # 刻意按**字节**读再解码，避免中文 Windows 下 GBK/UTF-8 混排把读取线程炸掉。
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    assert p.stdout is not None
    for raw in iter(p.stdout.readline, b""):
        out.write(raw); out.flush()
        if logf:
            logf.write(raw); logf.flush()
    rc = p.wait()

    w("-" * 100 + "\n")
    w(f"\x1b[1;31m[命令结束] 退出码 = {rc}\x1b[0m\n")
    if logf:
        logf.write(f"[命令结束] 退出码 = {rc}\n".encode("utf-8"))

    # ★ 哨兵文件：外部**别用 grep 日志**来判断"跑完了没"。
    #   实测日志里混着 ANSI 转义码和非 UTF-8 字节时，grep 会静默漏匹配，
    #   于是等待循环白等到超时。写一个专属的 .done 文件，判据只剩"存在与否"。
    if a.log:
        try:
            with open(a.log + ".done", "w", encoding="utf-8") as f:
                f.write(f"rc={rc}\n")
        except OSError:
            pass

    # 命令一结束就退出的话，控制台窗口会立刻关闭，外部**永远截不到结果那一屏**
    # （实测：doctor 约 30 秒跑完，我 30 秒后去截图，窗口已经不在了）。
    # --hold 让窗口把最终画面留在屏幕上。
    if a.hold > 0:
        w(f"\x1b[1;33m[窗口保持 {a.hold} 秒，便于截图；关闭本窗口不影响云端]\x1b[0m\n")
        if logf:
            logf.write(f"[窗口保持 {a.hold} 秒]\n".encode("utf-8"))
        try:
            time.sleep(a.hold)
        except KeyboardInterrupt:
            pass

    if logf:
        logf.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
