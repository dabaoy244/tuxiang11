"""「看得见的云端会话」控制器 —— 起窗 / 找窗 / 截窗。

配合 console_run.py 使用：
    console_run.py  在窗口**内部**跑命令（负责设标题、调大小、tee 日志）
    vis.py          在窗口**外部**操控它（负责启动、定位、截图）

常用：
    python vis.py launch --title VIB-Net-CLOUD --log outputs/screen/session.log -- ssh -p 34293 ...
    python vis.py find   --title VIB-Net-CLOUD
    python vis.py shot   --title VIB-Net-CLOUD outputs/screenshots/01_step.png
    python vis.py shot   --title VIB-Net-CLOUD out.png --no-focus   # 不抢焦点
    python vis.py list                                             # 列出所有控制台窗口

为什么截图要按窗口裁剪而不是全屏：
    全屏 2560x1440 里真正有信息的只有那个控制台，裁剪后截图又小又清楚，
    而且**不包含用户桌面上的其他隐私窗口**（微信、QQ、浏览器标签）——
    这些图是要交给你看、可能贴进汇报材料的，只截该截的那块。

★ 2026-09-29 又踩一层坑：**裁剪 ≠ 安全**。
  裁剪只是"按坐标从屏幕抓一块"，如果目标窗口此时被别的窗口盖住，
  抓到的是**遮挡物**。实测就拍下过整个 WorkBuddy 桌面（含用户会话列表）。
  SetForegroundWindow 并不能保证成功 —— 前台进程可以拒绝焦点切换。
  ⇒ 默认改用 **PrintWindow(hwnd, PW_RENDERFULLCONTENT)** 直接向窗口要位图，
    它与窗口的可见性/前后台关系无关，被遮挡也拿到正确画面；
    仅当 PrintWindow 拿到近纯色时才回退"置顶+抓屏"，且回退路径**强制校验**
    目标窗口确实是前台窗口，否则直接失败退出，绝不返回可疑图片。
  用法：`vis.py shot <out> --hwnd <N>`（句柄从 `vis.py list` 取）
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageGrab

_u32 = ctypes.windll.user32
_g32 = ctypes.windll.gdi32
WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]

HERE = Path(__file__).resolve().parent
CONSOLE_RUN = HERE / "console_run.py"
CREATE_NEW_CONSOLE = 0x00000010
SW_RESTORE = 9
SW_MAXIMIZE = 3

# ★ 窗口句柄比窗口标题可靠得多。
#   实测坑：标题里带中文/中点（如 "VIB-Net 云端 · 1 环境体检"），经 Git Bash 传给
#   Windows 进程时会被改坏，FindWindowW 就永远找不到 —— 于是截图**静默退回全屏**，
#   把用户桌面上微信/QQ/浏览器标签一起截进取证图里。这是不可接受的泄露。
#   所以 launch 时把 hwnd 落盘，shot 直接按 hwnd 截，标题只用于给人看。
STATE = HERE / "window_state.json"


def _python() -> str:
    return sys.executable


def find_window(title: str) -> int:
    return int(_u32.FindWindowW(None, title) or 0)


def save_state(title: str, hwnd: int) -> None:
    try:
        STATE.write_text(json.dumps({"title": title, "hwnd": hwnd,
                                     "ts": time.strftime("%F %T")},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_state_hwnd() -> int:
    """从上一次 launch 的记录里取 hwnd，并确认这个窗口还活着。"""
    try:
        d = json.loads(STATE.read_text(encoding="utf-8"))
        h = int(d.get("hwnd", 0))
    except Exception:  # noqa: BLE001
        return 0
    return h if h and _u32.IsWindow(h) else 0


def foreground_hwnd() -> int:
    return int(_u32.GetForegroundWindow() or 0)


def rect_of(hwnd: int) -> tuple[int, int, int, int]:
    r = wt.RECT()
    if not _u32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(r)):
        raise OSError("GetWindowRect 失败")
    return int(r.left), int(r.top), int(r.right), int(r.bottom)


def grab_via_printwindow(hwnd: int) -> tuple[Image.Image | None, str]:
    """用 PrintWindow 抓窗口**自身**画面 —— 被别的窗口挡住也能拿到正确内容。

    ★ 为什么必须这样：`ImageGrab.grab()` 抓的是"屏幕上那块矩形"。
      目标窗口一旦被别的窗口覆盖，抓到的就是**遮挡物**。
      2026-09-29 实测踩到：看板窗口被 WorkBuddy 挡住，取证图直接拍下了
      整个桌面（含用户会话列表）—— 截图工具变成隐私泄露源。
      而 `SetForegroundWindow` 并不可靠：前台进程有权拒绝焦点切换。
      所以默认走 PrintWindow（不依赖窗口可见性/前后台关系）。
    """
    l, t, r, b = rect_of(hwnd)
    w, h = r - l, b - t
    if w <= 40 or h <= 40:
        return None, f"窗口尺寸非法 {w}x{h}"
    hdc = _u32.GetWindowDC(wt.HWND(hwnd))
    if not hdc:
        return None, "GetWindowDC 失败"
    mdc = _g32.CreateCompatibleDC(hdc)
    bmp = _g32.CreateCompatibleBitmap(hdc, w, h)
    old = _g32.SelectObject(mdc, bmp)
    bi = BITMAPINFOHEADER()
    bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bi.biWidth = w
    bi.biHeight = -h            # 负高度 = 自顶向下，省一次翻转
    bi.biPlanes = 1
    bi.biBitCount = 32
    bi.biCompression = 0        # BI_RGB
    buf = ctypes.create_string_buffer(w * h * 4)
    try:
        ok = _u32.PrintWindow(wt.HWND(hwnd), mdc, 0x2)   # PW_RENDERFULLCONTENT
        got = _g32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
        if not got:
            return None, "GetDIBits 失败"
        im = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")
        return im, ("PrintWindow ok" if ok else "PrintWindow 返回 0")
    finally:
        _g32.SelectObject(mdc, old)
        _g32.DeleteObject(bmp)
        _g32.DeleteDC(mdc)
        _u32.ReleaseDC(wt.HWND(hwnd), hdc)


def grab_via_screen_topmost(hwnd: int) -> tuple[Image.Image | None, str]:
    """兜底：把窗口临时置顶 + 抢焦点后抓屏，且**强制校验**它真的是前台窗口。

    校验不过就**直接失败**，绝不返回一张可能拍到别人桌面的图。
    """
    HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
    SWP_NOMOVE, SWP_NOSIZE = 0x0002, 0x0001
    _u32.ShowWindow(wt.HWND(hwnd), SW_RESTORE)
    _u32.SetWindowPos(wt.HWND(hwnd), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    _u32.SetForegroundWindow(wt.HWND(hwnd))
    time.sleep(0.9)             # 等重绘；不等会截到切换中的半张画面
    try:
        if int(_u32.GetForegroundWindow() or 0) != hwnd:
            return None, ("无法把目标窗口置为前台（前台进程拒绝了焦点切换）"
                          " ⇒ 拒绝抓屏，否则会拍到遮挡在它上面的其它窗口")
        l, t, r, b = rect_of(hwnd)
        return ImageGrab.grab().crop((l, t, r, b)), "screen(topmost+前台校验通过)"
    finally:
        _u32.SetWindowPos(wt.HWND(hwnd), HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)


def all_console_windows() -> list[tuple[int, str, bool, tuple[int, int, int, int]]]:
    rows: list[tuple[int, str, bool, tuple[int, int, int, int]]] = []

    def cb(hwnd, _):
        cls = ctypes.create_unicode_buffer(256)
        _u32.GetClassNameW(hwnd, cls, 256)
        if cls.value != "ConsoleWindowClass":
            return True
        n = _u32.GetWindowTextLengthW(hwnd)
        b = ctypes.create_unicode_buffer(n + 1)
        _u32.GetWindowTextW(hwnd, b, n + 1)
        r = wt.RECT()
        _u32.GetWindowRect(hwnd, ctypes.byref(r))
        rows.append((hwnd, b.value, bool(_u32.IsWindowVisible(hwnd)),
                     (r.left, r.top, r.right - r.left, r.bottom - r.top)))
        return True

    _u32.EnumWindows(WNDENUMPROC(cb), 0)
    return rows


# ---------------------------------------------------------------- launch ----
def cmd_launch(a) -> int:
    # 先清掉上一次的句柄记录，避免"读到旧句柄"这种假成功
    hwnd_file = HERE / "console_hwnd.txt"
    try:
        hwnd_file.unlink()
    except OSError:
        pass

    payload = [_python(), str(CONSOLE_RUN), "--title", a.title,
               "--cols", str(a.cols), "--rows", str(a.rows),
               "--hold", str(a.hold),
               "--hwnd-file", str(hwnd_file)]
    if a.log:
        payload += ["--log", a.log]
    if a.banner:
        payload += ["--banner", a.banner]
    payload += ["--"] + a.cmd

    before = {r[0] for r in all_console_windows()}
    p = subprocess.Popen(payload, creationflags=CREATE_NEW_CONSOLE, cwd=str(os.getcwd()))
    print(f"已启动可见控制台 pid={p.pid}，等待窗口句柄……")

    hwnd = 0
    for _ in range(60):                     # 最多等 15 秒
        time.sleep(0.25)
        # 首选：窗口自己报上来的 GetConsoleWindow()
        if hwnd_file.exists():
            try:
                cand = int(hwnd_file.read_text(encoding="utf-8").splitlines()[0])
            except Exception:              # noqa: BLE001 —— 文件刚建还没写完
                cand = 0
            if cand and _u32.IsWindow(cand):
                hwnd = cand
                break
        # 次选：按标题（ASCII 标题通常可靠）
        cand = find_window(a.title)
        if cand:
            hwnd = cand
            break
        # 兜底：新出现的控制台窗口
        new = {r[0] for r in all_console_windows()} - before
        if new:
            hwnd = sorted(new)[-1]
            break
    if not hwnd:
        print("[X] 15 秒内没等到窗口", file=sys.stderr)
        return 1
    if a.maximize:
        _u32.ShowWindow(wt.HWND(hwnd), SW_MAXIMIZE)
        time.sleep(0.6)
    save_state(a.title, hwnd)
    l, t, r, b = rect_of(hwnd)
    print(f"窗口就绪 hwnd={hwnd} rect=({l},{t},{r - l}x{b - t})")
    print(f"日志文件: {a.log or '(未设置)'}")
    print(f"句柄已记录: {STATE}（后续截图按句柄，不依赖标题）")
    return 0


# ------------------------------------------------------------------ find ----
def resolve_hwnd(title: str | None) -> int:
    """先按标题找（最精确），标题不可用/被中文搞坏时用上次记录的句柄。"""
    if title:
        h = find_window(title)
        if h:
            return h
    return load_state_hwnd()


def cmd_find(a) -> int:
    hwnd = resolve_hwnd(a.title)
    if not hwnd:
        print(f"未找到窗口（标题={a.title!r}，句柄记录={STATE}）")
        return 1
    l, t, r, b = rect_of(hwnd)
    fg = "是" if foreground_hwnd() == hwnd else "否"
    vis = "是" if _u32.IsWindowVisible(wt.HWND(hwnd)) else "否"
    print(f"hwnd={hwnd} rect=({l},{t},{r - l}x{b - t}) 可见={vis} 前台={fg}")
    return 0


# ------------------------------------------------------------------ shot ----
def cmd_shot(a) -> int:
    # ★ --hwnd 优先级最高：长驻窗口（如云端看板）的句柄会被后续 launch 覆盖掉，
    #   这时按记录找必失败。显式给句柄是唯一可靠的方式。
    #   句柄来源：vis.py list
    hwnd = int(a.hwnd) if a.hwnd else resolve_hwnd(a.title)
    if not hwnd:
        # ★ 故意直接失败，不"退回全屏"：全屏会把用户桌面上微信/QQ/浏览器
        #   一起截进取证图，属于隐私泄露；而且这种失败很容易被漏看。
        print(f"[X] 找不到目标窗口（标题={a.title!r} 句柄记录={STATE}）—— "
              f"拒绝截图", file=sys.stderr)
        return 3

    if a.delay:
        time.sleep(a.delay)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if a.full:
        # --full 是显式请求（会拍到桌面其它窗口），仅特殊场合用
        im = ImageGrab.grab()
        return _finish(im, a.out, "full-screen(显式请求)", warn_plain=False)

    # ---- 首选 PrintWindow：不依赖窗口前后台关系，被遮挡也拿到正确画面 ----
    if a.method != "screen":
        im, note = grab_via_printwindow(hwnd)
        if im is None:
            print(f"[!] PrintWindow 不可用：{note} ⇒ 回退置顶抓屏", file=sys.stderr)
        else:
            lo, hi = im.convert("L").getextrema()
            if hi - lo >= 8:
                return _finish(im, a.out, f"window#{hwnd} PrintWindow")
            print(f"[!] PrintWindow 画面近乎纯色（{note}）⇒ 回退置顶抓屏", file=sys.stderr)

    # ---- 兜底：置顶 + 抢焦点 + **前台校验**（校验不过直接失败）----
    im, note = grab_via_screen_topmost(hwnd)
    if im is None:
        print(f"[X] {note}", file=sys.stderr)
        return 4
    return _finish(im, a.out, f"window#{hwnd} {note}")


def _finish(im: "Image.Image", out: str, mode: str, warn_plain: bool = True) -> int:
    im.save(out)
    lo, hi = im.convert("L").getextrema()
    print(f"OK {out}  size={im.size}  {mode}  bytes={os.path.getsize(out)}  "
          f"gray=({lo},{hi})")
    if warn_plain and hi - lo < 8:
        print("[!] 画面近乎纯色，大概率没截到内容", file=sys.stderr)
        return 2
    return 0


# ------------------------------------------------------------------ list ----
def cmd_list(a) -> int:
    rows = all_console_windows()
    print(f"控制台窗口 {len(rows)} 个：")
    for h, t, vis, (x, y, w, hh) in rows:
        print(f"  hwnd={h:>9} vis={str(vis):<5} rect=({x},{y},{w}x{hh}) title={t[:60]!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)

    p = sub.add_parser("launch"); p.set_defaults(fn=cmd_launch)
    p.add_argument("--title", required=True)
    p.add_argument("--log", default=None)
    p.add_argument("--banner", default=None)
    p.add_argument("--cols", type=int, default=150)
    p.add_argument("--rows", type=int, default=44)
    p.add_argument("--maximize", action="store_true")
    p.add_argument("--hold", type=int, default=0,
                   help="命令结束后窗口多留这么多秒，便于截到结果那一屏")
    p.add_argument("cmd", nargs=argparse.REMAINDER)

    p = sub.add_parser("find"); p.set_defaults(fn=cmd_find)
    p.add_argument("--title", default=None)

    p = sub.add_parser("shot"); p.set_defaults(fn=cmd_shot)
    p.add_argument("out")
    p.add_argument("--hwnd", type=int, default=0,
                   help="直接指定窗口句柄（优先级最高；长驻窗口句柄会被后续 launch 覆盖，"
                        "这时必须用它。句柄来源：vis.py list）")
    p.add_argument("--title", default=None)
    p.add_argument("--full", action="store_true",
                   help="显式全屏截图（会拍到桌面其它窗口，慎用）")
    p.add_argument("--method", choices=["print", "screen"], default="print",
                   help="print=PrintWindow 抓窗口自身(默认，被遮挡也正确)；"
                        "screen=置顶抓屏(带前台校验)")
    p.add_argument("--no-focus", action="store_true",
                   help="(已废弃，保留兼容) PrintWindow 本就不抢焦点")
    p.add_argument("--delay", type=float, default=0.0)

    p = sub.add_parser("list"); p.set_defaults(fn=cmd_list)

    a = ap.parse_args()
    if getattr(a, "cmd", None):
        a.cmd = [c for c in a.cmd if c != "--"]
        if not a.cmd:
            print("launch 后面要跟一条命令", file=sys.stderr)
            return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
