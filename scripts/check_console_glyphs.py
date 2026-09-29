"""在当前中文控制台字体下，逐个渲染候选符号，找出**缺字形**的那些。

为什么需要它（2026-09-29 实测，本机 Windows + 中文控制台字体）：
  训练日志里的 `✔`(U+2714) / `✗`(U+2717) / `❌`(U+274C) 会渲染成方框 `□`，
  而截图是要交给老师/写进材料的**取证**，方框很难看且显得不专业。
  同类的 `✓`(U+2713)、`⚠`(U+26A0，**不带**变体选择符)、`★ ● → ≥ ≈ ± × μ β`
  则渲染正常 —— 所以不能"看着像装饰就一律换掉"，要按码位逐个确认。

用法：
  1) 起一个可见控制台窗口跑它（别在无窗口的管道里跑，管道里看不出字形）：
     outputs/screen/vis.py launch --title "GLYPH" -- "$PY_WIN" scripts/check_console_glyphs.py
  2) 再用 PrintWindow 截图，人工判读哪一行的符号变成了方框。

判读口径：某行第二个符号位显示为 `□` / `?` / 斜杠 = 该码位在本字体无字形。
实测结论（本机，可复现）：
  缺字形 -> U+2714 ✔   U+2717 ✗   U+274C ❌   U+26A0 U+FE0F ⚠️(带变体选择符)
  正常   -> U+2713 ✓   U+26A0 ⚠   U+2605 ★   U+2606 ☆   U+25CF ●   U+25CB ○
            U+2192 →   U+2265 ≥   U+2264 ≤   U+2248 ≈   U+2026 …
            U+00B1 ±   U+00D7 ×   U+03BC μ   U+03B2 β   U+2500 ─   U+2588 █
"""
import ctypes, ctypes.wintypes as wt, sys, time
_k32 = ctypes.windll.kernel32
h = _k32.GetStdHandle(-11)
m = wt.DWORD()
_k32.GetConsoleMode(h, ctypes.byref(m))
_k32.SetConsoleMode(h, m.value | 0x0004)   # ENABLE_VIRTUAL_TERMINAL_PROCESSING
out = sys.stdout.buffer
def w(s):
    out.write(s.encode("utf-8", "replace")); out.flush()

_k32.SetConsoleTitleW("VIB-Net GLYPH TEST")
w("符号字形测试（每个符号后跟 OK 两字，若显示为 �%/方框/问号即缺字形）\n")
w("-" * 90 + "\n")
tests = [
    ("U+2714 ✔", "\u2714"), ("U+2717 ✗", "\u2717"), ("U+2713 ✓", "\u2713"),
    ("U+26A0 ⚠", "\u26a0"), ("U+26A0+FE0F ⚠️", "\u26a0\ufe0f"),
    ("U+2605 ★", "\u2605"), ("U+2606 ☆", "\u2606"),
    ("U+25CF ●", "\u25cf"), ("U+25CB ○", "\u25cb"),
    ("U+2192 →", "\u2192"), ("U+21D2 ⇒", "\u21d2"),
    ("U+2265 ≥", "\u2265"), ("U+2264 ≤", "\u2264"), ("U+2248 ≈", "\u2248"),
    ("U+2026 …", "\u2026"), ("U+00B1 ±", "\u00b1"), ("U+00D7 ×", "\u00d7"),
    ("U+03BC μ", "\u03bc"), ("U+03B2 β", "\u03b2"), ("U+03C3 σ", "\u03c3"),
    ("U+2500 ─", "\u2500"), ("U+2588 █", "\u2588"), ("U+2591 ░", "\u2591"),
    ("U+274C ❌", "\u274c"), ("U+FF21 Ａ(全角)", "\uff21"),
]
for name, ch in tests:
    w(f"  {name:<18} ->  {ch}  <这里应该是上面那个符号>\n")
w("-" * 90 + "\n")
w("判读：把光标移到符号上看到的方块/问号/斜杠 = 该码位在当前字体无字形。\n")
w("（本窗口 90 秒后自动关闭）\n")
time.sleep(120)
