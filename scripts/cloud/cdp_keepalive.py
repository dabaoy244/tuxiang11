"""拉起 Edge 的 CDP 调试端口并保活，供 AutoDL 控制台远程开机使用。

背景：实例若因余额耗尽被 AutoDL 自动关机，需要浏览器点「开机」；
本脚本让 Edge 常驻并开放 127.0.0.1:9223 的 DevTools 端口。

用法（后台跑）：
    python cdp_keepalive.py [保活小时数]
关键点：
- 必须用**独立** user-data-dir，否则已有 Edge 实例会吞掉新启动、调试端口不生效。
- ★ 判活**不能起子进程**：socket/urllib 探回环 TCP 在沙箱下不可信（connect 假通、
  urllib 拿 502）；而 `netstat` 又会被沙箱直接拒绝（WinError 5）⇒ 假失联 → 重拉
  Edge → 再被拒 → **整个保活脚本崩溃**（10-01 15:32 / 19:45 两次实测）。
  ⇒ 改为 **ctypes 直读 Windows TCP 表**（`GetExtendedTcpTable`，进程内调用、
  不执行任何外部程序）⇒ 沙箱拦不住，netstat 仅作兜底。
"""

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
PROFILE = r"C:\Users\Administrator\.workbuddy-cdp-profile3"
PORT = 9223


def _listeners_via_ctypes(port):
    """★ 进程内直读 Windows TCP 监听表 ⇒ 不启动子进程 ⇒ 沙箱拦不住。"""
    AF_INET = 2
    TCP_TABLE_OWNER_PID_LISTENER = 3

    class MIB_TCPROW_OWNER_PID(ctypes.Structure):
        _fields_ = [
            ("dwState", wt.DWORD),
            ("dwLocalAddr", wt.DWORD),
            ("dwLocalPort", wt.DWORD),
            ("dwRemoteAddr", wt.DWORD),
            ("dwRemotePort", wt.DWORD),
            ("dwOwningPid", wt.DWORD),
        ]

    iphlpapi = ctypes.windll.iphlpapi
    size = wt.DWORD(0)
    rc = iphlpapi.GetExtendedTcpTable(
        None, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0
    )
    if rc != 0 and size.value == 0:
        raise OSError(f"GetExtendedTcpTable(size) rc={rc}")
    buf = ctypes.create_string_buffer(size.value)
    rc = iphlpapi.GetExtendedTcpTable(
        buf, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0
    )
    if rc != 0:
        raise OSError(f"GetExtendedTcpTable rc={rc}")
    num = ctypes.cast(buf, ctypes.POINTER(wt.DWORD)).contents.value
    rows = ctypes.cast(
        ctypes.addressof(buf) + ctypes.sizeof(wt.DWORD),
        ctypes.POINTER(MIB_TCPROW_OWNER_PID),
    )
    pids = []
    for i in range(num):
        r = rows[i]
        raw = r.dwLocalPort & 0xFFFF          # 网络字节序
        if ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF) == port:
            pids.append(int(r.dwOwningPid))
    return pids


def _probe(timeout=3):
    """判断 9223 是否 LISTENING。首选 ctypes（沙箱安全），netstat 仅兜底。"""
    try:
        pids = _listeners_via_ctypes(PORT)
        return f"LISTENING pid={pids[0]}" if pids else "__ERR__ 未监听"
    except Exception as exc:  # noqa: BLE001
        ctypes_err = exc
    try:
        out = subprocess.run(
            ["netstat", "-ano"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 5,
        )
        text = out.stdout.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return f"__ERR__ 探活失败 ctypes={ctypes_err} netstat={exc}"
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() == "TCP" and parts[-2].upper() == "LISTENING":
            if parts[1].endswith(f":{PORT}"):
                return f"LISTENING pid={parts[-1]}"
    return "__ERR__ 未监听"


def ensure_edge():
    got = _probe()
    if not got.startswith("__ERR__"):
        print(f"[cdp] 端口已在监听，复用现有 Edge（{got}）", flush=True)
        return True
    edge = next((p for p in EDGE_CANDIDATES if os.path.exists(p)), None)
    if edge is None:
        print("[cdp] 找不到 msedge.exe", flush=True)
        return False
    try:
        subprocess.Popen(
            [
                edge,
                f"--remote-debugging-port={PORT}",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--no-first-run",
                "--no-default-browser-check",
                "--new-window",
                "about:blank",
            ],
            creationflags=0x00000008,
            close_fds=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[cdp] 拉起 Edge 失败（沙箱拒绝？）：{exc}", flush=True)
        print("[cdp] SANDBOX_BLOCKED 需人工非沙箱重拉 Edge", flush=True)
        return False
    for i in range(30):
        time.sleep(2)
        got = _probe()
        if not got.startswith("__ERR__"):
            print(f"[cdp] 就绪（第 {i + 1} 次探测）：{got}", flush=True)
            return True
    print(f"[cdp] 启动后仍不可达：{got}", flush=True)
    return False


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    ensure_edge()
    deadline = time.time() + hours * 3600
    misses = 0
    while time.time() < deadline:
        time.sleep(180)
        got = _probe()
        if got.startswith("__ERR__"):
            misses += 1
            print(f"[cdp] 端口失联（第 {misses} 次）：{got}", flush=True)
            if ensure_edge():
                misses = 0
        else:
            misses = 0
    print("[cdp] keepalive 到期退出（Edge 若以 DETACHED 方式启动仍会常驻）", flush=True)


if __name__ == "__main__":
    main()
