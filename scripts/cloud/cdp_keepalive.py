"""拉起 Edge 的 CDP 调试端口并保活，供 AutoDL 控制台远程开机使用。

背景：实例若因余额耗尽被 AutoDL 自动关机，需要浏览器点「开机」；
本脚本让 Edge 常驻并开放 127.0.0.1:9223 的 DevTools 端口。

用法（后台跑）：
    python cdp_keepalive.py [保活小时数]
关键点：
- 必须用**独立** user-data-dir，否则已有 Edge 实例会吞掉新启动、调试端口不生效。
- ★ 判活**只能用 netstat**：本机沙箱会拦截回环 TCP，connect() 能成功却读不到响应
  （urllib 则会拿到 HTTP 502），socket 探活会产生"假死"，进而每轮重拉一次 Edge。
"""

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


def _probe(timeout=3):
    """用 netstat 判断 9223 是否处于 LISTENING（沙箱下 socket 探活不可信）。"""
    try:
        out = subprocess.run(
            ["netstat", "-ano"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 5,
        )
        text = out.stdout.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return f"__ERR__ netstat {exc}"
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
    while time.time() < deadline:
        time.sleep(180)
        got = _probe()
        if got.startswith("__ERR__"):
            print(f"[cdp] 端口失联，尝试重拉：{got}", flush=True)
            ensure_edge()
    print("[cdp] keepalive 到期退出（Edge 若以 DETACHED 方式启动仍会常驻）", flush=True)


if __name__ == "__main__":
    main()
