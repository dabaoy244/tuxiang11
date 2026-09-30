#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把云端 AutoDL 上的长任务状态采集成本地一份「批量状态文件」。

为什么需要它
    `long-task-monitor` 技能的 `--batch-status` 只读 watcher 读的是**本地** JSON，
    而我们的长任务（消融链路 `ablation_chain.sh` + β 对照接力 `beta_control_chain.sh`）
    跑在远端实例上。本脚本做那层转译：一次 ssh 取回全部事实，落成
    `outputs/monitor/cloud_status.json`（字段符合技能 schema：`videos` + `last_heartbeat`）。

映射约定
    videos.<arm>.status  : done / processing / failed / pending
    videos.<arm>.stages  : train / cross_gen / coverage 三个宏阶段（done|running|pending）
    videos.<arm>.title   : 人读摘要（跑着的臂会带 epoch/loss/beta/kl）
    last_heartbeat       : 取「最新证据」的最大值 = 主/β 训练日志 mtime 与两链路心跳文件
                           （主链路的 chain.heartbeat 在训练期间是陈旧的，只有轮次切换才 touch，
                             所以**必须**把训练日志 mtime 纳进来，否则会误报心跳超时）

用法
    python scripts/cloud/collect_cloud_status.py [--out-dir outputs/monitor]
退出码
    0 = 采集成功；1 = ssh 失败（网络/关机）；2 = 状态文件解析失败
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

SSH_HOST = "autodl"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]

MAIN_ARMS = [
    "full",
    "no_vib",
    "no_cross_attention",
    "no_edge",
    "no_learnable_mask",
    "no_phase",
    "no_grad_stop",
]
BETA_ARMS = ["full", "no_vib"]
BETA_SUFFIX = "_beta"

RESTART_HINT = (
    "主链路：bash /root/autodl-tmp/chain_restart.sh（幂等，勿直接重跑 ablation_chain.sh）；"
    "β 接力：bash /root/autodl-tmp/beta_control_chain.sh"
)

# ---------------------------------------------------------------------------
# 远端探针：一次 ssh 取回全部事实（用 bash -s 走 stdin，避开层层引号转义）
# ---------------------------------------------------------------------------
REMOTE_PROBE = r'''
R=/root/autodl-tmp/vibnet-forgery-detector
CURL=$(cat /root/autodl-tmp/.current_train_log 2>/dev/null)
CURB=$(cat /root/autodl-tmp/.current_beta_log 2>/dev/null)
echo "@@NOW $(date +%s)"
echo "@@HOST $(hostname)"
echo "@@CHAIN"; cat /root/autodl-tmp/chain.status 2>/dev/null
echo "@@BETA"; cat /root/autodl-tmp/beta_control.status 2>/dev/null
echo "@@ARCH"; ls -1 "$R/outputs/ablation/_archive" 2>/dev/null
echo "@@ARCHB"; ls -1 "$R/outputs/ablation_betafull/_archive" 2>/dev/null
echo "@@CROSS"; ls -1 "$R/outputs/cross_gen" 2>/dev/null
echo "@@COV"; ls -1 "$R/outputs/coverage_ext" 2>/dev/null
echo "@@CURMAIN $CURL"
echo "@@CURBETA $CURB"
echo "@@MTMAIN $(stat -c %Y "$CURL" 2>/dev/null)"
echo "@@MTBETA $(stat -c %Y "$CURB" 2>/dev/null)"
echo "@@HBMAIN $(stat -c %Y /root/autodl-tmp/chain.heartbeat 2>/dev/null)"
echo "@@HBBETA $(stat -c %Y /root/autodl-tmp/beta_control.heartbeat 2>/dev/null)"
echo "@@BAL"; cat /root/autodl-tmp/budget.tsv 2>/dev/null
echo "@@PROCS"; pgrep -af 'ablation_chain\.sh$|beta_control_chain\.sh$|nightwatch\.sh$|src\.evaluation\.(ablation|beta_control)' 2>/dev/null
echo "@@EPOCHMAIN"; grep -hoE "\[[a-z0-9_]+\] epoch [0-9]+/[0-9]+ \(global [0-9]+\) lr=[^ ]+ beta=[^ ]+ loss=[^ ]+.*" "$CURL" 2>/dev/null | tail -1
echo "@@EPOCHBETA"; grep -hoE "\[[a-z0-9_]+\] epoch [0-9]+/[0-9]+ \(global [0-9]+\) lr=[^ ]+ beta=[^ ]+ loss=[^ ]+.*" "$CURB" 2>/dev/null | tail -1
echo "@@STAGESMAIN"; grep -hE "阶段 [0-9]+/3" "$CURL" 2>/dev/null | tail -1
echo "@@STAGESBETA"; grep -hE "阶段 [0-9]+/3" "$CURB" 2>/dev/null | tail -1
echo "@@MEM"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null
echo "@@END"
'''

EPOCH_RE = re.compile(
    r"\[(?P<stage>[a-z0-9_]+)\]\s+epoch\s+(?P<ep>\d+)/(?P<ept>\d+)\s+"
    r"\(global\s+(?P<g>\d+)\)\s+lr=(?P<lr>\S+)\s+beta=(?P<beta>\S+)\s+"
    r"loss=(?P<loss>\S+).*?kl_raw=(?P<kl>\S+)\s+kl_clip=(?P<klc>\S+)\s+"
    r"grad_norm=(?P<gn>\S+)\s+\((?P<secs>[0-9.]+)s\)"
)


def _ssh(script: str) -> str:
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, SSH_HOST, "bash", "-s"],
        input=script.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    if proc.returncode != 0 and "@@END" not in out:
        raise RuntimeError(f"ssh 失败 rc={proc.returncode}: {err.strip()[:300]}")
    return out


def parse_blocks(text: str) -> dict:
    """把 @@KEY 分块输出解析成 dict；块内多行存 list。"""
    blocks: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("@@"):
            head, _, rest = line.partition(" ")
            cur = head[2:]
            blocks.setdefault(cur, [])
            if rest.strip():
                blocks[cur].append(rest.rstrip())
        elif cur is not None:
            blocks[cur].append(line.rstrip())
    return blocks


def _one(blocks: dict, key: str) -> str:
    vals = blocks.get(key) or []
    return vals[0].strip() if vals else ""


def _kv(blocks: dict, key: str) -> dict:
    d = {}
    for line in blocks.get(key) or []:
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def parse_epoch(line: str) -> dict:
    m = EPOCH_RE.search(line or "")
    if not m:
        return {}
    g = m.groupdict()
    return {
        "stage": g["stage"],
        "epoch": int(g["ep"]),
        "epoch_total": int(g["ept"]),
        "global": int(g["g"]),
        "lr": g["lr"],
        "beta": g["beta"],
        "loss": g["loss"],
        "kl_raw": g["kl"],
        "grad_norm": g["gn"],
        "secs": float(g["secs"]),
    }


def iso_local(epoch_secs) -> str:
    try:
        ts = int(float(epoch_secs))
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def stage_state(flag_done: bool, flag_running: bool) -> str:
    if flag_done:
        return "done"
    return "running" if flag_running else "pending"


def build(blocks: dict, out_dir: str = "") -> dict:
    arch = set(blocks.get("ARCH") or [])
    archb = set(blocks.get("ARCHB") or [])
    cross = set(blocks.get("CROSS") or [])
    cov = set(blocks.get("COV") or [])

    chain = _kv(blocks, "CHAIN")
    beta = _kv(blocks, "BETA")
    chain_phase = chain.get("phase", "")
    beta_phase = beta.get("phase", "")
    chain_done = set(filter(None, chain.get("done", "").split(",")))
    beta_done = set(filter(None, beta.get("done", "").split(",")))

    e_main = parse_epoch(_one(blocks, "EPOCHMAIN"))
    e_beta = parse_epoch(_one(blocks, "EPOCHBETA"))

    def arm_entry(arm: str, *, is_beta: bool) -> dict:
        name = arm + BETA_SUFFIX if is_beta else arm
        arcs = archb if is_beta else arch
        archived = any(a.startswith(arm + "_") for a in arcs)
        matched = arm in (beta_done if is_beta else chain_done)
        running = (
            (beta_phase if is_beta else chain_phase).startswith(f"train:{arm}")
            or (beta_phase if is_beta else chain_phase).startswith(f"done:{arm}")
        )
        # 注意：`_beta` 后缀只属于 β 组的产物；主表臂拼上它就会永远匹配不到（曾踩过）
        suffix = BETA_SUFFIX if is_beta else ""
        xg = f"{arm}{suffix}" in cross
        cvg = f"external_COVERAGE_{arm}{suffix}.json" in cov
        stages = {
            "train": stage_state(archived or matched, running and not archived),
            "cross_gen": stage_state(xg, running and archived and not xg),
            "coverage": stage_state(cvg, running and xg and not cvg),
        }

        labels = [f"{name}（β生效版）" if is_beta else name]
        if archived or matched:
            labels.append("已归档")
            status = "done"
        elif running:
            status = "processing"
            ep = e_beta if is_beta else e_main
            if ep:
                labels.append(
                    "{stage} {epoch}/{epoch_total}（global {global}）"
                    " loss={loss} beta={beta} kl={kl_raw} gn={grad_norm} "
                    "{secs:.0f}s/轮".format(**ep)
                )
        else:
            status = "pending"
            labels.append("待跑")
        return {
            "status": status,
            "stages": stages,
            "title": " · ".join(labels)[:220],
            # 绝对路径：monitor.py 的 --progress-glob 以自身 cwd 为基准解析
            "item_dir": os.path.join(out_dir, "artifacts", name),
        }

    videos = {}
    for arm in MAIN_ARMS:
        videos[arm] = arm_entry(arm, is_beta=False)
    for arm in BETA_ARMS:
        videos[arm + BETA_SUFFIX] = arm_entry(arm, is_beta=True)

    hb_candidates = [
        _one(blocks, "MTMAIN"),
        _one(blocks, "MTBETA"),
        _one(blocks, "HBMAIN"),
        _one(blocks, "HBBETA"),
    ]
    hb = max((int(c) for c in hb_candidates if c.isdigit()), default=0)

    bal_line = (blocks.get("BAL") or [""])[0].split()
    bal = bal_line[1] if len(bal_line) > 1 else ""

    # 进程列表去噪：同一训练命令会有多条（timeout + 主进程 + DataLoader worker），
    # 只保留去掉 PID 后的唯一命令；并丢掉启动包装用的 `bash -c ...` 长行
    seen, procs = set(), []
    for raw in blocks.get("PROCS") or []:
        if not raw.strip() or "bash -c " in raw:
            continue
        cmd = raw.split(" ", 1)[1] if raw.split(" ", 1)[0].isdigit() else raw
        if cmd in seen:
            continue
        seen.add(cmd)
        procs.append(raw.strip())
    procs = procs[:10]

    data = {
        "last_heartbeat": iso_local(hb),
        "videos": videos,
        "_context": {
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "remote_now": iso_local(_one(blocks, "NOW")),
            "host": _one(blocks, "HOST"),
            "chain_phase": chain_phase,
            "chain_done": chain.get("done", ""),
            "chain_fails": chain.get("fails", ""),
            "beta_phase": beta_phase,
            "beta_done": beta.get("done", ""),
            "budget_est_cny": chain.get("budget_est_cny", ""),
            "balance_ledger_ts": bal_line[0] if bal_line else "",
            "balance_ledger_cny": bal,
            "procs": procs,
            "cur_main_log": _one(blocks, "CURMAIN"),
            "cur_beta_log": _one(blocks, "CURBETA"),
            "epoch_main": e_main,
            "epoch_beta": e_beta,
            "stage_main": _one(blocks, "STAGESMAIN"),
            "stage_beta": _one(blocks, "STAGESBETA"),
            "gpu": _one(blocks, "MEM"),
        },
    }
    return data


def write_markers(out_dir: str, data: dict) -> int:
    """把每个「已完成」的宏阶段落一个本地标记文件（幂等，保留首次时间）。

    作用是给技能的 `--progress-glob` 提供可统计的"子阶段文件数"，
    同时留下一份不依赖云端的审计轨迹。
    """
    root = os.path.join(out_dir, "artifacts")
    written = 0
    for arm, v in data["videos"].items():
        d = os.path.join(root, arm)
        for stage, state in v["stages"].items():
            if state != "done":
                continue
            fp = os.path.join(d, f"progress_{stage}.json")
            if os.path.exists(fp):
                continue
            os.makedirs(d, exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(
                    {"arm": arm, "stage": stage, "first_seen": data["_context"]["collected_at"]},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            written += 1
    return written


def write_summary(out_dir: str, data: dict) -> str:
    ctx = data["_context"]
    lines = [
        "# 云端长任务状态（rich 视图）",
        "",
        f"- **采集时间**：{ctx['collected_at']}（远端 {ctx['remote_now']}）",
        f"- **心跳（最新证据）**：{data['last_heartbeat']}",
        f"- **主链路 phase**：`{ctx['chain_phase']}` ｜ done=`{ctx['chain_done'] or '-'}` ｜ fails={ctx['chain_fails']}",
        f"- **β 接力 phase**：`{ctx['beta_phase']}` ｜ done=`{ctx['beta_done'] or '-'}`",
        f"- **账本余额**：{ctx['balance_ledger_cny'] or '?'} 元（账本时间 {iso_local(ctx['balance_ledger_ts']) or '?'}）",
        f"- **GPU**：{ctx['gpu'] or '不可读'}",
        f"- **当前阶段**：主 `{ctx['stage_main'] or '-'}`",
        "",
        "## 各臂状态",
        "",
        "| 臂 | 状态 | train | cross_gen | coverage | 摘要 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, v in data["videos"].items():
        s = v["stages"]
        lines.append(
            f"| {arm} | {v['status']} | {s['train']} | {s['cross_gen']} | {s['coverage']} | {v['title']} |"
        )
    lines += ["", "## 进程", ""]
    lines += [f"- `{p}`" for p in ctx["procs"]] or ["- （无）"]
    lines += [
        "",
        f"- 主日志：`{ctx['cur_main_log'] or '-'}`",
        f"- β 日志：`{ctx['cur_beta_log'] or '-'}`",
        "",
        "---",
        "_由 scripts/cloud/collect_cloud_status.py 生成_",
    ]
    fp = os.path.join(out_dir, "cloud_status.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description="采集云端长任务状态 -> 本地批量状态文件")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "monitor"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        blocks = parse_blocks(_ssh(REMOTE_PROBE))
    except Exception as exc:  # noqa: BLE001
        print(f"[collect] 采集失败：{exc}", file=sys.stderr, flush=True)
        return 1
    if "END" not in blocks:
        print("[collect] 远端探针未正常收尾（缺少 @@END）", file=sys.stderr, flush=True)
        return 2

    data = build(blocks, out_dir)
    json_fp = os.path.join(out_dir, "cloud_status.json")
    with open(json_fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    written = write_markers(out_dir, data)
    md_fp = write_summary(out_dir, data)

    if not args.quiet:
        ctx = data["_context"]
        print(f"[collect] {ctx['collected_at']} 主={ctx['chain_phase']} β={ctx['beta_phase']}")
        for arm, v in data["videos"].items():
            print(f"  {arm:<20} {v['status']:<11} {v['title'][:110]}")
        print(f"[collect] 余额账本={ctx['balance_ledger_cny']} 元  GPU={ctx['gpu']}")
        print(f"[collect] 新增阶段标记 {written} 个 -> {json_fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
