#!/usr/bin/env python3
"""Local job status: result.json / pid / last log line. No SSH."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def last_line(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace").strip().splitlines()
    return text[-1][:n] if text else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="")
    args = ap.parse_args()
    root = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))
    results = Path(os.environ.get("DS_RESULTS", root / "results"))
    logs = Path(os.environ.get("DS_LOGS", root / "logs"))
    grid_s = args.grid or os.environ.get("DS_GRID", str(root / "configs" / "grid.json"))
    grid_path = Path(grid_s)
    if not grid_path.is_absolute():
        grid_path = root / grid_path

    ids: list[str] = []
    if grid_path.exists():
        ids = [j["job_id"] for j in json.loads(grid_path.read_text()).get("jobs") or []]
    extra = sorted(p.name for p in results.glob("iso_*") if p.is_dir())
    for jid in extra:
        if jid not in ids:
            ids.append(jid)

    print(f"{'job':<22} {'pid':>8} {'state':<10} last")
    for jid in ids:
        out = results / jid
        res = out / "result.json"
        pid_f = out / "pid"
        log_f = logs / f"{jid}.log"
        pid = pid_f.read_text().strip() if pid_f.exists() else "-"
        if res.exists():
            blob = json.loads(res.read_text())
            if blob.get("abandoned"):
                state = "abandoned"
            elif blob.get("truncated"):
                state = "trunc"
            elif blob.get("ok"):
                state = "done"
            else:
                state = "failed"
        elif pid != "-" and pid.isdigit() and pid_alive(int(pid)):
            state = "run"
        elif pid_f.exists():
            state = "stale"
        else:
            state = "wait"
        print(f"{jid:<22} {pid:>8} {state:<10} {last_line(log_f)}")


if __name__ == "__main__":
    main()
