#!/usr/bin/env python3
"""Harvest result.json / train.jsonl into a CSV + JSON table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_live(results: Path) -> list[dict]:
    rows = []
    for p in sorted(results.glob("iso_*/result.json")):
        d = json.loads(p.read_text())
        meta = d.get("meta") or {}
        arch = meta.get("arch") or {}
        ev = d.get("final_eval") or {}
        jsonl = p.parent / "train.jsonl"
        n_log = 0
        last_tok_s = None
        if jsonl.exists():
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_log += 1
                if "tok_s" in rec:
                    last_tok_s = rec["tok_s"]
        rows.append(
            {
                "job_id": d.get("job_id", p.parent.name),
                "ok": d.get("ok"),
                "abandoned": d.get("abandoned", False),
                "arch_id": arch.get("arch_id") or d.get("arch_id"),
                "c_flops": meta.get("c_flops"),
                "M": arch.get("M"),
                "N1": arch.get("N1"),
                "D_tokens": meta.get("D_tokens"),
                "unique_train_tokens": meta.get("unique_train_tokens"),
                "batch_tokens_actual": meta.get("batch_tokens_actual"),
                "world_size": meta.get("world_size"),
                "steps": d.get("steps"),
                "tokens_seen": d.get("tokens_seen"),
                "wall_s": d.get("wall_s"),
                "best_val_bpb": d.get("best_val_bpb", (ev or {}).get("val_bpb")),
                "final_train_loss": d.get("final_train_loss"),
                "n_log_lines": n_log,
                "last_tok_s": last_tok_s,
                "source": "live",
                "path": str(p),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="")
    ap.add_argument("--published", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = Path(args.results_dir or root / "results")
    published = Path(args.published or root / "artifacts" / "published_grid.json")
    out_dir = Path(args.out_dir or root / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    live = load_live(results) if results.exists() else []
    pub_jobs = []
    if published.exists():
        pub_jobs = json.loads(published.read_text()).get("jobs") or []
        for j in pub_jobs:
            j = dict(j)
            j["source"] = "published"
            live_ids = {r["job_id"] for r in live}
            if j["job_id"] not in live_ids:
                live.append(j)

    table = out_dir / "collected_jobs.json"
    table.write_text(json.dumps({"n": len(live), "jobs": live}, indent=2, default=str) + "\n")
    csv_path = out_dir / "collected_jobs.csv"
    cols = [
        "job_id",
        "arch_id",
        "wave",
        "c_flops",
        "M",
        "D_tokens",
        "best_val_bpb",
        "ok",
        "abandoned",
        "truncated",
        "unique_train_tokens",
        "wall_s",
        "source",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in live:
            w.writerow(row)
    print(f"n={len(live)} json={table} csv={csv_path}")
    live_n = sum(1 for r in live if r.get("source") == "live")
    pub_n = sum(1 for r in live if r.get("source") == "published")
    print(f"live={live_n} published_only={pub_n}")


if __name__ == "__main__":
    main()
