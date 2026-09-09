#!/usr/bin/env python3
"""Download FineWeb-Edu sample/100BT parquet shards not in the 10BT 14-file set.

Skips filenames 000_00000.parquet … 013_00000.parquet (the 10BT sample).
Does not scrape; official parquet via ModelScope then hf-mirror / huggingface.co.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SKIP_10BT = {f"{i:03d}_00000.parquet" for i in range(14)}

TREE_URLS = [
    "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT",
    "https://hf-mirror.com/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT",
]

MIRRORS = [
    "https://www.modelscope.cn/datasets/HuggingFaceFW/fineweb-edu/resolve/master/sample/100BT/{name}",
    "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/{name}",
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/{name}",
]


def http_json(url: str) -> list:
    cmd = ["curl", "-sL", "--fail", "--max-time", "90", "-A", "ds-v1-isoflop", url]
    raw = subprocess.check_output(cmd)
    return json.loads(raw.decode())


def list_100bt() -> list[dict]:
    last = None
    for url in TREE_URLS:
        try:
            data = http_json(url)
        except Exception as e:
            last = e
            print(f"tree fail {url}: {e}", flush=True)
            continue
        if isinstance(data, list) and data:
            return data
        print(f"tree empty {url}: {data!r}"[:200], flush=True)
    raise SystemExit(f"could not list sample/100BT ({last})")


def curl_to(url: str, dest: Path, expected: int | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"reuse {dest.name} ({expected} bytes)", flush=True)
        return True
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
        "--retry-all-errors", "-C", "-",
        "-o", str(dest), url,
    ]
    print("+", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0 or not dest.exists():
        return False
    sz = dest.stat().st_size
    if expected and sz != expected:
        print(f"size mismatch {dest.name} got={sz} expected={expected}", flush=True)
        return False
    return sz > 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="directory for parquet files")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all remaining shards")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    listing = list_100bt()
    keep = []
    skipped = []
    for item in listing:
        if item.get("type") != "file":
            continue
        name = Path(item["path"]).name
        size = int(item.get("size") or 0)
        if name in SKIP_10BT:
            skipped.append(name)
            continue
        keep.append({"name": name, "size": size, "path": item["path"]})

    (out / "listing.json").write_text(
        json.dumps({"skipped_10bt": skipped, "keep": keep}, indent=2) + "\n"
    )
    print(f"100BT files={len(listing)} skip_10bt={len(skipped)} keep={len(keep)}", flush=True)
    if args.max_files:
        keep = keep[: args.max_files]
        print(f"capped to {len(keep)} files", flush=True)

    ok, fail = [], []
    for i, rec in enumerate(keep, 1):
        dest = out / rec["name"]
        print(f"[{i}/{len(keep)}] {rec['name']} {rec['size']}", flush=True)
        done = False
        for tmpl in MIRRORS:
            if curl_to(tmpl.format(name=rec["name"]), dest, rec["size"] or None):
                done = True
                break
            time.sleep(2)
        if done:
            ok.append(rec["name"])
        else:
            fail.append(rec["name"])
            dest.unlink(missing_ok=True)
            print(f"FAIL {rec['name']}", flush=True)
        (out / "progress.json").write_text(
            json.dumps({"ok": ok, "fail": fail, "n_ok": len(ok), "n_fail": len(fail)}, indent=2)
            + "\n"
        )

    print(json.dumps({"n_ok": len(ok), "n_fail": len(fail), "fail": fail}, indent=2), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
