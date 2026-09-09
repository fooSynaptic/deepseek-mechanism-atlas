#!/usr/bin/env python3
"""Stream a public corpus → packed GPT-2 uint16 tokens + per-token UTF-8 byte lengths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-tokens", type=int, default=3_500_000_000)
    p.add_argument("--val-tokens", type=int, default=50_000_000)
    p.add_argument("--source", default="fineweb-edu", choices=("fineweb-edu", "tinystories", "wikipedia", "local-txt", "jsonl-dir", "parquet-dir"))
    p.add_argument("--local-txt", default="", help="path to a utf-8 text file (for --source local-txt)")
    p.add_argument("--jsonl-dir", default="", help="directory of *.jsonl with a text field")
    p.add_argument("--parquet-dir", default="", help="directory of FineWeb-style *.parquet with a text column")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    try:
        import tiktoken
    except ImportError:
        raise SystemExit("tiktoken missing; pip install -r requirements.txt")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    byte_lut = np.ones(50257, dtype=np.uint8)
    for i in range(50257):
        try:
            byte_lut[i] = min(255, max(1, len(enc.decode([i]).encode("utf-8"))))
        except Exception:
            byte_lut[i] = 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_total = args.train_tokens + args.val_tokens
    tok_path = out / "all.bin"
    byt_path = out / "all.bytes.bin"
    tok = np.memmap(tok_path, dtype=np.uint16, mode="w+", shape=(n_total,))
    byt = np.memmap(byt_path, dtype=np.uint8, mode="w+", shape=(n_total,))

    written = 0
    docs = 0

    def push_ids(ids: list[int]):
        nonlocal written
        if written >= n_total:
            return
        take = min(len(ids), n_total - written)
        arr = np.asarray(ids[:take], dtype=np.int64)
        arr = np.clip(arr, 0, 50256)
        tok[written : written + take] = arr.astype(np.uint16)
        byt[written : written + take] = byte_lut[arr]
        written += take

    def push_text(text: str):
        if not text or written >= n_total:
            return
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        push_ids(ids)

    print(f"encoding {args.source} → {n_total} tokens", flush=True)
    if args.source == "local-txt":
        path = Path(args.local_txt)
        if not path.exists():
            raise SystemExit(f"missing --local-txt {path}")
        buf = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "":
                    if buf:
                        push_text("".join(buf))
                        docs += 1
                        buf = []
                        if written >= n_total:
                            break
                        if docs % 2000 == 0:
                            print(f"docs={docs} tokens={written}", flush=True)
                else:
                    buf.append(line)
            if buf and written < n_total:
                push_text("".join(buf))
                docs += 1
    elif args.source == "jsonl-dir":
        d = Path(args.jsonl_dir)
        files = sorted(d.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"no jsonl in {d}")
        for fp in files:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if written >= n_total:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    push_text(obj.get("text") or obj.get("content") or "")
                    docs += 1
                    if docs % 2000 == 0:
                        print(f"{fp.name} docs={docs} tokens={written}", flush=True)
            if written >= n_total:
                break
    elif args.source == "parquet-dir":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("pyarrow missing; pip install -r requirements.txt")
        d = Path(args.parquet_dir)
        files = sorted(d.glob("*.parquet"))
        if not files:
            raise SystemExit(f"no parquet in {d}")
        for fp in files:
            print(f"scan {fp.name}", flush=True)
            pf = pq.ParquetFile(fp)
            cols = [c for c in ("text", "content") if c in pf.schema.names]
            if not cols:
                print(f"skip {fp.name} cols={pf.schema.names}", flush=True)
                continue
            for batch in pf.iter_batches(batch_size=512, columns=cols[:1]):
                for t in batch.column(0).to_pylist():
                    if written >= n_total:
                        break
                    push_text(t or "")
                    docs += 1
                    if docs % 2000 == 0:
                        print(f"{fp.name} docs={docs} tokens={written}", flush=True)
                if written >= n_total:
                    break
            if written >= n_total:
                break
    elif args.source == "tinystories":
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for row in ds:
            push_text(row.get("text") or "")
            docs += 1
            if written >= n_total:
                break
            if docs % 10000 == 0:
                print(f"docs={docs} tokens={written}", flush=True)
    elif args.source == "wikipedia":
        from datasets import load_dataset

        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for row in ds:
            push_text(row.get("text") or "")
            docs += 1
            if written >= n_total:
                break
            if docs % 2000 == 0:
                print(f"docs={docs} tokens={written}", flush=True)
    else:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
        for row in ds:
            push_text(row.get("text") or "")
            docs += 1
            if written >= n_total:
                break
            if docs % 2000 == 0:
                print(f"docs={docs} tokens={written}", flush=True)

    tok.flush()
    byt.flush()
    del tok, byt
    if written < n_total:
        print(f"[warn] only wrote {written}/{n_total}; truncating files", flush=True)
        for path, dt in ((tok_path, np.uint16), (byt_path, np.uint8)):
            arr = np.memmap(path, dtype=dt, mode="r")
            slim = np.array(arr[:written])
            del arr
            slim.tofile(path)

    # split: last val_tokens = val (or 5% if short)
    n = written
    n_val = min(args.val_tokens, max(1, n // 20))
    n_train = n - n_val
    all_tok = np.memmap(tok_path, dtype=np.uint16, mode="r")
    all_byt = np.memmap(byt_path, dtype=np.uint8, mode="r")
    np.array(all_tok[:n_train]).tofile(out / "train.bin")
    np.array(all_byt[:n_train]).tofile(out / "train.bytes.bin")
    np.array(all_tok[n_train:]).tofile(out / "val.bin")
    np.array(all_byt[n_train:]).tofile(out / "val.bytes.bin")
    meta = {
        "source": args.source,
        "n_total_written": n,
        "n_train": n_train,
        "n_val": n_val,
        "tokenizer": "tiktoken-gpt2",
        "n_vocab": 50257,
        "docs": docs,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
