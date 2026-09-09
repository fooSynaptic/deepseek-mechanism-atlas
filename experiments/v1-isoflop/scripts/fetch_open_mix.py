#!/usr/bin/env python3
"""Pull a diverse open mix from official dumps / hubs.

Sources (all public, no arbitrary-site scrape):
  - FineWeb-Edu sample-10BT parquet (Hugging Face / ModelScope mirrors)
  - Wikipedia EN / ZH (Wikimedia via datasets)
  - The Stack smol (code)
  - C4 English sample
  - Project Gutenberg (HF dump, then gutenberg.org cache)
  - Common Crawl WET (data.commoncrawl.org)
  - OpenWebMath (optional)

Does not scrape random websites.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def curl_to(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"reuse {dest} ({dest.stat().st_size} bytes)", flush=True)
        return True
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
        "--retry-all-errors", "-C", "-",
        "-o", str(dest), url,
    ]
    return run(cmd) == 0 and dest.exists() and dest.stat().st_size > 1000


def write_jsonl(path: Path, rows_iter, n_docs: int, source: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for text in rows_iter:
            if not text or len(text) < 80:
                continue
            f.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"{source} docs={written}", flush=True)
            if written >= n_docs:
                break
    return written


def parquet_texts(path: Path):
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        print("pyarrow missing:", e, flush=True)
        return
    pf = pq.ParquetFile(path)
    cols = [c for c in ("text", "content") if c in pf.schema.names]
    if not cols:
        print(f"no text column in {path}: {pf.schema.names}", flush=True)
        return
    for batch in pf.iter_batches(batch_size=512, columns=cols[:1]):
        for t in batch.column(0).to_pylist():
            if t:
                yield t


def fetch_fineweb_parquets(out_jsonl: Path, n_files: int, n_docs: int) -> int:
    """Download FineWeb-Edu sample-10BT shards (official parquet, not a crawler)."""
    names = [f"{i:03d}_00000.parquet" for i in range(n_files)]
    rels = [f"sample/10BT/{n}" for n in names]
    mirrors = [
        "https://www.modelscope.cn/datasets/HuggingFaceFW/fineweb-edu/resolve/master/{rel}",
        "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/{rel}",
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/{rel}",
    ]
    raw_dir = out_jsonl.parent.parent / "fineweb_parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for rel, name in zip(rels, names):
        dest = raw_dir / name
        ok = False
        for tmpl in mirrors:
            if curl_to(tmpl.format(rel=rel), dest):
                ok = True
                break
        if ok:
            files.append(dest)
        else:
            print(f"skip parquet {name}", flush=True)
    if not files:
        return try_modelscope_stream(out_jsonl, n_docs)
    written = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for fp in files:
            print(f"scan {fp}", flush=True)
            for text in parquet_texts(fp):
                if not text or len(text) < 80:
                    continue
                f.write(json.dumps({"text": text, "source": "fineweb-edu"}, ensure_ascii=False) + "\n")
                written += 1
                if written % 2000 == 0:
                    print(f"fineweb docs={written}", flush=True)
                if written >= n_docs:
                    return written
    if written == 0:
        return try_modelscope_stream(out_jsonl, n_docs)
    return written


def try_modelscope_stream(out_jsonl: Path, n_docs: int) -> int:
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as e:
        print("modelscope sdk missing:", e, flush=True)
        return 0
    candidates = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("AI-ModelScope/fineweb-edu", None),
        ("HuggingFaceFW/fineweb-edu", None),
    ]
    for name, subset in candidates:
        try:
            kw = {"split": "train"}
            if subset:
                kw["subset_name"] = subset
            print(f"MsDataset.load {name} {subset}", flush=True)
            ds = MsDataset.load(name, **kw)

            def texts():
                for row in ds:
                    yield row.get("text") or row.get("content") or ""

            n = write_jsonl(out_jsonl, texts(), n_docs, name)
            if n:
                return n
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
    return 0


def stream_hf(out_jsonl: Path, n_docs: int, repo: str, config=None, split="train", source="") -> int:
    try:
        from datasets import load_dataset
    except Exception as e:
        print("datasets missing:", e, flush=True)
        return 0
    try:
        kw = {"split": split, "streaming": True}
        if config:
            ds = load_dataset(repo, config, **kw)
        else:
            ds = load_dataset(repo, **kw)
    except Exception as e:
        print(f"hf stream fail {repo}: {e}", flush=True)
        return 0

    def texts():
        for row in ds:
            yield row.get("text") or row.get("content") or row.get("code") or ""

    return write_jsonl(out_jsonl, texts(), n_docs, source or repo)


def fetch_gutenberg(out_jsonl: Path, n_books: int) -> int:
    n = stream_hf(out_jsonl, n_books, "manu/project_gutenberg", source="gutenberg")
    if n:
        return n
    ids = list(range(1, n_books + 1))
    written = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i in ids:
            url = f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"
            dest = out_jsonl.parent / f"pg{i}.txt"
            if not curl_to(url, dest):
                continue
            try:
                text = dest.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            dest.unlink(missing_ok=True)
            if len(text) < 500:
                continue
            f.write(json.dumps({"text": text, "source": "gutenberg"}, ensure_ascii=False) + "\n")
            written += 1
            if written % 20 == 0:
                print(f"gutenberg books={written}", flush=True)
    return written


def fetch_cc_wet_sample(out_jsonl: Path, n_warcs: int = 4) -> int:
    raw = out_jsonl.parent.parent / "wet.paths.gz"
    if not curl_to("https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/wet.paths.gz", raw):
        print("CC wet.paths.gz unreachable", flush=True)
        return 0
    urls = []
    with gzip.open(raw, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                urls.append("https://data.commoncrawl.org/" + line)
            if len(urls) >= n_warcs:
                break
    written = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as out:
        for u in urls:
            dest = out_jsonl.parent.parent / Path(u).name
            if not curl_to(u, dest):
                continue
            try:
                with gzip.open(dest, "rt", errors="replace") as g:
                    buf = []
                    for line in g:
                        if line.startswith("WARC/"):
                            if buf:
                                text = "".join(buf)
                                if len(text) > 200:
                                    out.write(
                                        json.dumps({"text": text, "source": "cc-wet"}, ensure_ascii=False)
                                        + "\n"
                                    )
                                    written += 1
                            buf = []
                        elif line.startswith("WARC-") or line.startswith("Content-"):
                            continue
                        else:
                            buf.append(line)
            except Exception as e:
                print("wet parse", dest, e, flush=True)
            dest.unlink(missing_ok=True)
            print(f"cc-wet docs~{written} after {u}", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--preset",
        choices=("demo", "wave-a", "full"),
        default="",
        help="demo: small public sample; wave-a: ~4B-token mix; full: report-scale fetch",
    )
    ap.add_argument("--fineweb-docs", type=int, default=3_000_000)
    ap.add_argument("--fineweb-files", type=int, default=8)
    ap.add_argument("--wiki-en-docs", type=int, default=250_000)
    ap.add_argument("--wiki-zh-docs", type=int, default=120_000)
    ap.add_argument("--code-docs", type=int, default=200_000)
    ap.add_argument("--c4-docs", type=int, default=400_000)
    ap.add_argument("--math-docs", type=int, default=80_000)
    ap.add_argument("--gutenberg-books", type=int, default=400)
    ap.add_argument("--cc-warcs", type=int, default=4)
    ap.add_argument("--zh-edu-docs", type=int, default=80_000)
    args = ap.parse_args()
    if args.preset == "demo":
        args.fineweb_docs = 2_000
        args.fineweb_files = 1
        args.wiki_en_docs = 200
        args.wiki_zh_docs = 100
        args.code_docs = 0
        args.c4_docs = 0
        args.math_docs = 200
        args.gutenberg_books = 8
        args.cc_warcs = 1
        args.zh_edu_docs = 0
    elif args.preset == "wave-a":
        args.fineweb_docs = 3_000_000
        args.fineweb_files = 8
        args.wiki_en_docs = 250_000
        args.wiki_zh_docs = 120_000
        args.code_docs = 0
        args.c4_docs = 0
        args.math_docs = 80_000
        args.gutenberg_books = 375
        args.cc_warcs = 4
        args.zh_edu_docs = 80_000
    elif args.preset == "full":
        args.fineweb_docs = 8_000_000
        args.fineweb_files = 24
        args.wiki_en_docs = 500_000
        args.wiki_zh_docs = 250_000
        args.code_docs = 0
        args.c4_docs = 0
        args.math_docs = 200_000
        args.gutenberg_books = 800
        args.cc_warcs = 12
        args.zh_edu_docs = 80_000
    out = Path(args.out_dir)
    mix = out / "mix"
    mix.mkdir(parents=True, exist_ok=True)
    stats = {}
    stats["fineweb"] = fetch_fineweb_parquets(mix / "fineweb.jsonl", args.fineweb_files, args.fineweb_docs)
    stats["wiki_en"] = stream_hf(
        mix / "wiki_en.jsonl", args.wiki_en_docs, "wikimedia/wikipedia", "20231101.en", source="wikipedia-en"
    )
    stats["wiki_zh"] = stream_hf(
        mix / "wiki_zh.jsonl", args.wiki_zh_docs, "wikimedia/wikipedia", "20231101.zh", source="wikipedia-zh"
    )
    stats["code"] = (
        stream_hf(mix / "stack.jsonl", args.code_docs, "bigcode/the-stack-smol", source="the-stack-smol")
        if args.code_docs
        else 0
    )
    stats["c4"] = stream_hf(mix / "c4.jsonl", args.c4_docs, "allenai/c4", "en", source="c4-en") if args.c4_docs else 0
    stats["math"] = stream_hf(
        mix / "openwebmath.jsonl", args.math_docs, "open-web-math/open-web-math", source="open-web-math"
    )
    stats["zh_edu"] = (
        stream_hf(
            mix / "zh_edu.jsonl", args.zh_edu_docs, "opencsg/Fineweb-Edu-Chinese-V2.1", source="zh-fineweb-edu"
        )
        if args.zh_edu_docs
        else 0
    )
    stats["gutenberg"] = fetch_gutenberg(mix / "gutenberg.jsonl", args.gutenberg_books)
    stats["cc_wet"] = fetch_cc_wet_sample(mix / "cc_wet.jsonl", args.cc_warcs)
    (out / "mix_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    if sum(stats.values()) == 0:
        raise SystemExit("no corpus source succeeded")


if __name__ == "__main__":
    main()
