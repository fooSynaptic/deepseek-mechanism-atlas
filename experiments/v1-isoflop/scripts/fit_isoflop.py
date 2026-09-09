#!/usr/bin/env python3
"""Fit IsoFLOP U-curves and a descriptive M_opt(C) slope. Writes SVG + JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_iso(results: Path) -> list[dict]:
    rows = []
    for p in sorted(results.glob("iso_*/result.json")):
        d = json.loads(p.read_text())
        meta = d.get("meta") or {}
        arch = meta.get("arch") or {}
        ev = d.get("final_eval") or {}
        rows.append(
            {
                "job_id": d.get("job_id", p.parent.name),
                "c_flops": meta.get("c_flops"),
                "M": arch.get("M"),
                "six_N1": arch.get("six_N1"),
                "D_tokens": meta.get("D_tokens"),
                "val_bpb": (ev or {}).get("val_bpb", d.get("best_val_bpb")),
                "best_val_bpb": d.get("best_val_bpb"),
                "arch_id": arch.get("arch_id"),
            }
        )
    return [r for r in rows if r["val_bpb"] is not None and r["M"]]


def load_f1(results: Path) -> list[dict]:
    rows = []
    for name in ("f1_opt", "f1_Bhalf", "f1_B2", "f1_etahalf"):
        p = results / name / "result.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ev = d.get("final_eval") or {}
        meta = d.get("meta") or {}
        rows.append(
            {
                "job_id": name,
                "batch_tokens": meta.get("batch_tokens_target"),
                "max_lr": meta.get("max_lr"),
                "val_bpb": (ev or {}).get("val_bpb", d.get("best_val_bpb")),
            }
        )
    return rows


def argmin(rows, key="val_bpb"):
    return min(rows, key=lambda r: r[key])


def svg_scatter(points, title, xlabel, ylabel, out: Path):
    w, h, pad = 720, 420, 60
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin * 1.1 + 1
    if ymax == ymin:
        ymax = ymin * 1.1 + 1e-6

    def xx(x):
        return pad + (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin) + 1e-12) * (w - 2 * pad)

    def yy(y):
        return h - pad - (y - ymin) / (ymax - ymin) * (h - 2 * pad)

    dots = []
    for x, y, lab in points:
        dots.append(
            f'<circle cx="{xx(x):.1f}" cy="{yy(y):.1f}" r="5" fill="#2563eb"/>'
            f'<text x="{xx(x)+8:.1f}" y="{yy(y)+4:.1f}" font-size="11" font-family="sans-serif">{lab}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{w/2}" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>
  <text x="{w/2}" y="{h-12}" text-anchor="middle" font-size="12" font-family="sans-serif">{xlabel}</text>
  <text x="16" y="{h/2}" font-size="12" font-family="sans-serif" transform="rotate(-90 16 {h/2})">{ylabel}</text>
  {''.join(dots)}
</svg>
'''
    out.write_text(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    results = Path(args.results_dir)
    iso = load_iso(results)
    f1 = load_f1(results)
    by_c: dict[float, list] = {}
    for r in iso:
        by_c.setdefault(float(r["c_flops"]), []).append(r)

    summary = {"n_iso": len(iso), "n_f1": len(f1), "per_c": {}, "formula1": f1}
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mopts = []
    for c, rows in sorted(by_c.items()):
        best = argmin(rows)
        summary["per_c"][f"{c:.0e}"] = {
            "n": len(rows),
            "best_job": best["job_id"],
            "M_opt_emp": best["M"],
            "D_opt_emp": best["D_tokens"],
            "val_bpb": best["val_bpb"],
            "points": rows,
        }
        mopts.append((c, best["M"]))
        svg_scatter(
            [(r["M"], r["val_bpb"], r["arch_id"]) for r in rows],
            title=f"IsoFLOP C={c:.0e}  val BPB vs M",
            xlabel="M (non-emb FLOPs/token, log)",
            ylabel="val bits-per-byte",
            out=fig_dir / f"isoflop_{c:.0e}.svg".replace("+", ""),
        )
        svg_scatter(
            [(r["six_N1"], r["val_bpb"], r["arch_id"]) for r in rows],
            title=f"Same runs vs 6N1  C={c:.0e}",
            xlabel="6N1 (log)",
            ylabel="val bits-per-byte",
            out=fig_dir / f"sixN1_{c:.0e}.svg".replace("+", ""),
        )

    if len(mopts) >= 2:
        # log M = a log C + const  using first/last only when n=2
        (c0, m0), (c1, m1) = mopts[0], mopts[-1]
        a = (math.log(m1) - math.log(m0)) / (math.log(c1) - math.log(c0))
        summary["fit_a_two_point"] = a
        summary["fit_b_two_point"] = 1.0 - a
        summary["paper_a"] = 0.5243
        summary["paper_b"] = 0.4757
        summary["note"] = "two-point slope is illustrative; paper used 8 budgets"

    if f1:
        best = argmin(f1)
        opt = next((x for x in f1 if x["job_id"] == "f1_opt"), None)
        summary["formula1_best"] = best["job_id"]
        if opt and opt["val_bpb"]:
            rel = abs(opt["val_bpb"] - best["val_bpb"]) / best["val_bpb"]
            summary["formula1_opt_rel_gap"] = rel
            summary["formula1_near_opt_0p25pct"] = rel <= 0.0025

    out = results / "fit_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "per_c"}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
