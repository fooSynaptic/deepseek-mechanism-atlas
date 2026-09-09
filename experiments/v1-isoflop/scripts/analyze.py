#!/usr/bin/env python3
"""Formula 4 readout: IsoFLOP family + log-log OLS on valley proxies.

Default inputs: artifacts/published_grid.json (no training required).
Live results/*/result.json override published rows with the same job_id.

Valley proxies match docs/EXPERIMENT_REPORT.md §5:
  C=3e18 → iso_c3e18_u3, C=1e19 → iso_c1e19_t1, C=3e19 → iso_c3e19_t2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PAPER_A = 0.5243
PAPER_B = 0.4757
PAPER_K_M = 0.1715
PAPER_K_D = 5.8316

# Report §5 proxies (closest on-grid width to paper M_opt at each C).
DEFAULT_VALLEY = {
    3e18: "iso_c3e18_u3",
    1e19: "iso_c1e19_t1",
    3e19: "iso_c3e19_t2",
}

C_COLORS = {
    3e18: "#1d4ed8",
    1e19: "#c2410c",
    3e19: "#15803d",
}


def _c_key(c: float) -> float:
    for k in (3e18, 1e19, 3e19, 1e17, 3e17):
        if abs(float(c) - k) / k < 1e-6:
            return k
    return float(c)


def load_published(path: Path) -> list[dict]:
    if not path.exists():
        return []
    blob = json.loads(path.read_text())
    jobs = blob.get("jobs") or []
    for j in jobs:
        j.setdefault("source", "published")
    return jobs


def load_live(results: Path) -> list[dict]:
    rows = []
    if not results.exists():
        return rows
    for p in sorted(results.glob("iso_*/result.json")):
        d = json.loads(p.read_text())
        meta = d.get("meta") or {}
        arch = meta.get("arch") or {}
        ev = d.get("final_eval") or {}
        rows.append(
            {
                "job_id": d.get("job_id", p.parent.name),
                "arch_id": arch.get("arch_id") or d.get("arch_id"),
                "c_flops": meta.get("c_flops"),
                "M": arch.get("M"),
                "D_tokens": meta.get("D_tokens"),
                "best_val_bpb": d.get("best_val_bpb", (ev or {}).get("val_bpb")),
                "ok": d.get("ok"),
                "abandoned": d.get("abandoned", False),
                "truncated": d.get("truncated", False),
                "unique_train_tokens": meta.get("unique_train_tokens"),
                "source": "live",
            }
        )
    return rows


def merge_jobs(published: list[dict], live: list[dict]) -> list[dict]:
    by_id = {j["job_id"]: j for j in published}
    for j in live:
        by_id[j["job_id"]] = j
    return list(by_id.values())


def usable(job: dict) -> bool:
    if job.get("abandoned") or job.get("truncated"):
        return False
    if job.get("ok") is False:
        return False
    if job.get("best_val_bpb") is None or job.get("M") is None:
        return False
    if job.get("c_flops") is None:
        return False
    return True


def ols_loglog(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit log y = a log x + c. Returns (a, k) with y = k * x^a."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    var = sum((x - mx) ** 2 for x in lx)
    if var <= 0:
        raise SystemExit("OLS variance is zero")
    cov = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    a = cov / var
    intercept = my - a * mx
    return a, math.exp(intercept)


def two_point_a(c0: float, m0: float, c1: float, m1: float) -> float:
    return (math.log(m1) - math.log(m0)) / (math.log(c1) - math.log(c0))


def paper_m_opt(c: float) -> float:
    return PAPER_K_M * (c**PAPER_A)


def fmt_c(c: float) -> str:
    return f"{c:.0e}".replace("+0", "").replace("+", "")


def write_family_svg(jobs: list[dict], valley: dict[float, dict], out: Path) -> None:
    pts = [j for j in jobs if usable(j)]
    if not pts:
        return
    w, h = 960, 560
    left, right, top, bottom = 72, 36, 72, 56
    xs = [math.log10(float(j["M"])) for j in pts]
    ys = [float(j["best_val_bpb"]) for j in pts]
    xmin, xmax = min(xs) - 0.08, max(xs) + 0.08
    ymin, ymax = min(ys) - 0.08, max(ys) + 0.12

    def xx(m: float) -> float:
        return left + (math.log10(m) - xmin) / (xmax - xmin) * (w - left - right)

    def yy(bpb: float) -> float:
        return top + (ymax - bpb) / (ymax - ymin) * (h - top - bottom)

    by_c: dict[float, list[dict]] = {}
    for j in pts:
        by_c.setdefault(_c_key(j["c_flops"]), []).append(j)

    polylines = []
    dots = []
    for c, rows in sorted(by_c.items()):
        rows = sorted(rows, key=lambda r: float(r["M"]))
        color = C_COLORS.get(c, "#334155")
        d = " ".join(f"{xx(float(r['M'])):.1f},{yy(float(r['best_val_bpb'])):.1f}" for r in rows)
        polylines.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"/>'
        )
        for r in rows:
            m, bpb = float(r["M"]), float(r["best_val_bpb"])
            is_valley = valley.get(c, {}).get("job_id") == r["job_id"]
            rdot = 6 if is_valley else 4
            stroke = "#0f172a" if is_valley else color
            fill = "#fbbf24" if is_valley else color
            dots.append(
                f'<circle cx="{xx(m):.1f}" cy="{yy(bpb):.1f}" r="{rdot}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            )
            lab = r.get("arch_id") or ""
            if is_valley or lab in {"u0", "u3", "t0", "t1", "t2", "s1", "s2"}:
                dots.append(
                    f'<text x="{xx(m)+7:.1f}" y="{yy(bpb)-7:.1f}" font-size="10" '
                    f'fill="#334155">{lab}</text>'
                )

    ticks = []
    for exp in range(math.floor(xmin), math.ceil(xmax) + 1):
        x = xx(10**exp)
        ticks.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}" '
            f'stroke="#e2e8f0"/>'
            f'<text x="{x:.1f}" y="{h - bottom + 18}" text-anchor="middle" '
            f'font-size="11" fill="#64748b">10^{exp}</text>'
        )
    y_ticks = []
    y0 = math.floor(ymin * 10) / 10
    yv = y0
    while yv <= ymax + 1e-9:
        y = yy(yv)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#64748b">{yv:.1f}</text>'
        )
        yv = round(yv + 0.2, 10)

    legend = []
    lx0, ly0 = w - 210, 88
    for i, c in enumerate(sorted(by_c)):
        color = C_COLORS.get(c, "#334155")
        legend.append(
            f'<line x1="{lx0}" y1="{ly0 + i * 18}" x2="{lx0 + 22}" y2="{ly0 + i * 18}" '
            f'stroke="{color}" stroke-width="3"/>'
            f'<text x="{lx0 + 28}" y="{ly0 + i * 18 + 4}" font-size="12" fill="#334155">'
            f"C = {fmt_c(c)}</text>"
        )
    legend.append(
        f'<circle cx="{lx0 + 8}" cy="{ly0 + len(by_c) * 18 + 4}" r="6" '
        f'fill="#fbbf24" stroke="#0f172a"/>'
        f'<text x="{lx0 + 28}" y="{ly0 + len(by_c) * 18 + 8}" font-size="12" fill="#334155">'
        f"valley proxy</text>"
    )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"
     font-family="Helvetica Neue,Arial,sans-serif">
  <rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="40" y="32" font-size="18" font-weight="700" fill="#1a1a2e">IsoFLOP family (val BPB vs M)</text>
  <text x="40" y="52" font-size="12" fill="#64748b">Skipped abandoned / truncated jobs. Gold markers are Formula 4 valley proxies.</text>
  {''.join(ticks)}
  {''.join(y_ticks)}
  {''.join(polylines)}
  {''.join(dots)}
  {''.join(legend)}
  <text x="{(left + w - right) / 2:.1f}" y="{h - 12}" text-anchor="middle" font-size="12" fill="#334155">M (non-embedding FLOPs / token, log10)</text>
  <text x="18" y="{h / 2:.1f}" font-size="12" fill="#334155" transform="rotate(-90 18 {h / 2:.1f})">held-out bits-per-byte</text>
</svg>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", default="")
    ap.add_argument("--results-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--valley",
        default="3e18=iso_c3e18_u3,1e19=iso_c1e19_t1,3e19=iso_c3e19_t2",
        help="C=job_id pairs used for the Formula 4 OLS",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    published_path = Path(args.published or root / "artifacts" / "published_grid.json")
    results = Path(args.results_dir or root / "results")
    out_dir = Path(args.out_dir or root / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = merge_jobs(load_published(published_path), load_live(results))
    kept = [j for j in jobs if usable(j)]
    skipped = [j["job_id"] for j in jobs if j["job_id"] not in {k["job_id"] for k in kept}]

    valley_spec: dict[float, str] = {}
    for part in args.valley.split(","):
        c_s, jid = part.split("=", 1)
        valley_spec[float(c_s)] = jid.strip()

    by_id = {j["job_id"]: j for j in jobs}
    valley_rows: dict[float, dict] = {}
    missing = []
    for c, jid in valley_spec.items():
        job = by_id.get(jid)
        if not job or not usable(job):
            missing.append(jid)
            continue
        valley_rows[_c_key(c)] = job
    if len(valley_rows) < 2:
        raise SystemExit(f"need ≥2 usable valley proxies; missing={missing}")

    cs = [c for c, _ in sorted(valley_rows.items())]
    ms = [float(valley_rows[c]["M"]) for c in cs]
    a, k = ols_loglog(cs, ms)
    b = 1.0 - a
    a2 = two_point_a(cs[0], ms[0], cs[-1], ms[-1]) if len(cs) >= 2 else None

    proxies = []
    for c in cs:
        job = valley_rows[c]
        m_lab = float(job["M"])
        m_paper = paper_m_opt(c)
        proxies.append(
            {
                "c_flops": c,
                "job_id": job["job_id"],
                "arch_id": job.get("arch_id"),
                "M_lab": m_lab,
                "M_paper": m_paper,
                "M_lab_over_paper": m_lab / m_paper,
                "best_val_bpb": job.get("best_val_bpb"),
                "source": job.get("source"),
            }
        )

    summary = {
        "paper": {"a": PAPER_A, "b": PAPER_B, "M_opt": "0.1715 * C ** 0.5243"},
        "n_jobs_loaded": len(jobs),
        "n_jobs_plotted": len(kept),
        "skipped": skipped,
        "valley_proxies": proxies,
        "fit": {
            "a": a,
            "b": b,
            "k_M": k,
            "n_points": len(cs),
            "two_point_a_first_last": a2,
            "note": (
                "Illustrative 3-point OLS on valley proxies, not the paper's 8-budget fit. "
                "Two of three proxies sit on older corpus snapshots."
            ),
        },
    }
    fit_path = out_dir / "isoflop_fit.json"
    fit_path.write_text(json.dumps(summary, indent=2) + "\n")
    svg_path = out_dir / "isoflop_family.svg"
    write_family_svg(jobs, valley_rows, svg_path)

    print(f"a={a:.4f}  (paper {PAPER_A})")
    print(f"b={b:.4f}  (paper {PAPER_B})")
    print(f"k_M={k:.4f}  (paper {PAPER_K_M})")
    if a2 is not None:
        print(f"two_point_a={a2:.4f}")
    for p in proxies:
        print(
            f"  C={fmt_c(p['c_flops'])}  {p['job_id']}  "
            f"M={p['M_lab']:.3e}  paper={p['M_paper']:.3e}  "
            f"ratio={p['M_lab_over_paper']:.2f}  bpb={p['best_val_bpb']}"
        )
    print(f"wrote {fit_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
