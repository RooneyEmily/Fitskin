#!/usr/bin/env python3
"""
Aggregate flash/no-flash skin Lab results for report tables.

Reads ``flash_noflash_skin_lab.csv`` from JPEG and/or DNG output dirs; writes:
  - ``flash_noflash_method_comparison_per_trial.csv``
  - ``flash_noflash_method_comparison_summary.csv``
  - ``flash_noflash_method_comparison.md`` (paste into report)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent

METHODS = [
    ("noflash", "No-flash (display Lab)"),
    ("flash_aligned", "Flash aligned (display Lab)"),
    ("lu_wb", "Lu 2006 WB (display Lab)"),
    ("scr_awb_wb", "SCR-AWB Zhou 2025 (display Lab)"),
    ("reflectance", "Reflectance √(F×NF) (linear Lab)"),
]

DE_COLS = {
    "noflash": "noflash_cheek_de00",
    "flash_aligned": "flash_aligned_cheek_de00",
    "lu_wb": "lu_wb_cheek_de00",
    "scr_awb_wb": "scr_awb_wb_cheek_de00",
    "reflectance": "reflectance_cheek_de00",
}

LAB_PREFIX = {
    "noflash": "noflash",
    "flash_aligned": "flash_aligned",
    "lu_wb": "lu_wb",
    "reflectance": "reflectance",
    "scr_awb_wb": "scr_awb_wb",
}


def load_csv(path: Path, dataset: str) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_dataset"] = dataset
    return rows


def _f(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def pid(row: Dict[str, str]) -> str:
    sid = row.get("subject_id", "")
    if sid.startswith("P1"):
        return "P1"
    if sid.startswith("P2"):
        return "P2"
    return row.get("participant", "?")


def aggregate(rows: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict]]:
    per_trial: List[Dict] = []
    for row in rows:
        rec = {
            "dataset": row["_dataset"],
            "subject_id": row["subject_id"],
            "participant": pid(row),
            "trial": row.get("trial", ""),
            "fitskin_L": _f(row, "fitskin_cheek_L"),
            "fitskin_a": _f(row, "fitskin_cheek_a"),
            "fitskin_b": _f(row, "fitskin_cheek_b"),
            "lu_cct_k": _f(row, "ambient_cct_k"),
        }
        for key, _label in METHODS:
            p = LAB_PREFIX[key]
            rec[f"{key}_L"] = _f(row, f"{p}_L")
            rec[f"{key}_a"] = _f(row, f"{p}_a")
            rec[f"{key}_b"] = _f(row, f"{p}_b")
            rec[f"{key}_de00"] = _f(row, DE_COLS[key])
        per_trial.append(rec)

    summary_keys: List[Tuple[str, str, str]] = []
    for ds in sorted({r["dataset"] for r in per_trial}):
        for part in ("P1", "P2", "ALL"):
            summary_keys.append((ds, part, "per_participant"))
        for mkey, _ in METHODS:
            summary_keys.append((ds, "ALL", mkey))

    summary: List[Dict] = []
    seen = set()

    def pool(filter_fn) -> List[Dict]:
        return [r for r in per_trial if filter_fn(r)]

    for ds in sorted({r["dataset"] for r in per_trial}):
        ds_rows = [r for r in per_trial if r["dataset"] == ds]
        for mkey, mlabel in METHODS:
            de = [r[f"{mkey}_de00"] for r in ds_rows if np.isfinite(r[f"{mkey}_de00"])]
            if de:
                summary.append(
                    {
                        "dataset": ds,
                        "group": "ALL",
                        "method": mkey,
                        "method_label": mlabel,
                        "n_trials": len(de),
                        "de00_mean": float(np.mean(de)),
                        "de00_median": float(np.median(de)),
                        "de00_std": float(np.std(de, ddof=1)) if len(de) > 1 else 0.0,
                        "de00_min": float(np.min(de)),
                        "de00_max": float(np.max(de)),
                    }
                )
        for part in ("P1", "P2"):
            part_rows = [r for r in ds_rows if r["participant"] == part]
            for mkey, mlabel in METHODS:
                de = [r[f"{mkey}_de00"] for r in part_rows if np.isfinite(r[f"{mkey}_de00"])]
                if de:
                    summary.append(
                        {
                            "dataset": ds,
                            "group": part,
                            "method": mkey,
                            "method_label": mlabel,
                            "n_trials": len(de),
                            "de00_mean": float(np.mean(de)),
                            "de00_median": float(np.median(de)),
                            "de00_std": float(np.std(de, ddof=1)) if len(de) > 1 else 0.0,
                            "de00_min": float(np.min(de)),
                            "de00_max": float(np.max(de)),
                        }
                    )

    return per_trial, summary


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, per_trial: List[Dict], summary: List[Dict]) -> None:
    lines = [
        "# Flash / no-flash skin color — method comparison",
        "",
        "Reference: **FitSkin cheek Lab** (D65, scan-sessions CSV). **P2_T1 excluded** (May 8 scan session).",
        "",
        "No color checker; no known booth illuminant used at runtime (Lu & Drew 2006 estimates CCT from each pair).",
        "",
        "## ΔE₀₀ vs FitSkin cheek — summary by dataset and method",
        "",
        "| Dataset | Group | Method | *n* | Mean | Median | SD | Min | Max |",
        "|---------|-------|--------|-----|------|--------|-----|-----|-----|",
    ]
    for s in summary:
        if s["group"] in ("P1", "P2"):
            continue
        lines.append(
            f"| {s['dataset']} | {s['group']} | {s['method_label']} | {s['n_trials']} | "
            f"{s['de00_mean']:.2f} | {s['de00_median']:.2f} | {s['de00_std']:.2f} | "
            f"{s['de00_min']:.2f} | {s['de00_max']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## ΔE₀₀ vs FitSkin — by participant",
            "",
            "| Dataset | Participant | Method | *n* | Mean | Median |",
            "|---------|-------------|--------|-----|------|--------|",
        ]
    )
    for s in summary:
        if s["group"] not in ("P1", "P2"):
            continue
        lines.append(
            f"| {s['dataset']} | {s['group']} | {s['method_label']} | {s['n_trials']} | "
            f"{s['de00_mean']:.2f} | {s['de00_median']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-trial detail (reflectance method)",
            "",
            "| Dataset | Trial | FitSkin L*a*b* | Reflectance L*a*b* | ΔE₀₀ | Lu CCT (K) |",
            "|---------|-------|----------------|---------------------|------|------------|",
        ]
    )
    for r in per_trial:
        lines.append(
            f"| {r['dataset']} | {r['subject_id']} | "
            f"{r['fitskin_L']:.1f}, {r['fitskin_a']:.1f}, {r['fitskin_b']:.1f} | "
            f"{r['reflectance_L']:.1f}, {r['reflectance_a']:.1f}, {r['reflectance_b']:.1f} | "
            f"{r['reflectance_de00']:.2f} | {r['lu_cct_k']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate flash/no-flash results for report tables.")
    ap.add_argument(
        "--jpeg-csv",
        type=Path,
        default=ROOT / "flash_noflash_skin_output" / "flash_noflash_skin_lab.csv",
    )
    ap.add_argument(
        "--dng-csv",
        type=Path,
        default=ROOT / "flash_noflash_dng_output" / "flash_noflash_skin_lab.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "flash_noflash_report",
    )
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    if args.jpeg_csv.is_file():
        rows.extend(load_csv(args.jpeg_csv, "JPEG"))
    if args.dng_csv.is_file():
        rows.extend(load_csv(args.dng_csv, "DNG"))
    if not rows:
        raise SystemExit("No input CSVs found.")

    per_trial, summary = aggregate(rows)
    out = args.out_dir
    write_csv(
        out / "flash_noflash_method_comparison_per_trial.csv",
        per_trial,
        list(per_trial[0].keys()),
    )
    write_csv(
        out / "flash_noflash_method_comparison_summary.csv",
        summary,
        list(summary[0].keys()),
    )
    write_markdown(out / "flash_noflash_method_comparison.md", per_trial, summary)
    print(f"Wrote {len(per_trial)} trial row(s), {len(summary)} summary row(s) → {out}/")


if __name__ == "__main__":
    main()
