#!/usr/bin/env python3
"""Aggregate ring-light ΔE₀₀ eval CSV rows by capture wb_cell (A–E).

Reads ``torch_illuminant_ringlight.csv`` (or compatible TSV) and writes a compact
JSON summary for Colab / docs — no re-inference required.

Example:
  python3 scripts/summarize_ring_eval_by_wb_cell.py \\
    --csv results/torch_illuminant_ringlight/torch_illuminant_ringlight.csv \\
    --out data/ring_light/eval_n84_by_wb_cell.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]

# Booth Lighting.xlsx factorial (capture WB set on phone, not applied in pipeline)
CAPTURE_WB_K = {
    "D65": {"A": 5500, "B": 6000, "C": 6500, "D": 7000, "E": 7500},
    "F12": {"A": 2500, "B": 2500, "C": 3000, "D": 3500, "E": 4000},
}

DEFAULT_ARMS = [
    "frozen_5500",
    "hybrid_deploy",
    "hybrid_multi_lab",
]


def _stats(vals: Iterable[float]) -> Dict[str, Any]:
    v = [float(x) for x in vals if x is not None and x == x]
    if not v:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(v), "mean": round(mean(v), 4), "median": round(median(v), 4)}


def _de_cols(row: dict, arms: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for arm in arms:
        key = f"de00_{arm}"
        raw = row.get(key)
        if raw is None or raw == "":
            out[arm] = None
        else:
            try:
                out[arm] = float(raw)
            except (TypeError, ValueError):
                out[arm] = None
    return out


def summarize_rows(rows: List[dict], arms: List[str]) -> Dict[str, Any]:
    by_wb: Dict[str, List[dict]] = defaultdict(list)
    by_ill_wb: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        cell = str(r.get("wb_cell") or "").strip().upper()
        ill = str(r.get("illuminant") or "").strip().upper()
        if not cell:
            continue
        by_wb[cell].append(r)
        if ill:
            by_ill_wb[f"{ill}_{cell}"].append(r)

    def arm_stats(grp: List[dict]) -> Dict[str, Any]:
        return {arm: _stats([_de_cols(x, arms)[arm] for x in grp]) for arm in arms}

    out: Dict[str, Any] = {
        "n_trials": len(rows),
        "arms": arms,
        "capture_wb_k": CAPTURE_WB_K,
        "overall": arm_stats(rows),
        "by_wb_cell": {cell: arm_stats(grp) for cell, grp in sorted(by_wb.items())},
        "by_illuminant_wb_cell": {},
    }
    for key, grp in sorted(by_ill_wb.items()):
        ill, cell = key.split("_", 1)
        out["by_illuminant_wb_cell"].setdefault(ill, {})[cell] = arm_stats(grp)
    return out


def load_rows(path: Path) -> List[dict]:
    path = path.expanduser().resolve()
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=delim))
    if not rows:
        return rows
    # ringlight_de00.csv uses plain ``de00`` for frozen-5500 forehead eval
    if "de00" in rows[0] and "de00_frozen_5500" not in rows[0]:
        for r in rows:
            r["de00_frozen_5500"] = r.get("de00")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "torch_illuminant_ringlight" / "torch_illuminant_ringlight.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "ring_light" / "eval_n84_by_wb_cell.json",
    )
    ap.add_argument(
        "--arms",
        nargs="*",
        default=DEFAULT_ARMS,
        help="de00_* column suffixes to aggregate (default: best-stack arms)",
    )
    ap.add_argument(
        "--source-note",
        default="scripts/evaluate_ringlight_torch_illuminant.py on manifest_ring_cc_all (n=84)",
    )
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    summary = summarize_rows(rows, list(args.arms))
    summary["source_csv"] = str(args.csv.resolve())
    summary["source_note"] = args.source_note

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}  (n={summary['n_trials']} trials, arms={args.arms})")

    for arm in args.arms:
        overall = summary["overall"].get(arm, {})
        if overall.get("mean") is not None:
            print(f"  {arm}: mean ΔE={overall['mean']:.2f}  (n={overall['n']})")


if __name__ == "__main__":
    main()
