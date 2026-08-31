#!/usr/bin/env python3
"""Join Pansor FairFace7 ΔE with no-flash DNG EXIF (ISO / shutter).

Flags out-of-band captures: ISO>=200, shutter>=1/60, pipeline L*>=75.

Example:
  python3 scripts/audit_pansor_exposure.py \\
    --results results/pansor20_fairface7/pansor20_chartfree_d65.csv \\
    --out-dir results/pansor20_exposure_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dng_exif import read_noflash_exposure_from_zip  # noqa: E402

ISO_BAD = 200.0
SHUTTER_BAD = 1.0 / 60.0
LSTAR_BAD = 75.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results/pansor20_fairface7/pansor20_chartfree_d65.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/pansor20_exposure_audit",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows: List[Dict[str, Any]] = []
    n_iso_bad = n_shutter_bad = n_l_bad = n_ok = 0
    for r in rows:
        zip_path = Path(r["zip_path"])
        try:
            ex = read_noflash_exposure_from_zip(zip_path)
            err = ""
        except Exception as e:  # noqa: BLE001
            ex = {
                "iso": float("nan"),
                "shutter_s": float("nan"),
                "shutter_raw": "",
                "white_balance": -1,
                "model": "",
            }
            err = str(e)

        iso = float(ex["iso"])
        shutter_s = float(ex["shutter_s"])
        L = float(r["pipeline_L"])
        de = float(r["de00"])
        flags = []
        if iso == iso and iso >= ISO_BAD:
            flags.append("iso_high")
            n_iso_bad += 1
        if shutter_s == shutter_s and shutter_s >= SHUTTER_BAD:
            flags.append("shutter_long")
            n_shutter_bad += 1
        if L >= LSTAR_BAD:
            flags.append("L_high")
            n_l_bad += 1
        in_band = not flags
        if in_band:
            n_ok += 1

        ev = float("nan")
        if iso == iso and shutter_s == shutter_s and iso > 0 and shutter_s > 0:
            import math

            ev = math.log2(iso * shutter_s)

        out_rows.append(
            {
                "subject_id": r["subject_id"],
                "participant_id": r["participant_id"],
                "name": r["name"],
                "ethnicity": r["ethnicity"],
                "trial": r["trial"],
                "zip_stem": r["zip_stem"],
                "de00": de,
                "pipeline_L": L,
                "pipeline_a": float(r["pipeline_a"]),
                "pipeline_b": float(r["pipeline_b"]),
                "fitskin_L": float(r["fitskin_L"]),
                "fitskin_a": float(r["fitskin_a"]),
                "fitskin_b": float(r["fitskin_b"]),
                "iso": iso,
                "shutter_s": shutter_s,
                "shutter_raw": ex.get("shutter_raw", ""),
                "white_balance": ex.get("white_balance", -1),
                "model": ex.get("model", ""),
                "log2_iso_shutter": ev,
                "in_band": int(in_band),
                "flags": "|".join(flags),
                "exif_error": err,
            }
        )
        print(
            f"{r['subject_id']:8s} ISO={iso:6.1f} {ex.get('shutter_raw',''):8s} "
            f"L*={L:5.1f} ΔE={de:5.2f} {'OK' if in_band else flags}"
        )

    fieldnames = list(out_rows[0].keys())
    csv_path = args.out_dir / "pansor20_exposure_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    des = [r["de00"] for r in out_rows]
    isos = [r["iso"] for r in out_rows if r["iso"] == r["iso"]]
    shs = [r["shutter_s"] for r in out_rows if r["shutter_s"] == r["shutter_s"]]
    summary = {
        "n": len(out_rows),
        "n_in_band": n_ok,
        "n_iso_ge_200": n_iso_bad,
        "n_shutter_ge_1_60": n_shutter_bad,
        "n_L_ge_75": n_l_bad,
        "frac_in_band": n_ok / max(len(out_rows), 1),
        "mean_de00": mean(des),
        "median_de00": median(des),
        "iso_min": min(isos) if isos else None,
        "iso_max": max(isos) if isos else None,
        "iso_median": median(isos) if isos else None,
        "shutter_s_min": min(shs) if shs else None,
        "shutter_s_max": max(shs) if shs else None,
        "shutter_s_median": median(shs) if shs else None,
        "thresholds": {
            "iso_bad_ge": ISO_BAD,
            "shutter_bad_ge_s": SHUTTER_BAD,
            "L_bad_ge": LSTAR_BAD,
        },
        "source_results": str(args.results),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Sheets-friendly TSV
    tsv_path = args.out_dir / "pansor20_exposure_audit.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    print("\n=== Exposure audit ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {tsv_path}")


if __name__ == "__main__":
    main()
