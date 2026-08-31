#!/usr/bin/env python3
"""
Evaluate affine vs residual color projector on the local Pansor cohort.

Runs chart-free flash/no-flash cheek Lab vs FitSkin for each Pansor trial and
prints mean/median ΔE00 for show tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "flash_no_flash_skin_lab.py"
DEFAULT_MANIFEST = ROOT / "data" / "pansor" / "manifest_pansor_fitskin.csv"
DEFAULT_CAL = ROOT / "calibration" / "tier3_affine"
DEFAULT_PROJECTOR = ROOT / "calibration" / "color_projector_pansor" / "color_projector.npz"


def _load_eval_rows(
    manifest: Path,
    *,
    only_non_cc: bool = False,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("include_in_eval") != "yes":
                continue
            if not all(row.get(k) for k in ("path_noflash", "path_flash", "fitskin_cheek_L")):
                continue
            if only_non_cc and row.get("condition_code") == "CC":
                continue
            rows.append(row)
    return rows


def _write_fnf_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    fields = [
        "subject_id",
        "participant",
        "trial",
        "condition",
        "condition_code",
        "path_noflash",
        "path_flash",
        "fitskin_cheek_L",
        "fitskin_cheek_a",
        "fitskin_cheek_b",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run_pipeline(
    *,
    manifest: Path,
    out_dir: Path,
    cal_dir: Path,
    projector: Path | None,
    camera_wb: bool,
    fitskin_lightness: bool = False,
    bag_cat02: str = "auto",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--manifest",
        str(manifest),
        "--input-mode",
        "dng",
        "--iphone-calibration",
        str(cal_dir),
        "--cheek-roi",
        "--exposure-scale-skin-mask",
        "--known-ambient-cct-k",
        "6546",
        "--known-ambient-duv",
        "0.0017",
        "--bag-cat02",
        bag_cat02,
        "--production",
        "--out-dir",
        str(out_dir),
    ]
    if camera_wb:
        cmd.append("--raw-camera-wb")
    if projector is not None:
        cmd.extend(["--color-projector", str(projector)])
    if fitskin_lightness:
        cmd.append("--fitskin-lightness-calibration")
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out_dir / "flash_noflash_skin_lab.csv"


def _summarize(csv_path: Path) -> Dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    by_cond: Dict[str, List[float]] = {"CC": [], "BAG": [], "ALL": []}
    per_trial: List[Dict[str, Any]] = []
    for r in rows:
        try:
            de = float(r["reflectance_cheek_de00"])
        except (KeyError, TypeError, ValueError):
            continue
        if de != de:
            continue
        sid = r.get("subject_id", "")
        code = "BAG" if "BAG" in sid.upper() or "bag" in str(r.get("condition", "")).lower() else "CC"
        # Prefer condition_code if present in CSV extras — subject_id is reliable here.
        if "_BAG_" in sid.upper():
            code = "BAG"
        elif "_CC_" in sid.upper():
            code = "CC"
        by_cond[code].append(de)
        by_cond["ALL"].append(de)
        per_trial.append(
            {
                "subject_id": sid,
                "condition": code,
                "pipeline_L": r.get("reflectance_L"),
                "pipeline_a": r.get("reflectance_a"),
                "pipeline_b": r.get("reflectance_b"),
                "fitskin_L": r.get("fitskin_cheek_L"),
                "fitskin_a": r.get("fitskin_cheek_a"),
                "fitskin_b": r.get("fitskin_cheek_b"),
                "de00": de,
            }
        )

    summary: Dict[str, Any] = {"per_trial": per_trial, "stats": {}}
    for k, vals in by_cond.items():
        if not vals:
            continue
        summary["stats"][k] = {
            "n": len(vals),
            "mean_de00": float(mean(vals)),
            "median_de00": float(median(vals)),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--cal-dir", type=Path, default=DEFAULT_CAL)
    ap.add_argument("--projector", type=Path, default=DEFAULT_PROJECTOR)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "pansor_projector_show")
    ap.add_argument("--skip-affine", action="store_true")
    ap.add_argument("--skip-projector", action="store_true")
    ap.add_argument(
        "--only-non-cc",
        action="store_true",
        help="Evaluate only captures whose scene has no ColorChecker.",
    )
    ap.add_argument(
        "--bag-cat02",
        choices=("off", "auto", "on"),
        default="auto",
        help="Use 'off' for a fully chart/reference-free inference gate.",
    )
    ap.add_argument(
        "--with-fitskin-lightness",
        action="store_true",
        help="Also evaluate projector + FitSkin L* gain (pilot show calibration).",
    )
    args = ap.parse_args()

    rows = _load_eval_rows(args.manifest, only_non_cc=args.only_non_cc)
    man_path = args.out_dir / "manifest_pansor_eval.csv"
    _write_fnf_manifest(rows, man_path)

    report: Dict[str, Any] = {"n_trials": len(rows), "methods": {}}

    if not args.skip_affine:
        csv_a = _run_pipeline(
            manifest=man_path,
            out_dir=args.out_dir / "affine",
            cal_dir=args.cal_dir,
            projector=None,
            camera_wb=True,
            bag_cat02=args.bag_cat02,
        )
        report["methods"]["affine"] = _summarize(csv_a)

    if not args.skip_projector:
        if not args.projector.is_file():
            raise SystemExit(f"Missing projector: {args.projector}")
        csv_p = _run_pipeline(
            manifest=man_path,
            out_dir=args.out_dir / "projector",
            cal_dir=args.cal_dir,
            projector=args.projector,
            camera_wb=True,
            bag_cat02=args.bag_cat02,
        )
        report["methods"]["projector"] = _summarize(csv_p)

    if args.with_fitskin_lightness:
        if not args.projector.is_file():
            raise SystemExit(f"Missing projector: {args.projector}")
        csv_pl = _run_pipeline(
            manifest=man_path,
            out_dir=args.out_dir / "projector_fitskin_L",
            cal_dir=args.cal_dir,
            projector=args.projector,
            camera_wb=True,
            fitskin_lightness=True,
            bag_cat02=args.bag_cat02,
        )
        report["methods"]["projector_fitskin_L"] = _summarize(csv_pl)

    out_json = args.out_dir / "comparison_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== Pansor show comparison ===")
    for name, block in report["methods"].items():
        print(f"\n{name}")
        for cond, stats in block.get("stats", {}).items():
            print(
                f"  {cond:4s}  n={stats['n']:2d}  "
                f"mean={stats['mean_de00']:.2f}  median={stats['median_de00']:.2f}"
            )
        for t in block.get("per_trial", []):
            print(f"    {t['subject_id']:12s}  ΔE00={t['de00']:.2f}")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
