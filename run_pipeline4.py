#!/usr/bin/env python3
"""
Pipeline 4 — single entry point for chart-free flash/no-flash skin Lab vs FitSkin.

Reproduces the Phase 4 evaluation stack (median ΔE₀₀ ≈ 3.5 on the five-trial cohort).

Usage (full booth dataset — discovers all *NoFlash* / *Flash* DNG pairs):

    python3 run_pipeline4.py /path/to/RAW/Dataset

Expected folder layout::

    RAW Dataset/
      Participant 1/Trial 1/IMG_*_NoFlash.DNG
      Participant 1/Trial 1/IMG_*_Flash.DNG
      ...

Usage (one trial — two DNG files):

    python3 run_pipeline4.py --trial P1_T2 noflash.dng flash.dng

FitSkin cheek reference Lab for each trial is bundled in
``data/phase4_fitskin_reference.csv`` (same-session scanner values from the paper).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "flash_no_flash_skin_lab.py"
CAL_DIR = ROOT / "calibration" / "tier3_affine"
FITSKIN_REF = ROOT / "data" / "phase4_fitskin_reference.csv"
DEFAULT_OUT = ROOT / "pipeline4_output"


def _load_fitskin_reference() -> Dict[str, Dict[str, str]]:
    if not FITSKIN_REF.is_file():
        raise SystemExit(f"Missing bundled FitSkin reference: {FITSKIN_REF}")
    out: Dict[str, Dict[str, str]] = {}
    with FITSKIN_REF.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row["subject_id"]).strip()
            out[sid] = {k: str(row[k]) for k in row if k != "subject_id"}
    return out


def _attach_fitskin(rows: List[Dict[str, str]], ref: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for row in rows:
        rec = dict(row)
        sid = str(rec.get("subject_id", "")).strip()
        if sid in ref:
            rec.update(ref[sid])
        merged.append(rec)
    return merged


def _discover_pairs(data_root: Path) -> List[Dict[str, str]]:
    sys.path.insert(0, str(ROOT))
    from flash_no_flash_skin_lab import discover_raw_pairs

    return discover_raw_pairs(data_root)


def _one_pair_row(trial_id: str, nf: Path, fl: Path, ref: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    trial_id = trial_id.strip().upper()
    if trial_id not in ref:
        known = ", ".join(sorted(ref))
        raise SystemExit(f"Unknown --trial {trial_id!r}. Known trials: {known}")
    meta = ref[trial_id]
    return {
        "subject_id": trial_id,
        "participant": meta["participant"],
        "trial": meta["trial"],
        "path_noflash": str(nf.expanduser().resolve()),
        "path_flash": str(fl.expanduser().resolve()),
        **{k: v for k, v in meta.items() if k.startswith("fitskin_")},
    }


def _write_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    fields = [
        "subject_id",
        "participant",
        "trial",
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
        w.writerows(rows)


def _print_summary(out_dir: Path) -> None:
    csv_path = out_dir / "flash_noflash_skin_lab.csv"
    if not csv_path.is_file():
        print(f"No results CSV at {csv_path}", file=sys.stderr)
        return

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    de_vals = []
    print("\n" + "=" * 60)
    print("RESULTS  (reflectance cheek Lab vs FitSkin)")
    print("=" * 60)
    for r in rows:
        sid = r.get("subject_id", "")
        try:
            de = float(r["reflectance_cheek_de00"])
        except (KeyError, TypeError, ValueError):
            de = float("nan")
        if de == de:
            de_vals.append(de)
        L = r.get("reflectance_L", "n/a")
        a = r.get("reflectance_a", "n/a")
        b = r.get("reflectance_b", "n/a")
        Lf = r.get("fitskin_cheek_L", "n/a")
        af = r.get("fitskin_cheek_a", "n/a")
        bf = r.get("fitskin_cheek_b", "n/a")
        de_str = f"{de:.2f}" if de == de else "n/a"
        print(f"\n{sid}")
        print(f"  Pipeline:  L*={L}  a*={a}  b*={b}")
        print(f"  FitSkin:   L*={Lf}  a*={af}  b*={bf}")
        print(f"  ΔE₀₀ = {de_str}")

    if de_vals:
        med = float(median(de_vals))
        mean = sum(de_vals) / len(de_vals)
        print("\n" + "-" * 60)
        print(f"Trials: {len(de_vals)}   mean ΔE₀₀ = {mean:.2f}   median ΔE₀₀ = {med:.2f}")
        print("Paper target (5-trial chart-free cohort): median ΔE₀₀ ≈ 3.50")
        print("=" * 60)

    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        print(f"\nFull output: {csv_path}")
        print(f"Summary:     {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Pipeline 4 (chart-free flash/no-flash skin Lab vs FitSkin).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        help="Root folder with Participant */Trial */*NoFlash*.DNG pairs (booth RAW dataset).",
    )
    ap.add_argument(
        "--trial",
        metavar="ID",
        help="Single-trial mode: subject id from reference CSV (e.g. P1_T2).",
    )
    ap.add_argument("dng_files", nargs="*", type=Path, help="With --trial: noflash.dng flash.dng")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT.name}/)",
    )
    ap.add_argument(
        "--app-proraw",
        action="store_true",
        help="Use embedded camera WB (required for app-exported ProRAW, not booth RAW).",
    )
    args = ap.parse_args()

    if not PIPELINE.is_file():
        raise SystemExit(f"Pipeline script not found: {PIPELINE}")
    if not (CAL_DIR / "iphone_calibration_bundle.json").is_file():
        raise SystemExit(f"Missing calibration bundle: {CAL_DIR}")

    ref = _load_fitskin_reference()
    out_dir = args.out_dir.expanduser().resolve()
    manifest_path = out_dir / "manifest_pipeline4.csv"

    if args.trial:
        if len(args.dng_files) != 2:
            raise SystemExit("Single-trial mode: python3 run_pipeline4.py --trial P1_T2 noflash.dng flash.dng")
        nf, fl = args.dng_files
        for p in (nf, fl):
            if not p.is_file():
                raise SystemExit(f"Missing DNG: {p}")
        rows = [_one_pair_row(args.trial, nf, fl, ref)]
    else:
        data_root = args.data_root or Path(
            __import__("os").environ.get("DATA_ROOT", "")
        ).expanduser()
        if not data_root or not data_root.is_dir():
            raise SystemExit(
                "Provide DATA_ROOT path:\n"
                "  python3 run_pipeline4.py /path/to/RAW/Dataset\n"
                "Or one trial:\n"
                "  python3 run_pipeline4.py --trial P1_T2 noflash.dng flash.dng"
            )
        rows = _attach_fitskin(_discover_pairs(data_root), ref)

    _write_manifest(rows, manifest_path)

    cmd = [
        sys.executable,
        str(PIPELINE),
        "--manifest",
        str(manifest_path),
        "--input-mode",
        "dng",
        "--iphone-calibration",
        str(CAL_DIR),
        "--cheek-roi",
        "--exposure-scale-skin-mask",
        "--exposure-anchor-from-training",
        "--known-ambient-cct-k",
        "6546",
        "--known-ambient-duv",
        "0.0017",
        "--exclude-trials",
        "P2_T1",
        "--production",
        "--out-dir",
        str(out_dir),
    ]
    if args.app_proraw:
        cmd.append("--raw-camera-wb")

    print("Running Pipeline 4...")
    print(" ".join(cmd))
    print()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    _print_summary(out_dir)


if __name__ == "__main__":
    main()
