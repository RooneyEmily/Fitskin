#!/usr/bin/env python3
"""Merge Emily/Liki ColorChecker training pairs from all available cohorts.

Sources (per tone tier):
  - Pansor iPhone ProRAW DNGs (``raw_camera_wb=yes``)
  - Phase-4 booth RAW DNGs (``raw_camera_wb=no``)
  - Bundled chart_cc JPEG pairs (8-bit; no camera WB)

Example::

    python3 scripts/build_tone_training_manifest.py \\
        --booth-raw-root "/path/to/RAW Dataset"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_flash_noflash_checker_calibration import _discover_pairs

OUT_FIELDS = [
    "subject_id",
    "person",
    "participant",
    "source",
    "path_noflash",
    "path_flash",
    "raw_camera_wb",
    "include_in_eval",
]


def _dedupe(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        key = (row["path_noflash"], row["path_flash"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _pansor_cc_rows(manifest: Path, *, person: str, participant: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("include_in_eval", "yes")).strip().lower() not in ("yes", "1", "true"):
                continue
            if row.get("condition_code", "CC") != "CC":
                continue
            if row.get("person", "") != person:
                continue
            nf = str(row.get("path_noflash", "")).strip()
            fl = str(row.get("path_flash", "")).strip()
            if not nf or not fl:
                continue
            rows.append(
                {
                    "subject_id": str(row.get("subject_id", "")),
                    "person": person,
                    "participant": participant,
                    "source": "pansor_dng",
                    "path_noflash": nf,
                    "path_flash": fl,
                    "raw_camera_wb": "yes",
                    "include_in_eval": "yes",
                }
            )
    return rows


def _booth_rows(booth_root: Path, *, prefix: str, person: str, participant: str) -> List[Dict[str, str]]:
    if not booth_root.is_dir():
        return []
    rows: List[Dict[str, str]] = []
    for pair in _discover_pairs(booth_root):
        sid = pair["subject_id"]
        if not sid.startswith(prefix):
            continue
        rows.append(
            {
                "subject_id": f"{sid}_booth",
                "person": person,
                "participant": participant,
                "source": "booth_dng",
                "path_noflash": pair["path_noflash"],
                "path_flash": pair["path_flash"],
                "raw_camera_wb": "no",
                "include_in_eval": "yes",
            }
        )
    return rows


def _jpeg_rows(jpeg_root: Path, *, prefix: str, person: str, participant: str) -> List[Dict[str, str]]:
    part_dir = jpeg_root / f"Participant_{prefix[1]}"
    if not part_dir.is_dir():
        return []
    rows: List[Dict[str, str]] = []
    for trial_dir in sorted(part_dir.glob("Trial_*")):
        nf = fl = None
        for p in trial_dir.iterdir():
            if not p.is_file():
                continue
            name = p.name.lower()
            if "noflash" in name and p.suffix.lower() in (".jpg", ".jpeg"):
                nf = p
            elif "flash" in name and "noflash" not in name and p.suffix.lower() in (".jpg", ".jpeg"):
                fl = p
        if nf and fl:
            tnum = trial_dir.name.split("_")[-1]
            rows.append(
                {
                    "subject_id": f"{prefix}_JPEG_T{tnum}",
                    "person": person,
                    "participant": participant,
                    "source": "chart_cc_jpeg",
                    "path_noflash": str(nf.resolve()),
                    "path_flash": str(fl.resolve()),
                    "raw_camera_wb": "no",
                    "include_in_eval": "yes",
                }
            )
    return rows


def _write_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pansor-manifest",
        type=Path,
        default=ROOT / "data" / "pansor" / "manifest_pansor_fitskin.csv",
    )
    ap.add_argument(
        "--booth-raw-root",
        type=Path,
        default=Path(
            "/home/mabl-main/Documents/RAW Dataset-20260531T233644Z-3-001/RAW Dataset"
        ),
    )
    ap.add_argument(
        "--jpeg-root",
        type=Path,
        default=ROOT / "data" / "chart_cc_jpeg",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "training",
    )
    ap.add_argument(
        "--dark-sources",
        choices=("pansor", "pansor+booth", "all"),
        default="pansor",
        help=(
            "Dark-tier training sources. Booth/JPEG pull the matrix toward lighter L* on "
            "Pansor ProRAW (default: pansor only)."
        ),
    )
    args = ap.parse_args()

    light = _dedupe(
        _pansor_cc_rows(args.pansor_manifest, person="Emily", participant="Participant 1")
        + _booth_rows(args.booth_raw_root, prefix="P1", person="Emily", participant="Participant 1")
        + _jpeg_rows(args.jpeg_root, prefix="P1", person="Emily", participant="Participant 1")
    )
    dark_parts = _pansor_cc_rows(args.pansor_manifest, person="Liki", participant="Participant 2")
    if args.dark_sources in ("pansor+booth", "all"):
        dark_parts += _booth_rows(
            args.booth_raw_root, prefix="P2", person="Liki", participant="Participant 2"
        )
    if args.dark_sources == "all":
        dark_parts += _jpeg_rows(
            args.jpeg_root, prefix="P2", person="Liki", participant="Participant 2"
        )
    dark = _dedupe(dark_parts)

    light_path = args.out_dir / "manifest_tone_light.csv"
    dark_path = args.out_dir / "manifest_tone_dark.csv"
    _write_manifest(light, light_path)
    _write_manifest(dark, dark_path)

    print(f"Light training manifest: {len(light)} pair(s) → {light_path}")
    for r in light:
        print(f"  {r['subject_id']:16} {r['source']:14} wb={r['raw_camera_wb']}")
    print(f"Dark training manifest: {len(dark)} pair(s) → {dark_path}")
    for r in dark:
        print(f"  {r['subject_id']:16} {r['source']:14} wb={r['raw_camera_wb']}")


if __name__ == "__main__":
    main()
