#!/usr/bin/env python3
"""Extract ring-light torch zips and build ColorChecker training manifests.

The torch captures include an in-frame MCC24 (face + chart). This script
extracts DNG pairs, verifies patch detection, and writes CSV manifests for
``train_flash_noflash_checker_calibration.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import physio_skin_lab_raw_pr250 as pr250  # noqa: E402
from scripts.evaluate_pansor20_chartfree_d65 import (  # noqa: E402
    extract_zip,
    linear_rgb_to_preview_bgr,
    load_dng_linear,
)
from scripts.evaluate_ringlight_torch_illuminant import (  # noqa: E402
    default_data_root,
    discover_trials,
)


def detect_cc_on_dng(noflash_dng: Path, *, half_size: bool = True) -> bool:
    A0 = load_dng_linear(noflash_dng, half_size=half_size, use_camera_wb=False)
    preview = linear_rgb_to_preview_bgr(A0)
    return pr250.patch_linear_rgb_24(A0, preview, use_median=True) is not None


def ensure_extracted(zip_path: Path, cache_dir: Path) -> Optional[Dict[str, Path]]:
    out = cache_dir / zip_path.stem
    if out.is_dir() and (out / "NoFlash.dng").is_file():
        nf = out / "NoFlash.dng"
        fl = out / "Flash.dng"
    else:
        out.mkdir(parents=True, exist_ok=True)
        try:
            nf, fl, _ = extract_zip(zip_path, out)
        except Exception:
            shutil.rmtree(out, ignore_errors=True)
            return None
    lm = out / "face_landmarks.json"
    if not nf.is_file() or not fl.is_file():
        return None
    return {"noflash": nf, "flash": fl, "landmarks": lm}


def build_rows(
    trials: List[Dict[str, Any]],
    cache_dir: Path,
    *,
    half_size: bool,
    illuminant: Optional[str] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for trial in trials:
        if illuminant and trial["illuminant"] != illuminant:
            continue
        zpath = Path(trial["zip_path"])
        paths = ensure_extracted(zpath, cache_dir)
        if paths is None:
            print(f"skip extract fail: {zpath.name}")
            continue
        if not detect_cc_on_dng(paths["noflash"], half_size=half_size):
            print(f"skip no CC: {zpath.name}")
            continue
        rows.append(
            {
                "subject_id": trial["subject_id"],
                "person": trial["person"],
                "illuminant": trial["illuminant"],
                "wb_cell": trial.get("wb_cell", ""),
                "condition_code": "CC",
                "include_in_eval": "yes",
                "path_noflash": str(paths["noflash"].resolve()),
                "path_flash": str(paths["flash"].resolve()),
                "path_face_landmarks": str(paths["landmarks"].resolve()),
                "zip_path": str(zpath.resolve()),
                "zip_stem": trial.get("zip_stem", zpath.stem),
            }
        )
    return rows


def write_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id",
        "person",
        "illuminant",
        "wb_cell",
        "condition_code",
        "include_in_eval",
        "path_noflash",
        "path_flash",
        "path_face_landmarks",
        "zip_path",
        "zip_stem",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "ring_light_cc_cache",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "ring_light",
    )
    ap.add_argument("--half-size", action="store_true", default=True)
    args = ap.parse_args()

    data_root = Path(args.data_root or default_data_root()).expanduser().resolve()
    trials = discover_trials(data_root)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    all_rows = build_rows(trials, cache_dir, half_size=bool(args.half_size))
    f12_rows = [r for r in all_rows if r["illuminant"] == "F12"]
    d65_rows = [r for r in all_rows if r["illuminant"] == "D65"]

    write_manifest(all_rows, out_dir / "manifest_ring_cc_all.csv")
    write_manifest(f12_rows, out_dir / "manifest_ring_cc_f12.csv")
    write_manifest(d65_rows, out_dir / "manifest_ring_cc_d65.csv")

    summary = {
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "n_trials_scanned": len(trials),
        "n_cc_detected": len(all_rows),
        "n_f12": len(f12_rows),
        "n_d65": len(d65_rows),
        "by_person": {},
    }
    for ill_key, subset in (("all", all_rows), ("F12", f12_rows), ("D65", d65_rows)):
        bp: Dict[str, int] = {}
        for r in subset:
            bp[r["person"]] = bp.get(r["person"], 0) + 1
        summary["by_person"][ill_key] = bp

    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
