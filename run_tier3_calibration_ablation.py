#!/usr/bin/env python3
"""
Train Tier-3 calibration variants and measure reflectance ΔE₀₀ vs FitSkin.

Inference stack (fixed): --cheek-roi --exposure-scale-skin-mask --exposure-anchor-from-training
Baseline matrix: iphone17pro_trained (unweighted lstsq, no ISSA rows).

Each Tier-3 training change is trained to ``calibration/tier3_<name>/`` and evaluated in isolation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    "/home/mabl-main/Documents/RAW Dataset-20260531T233644Z-3-001/RAW Dataset"
)
MONO = ROOT / "calibration" / "iphone17pro_camera_color"
BASE_CAL = ROOT / "calibration" / "iphone17pro_trained"
CAL_ROOT = ROOT / "calibration"
ISSA_PRIORS = "issa_median_caucasian,issa_median_african"

# (name, extra train CLI args beyond shared flags)
TIER3_TRAINS: List[Tuple[str, List[str]]] = [
    ("baseline", []),  # copy/reference only — uses BASE_CAL
    ("weighted_lstsq", ["--weighted-matrix"]),
    ("issa_lstsq", ["--issa-skin-rows", ISSA_PRIORS]),
    ("issa_weighted", ["--weighted-matrix", "--issa-skin-rows", ISSA_PRIORS]),
    ("affine", ["--matrix-affine"]),
    ("issa_affine", ["--matrix-affine", "--issa-skin-rows", ISSA_PRIORS]),
    ("huber_stacked", ["--huber-matrix-stacked", "--weighted-matrix"]),
    (
        "huber_issa",
        ["--huber-matrix-stacked", "--weighted-matrix", "--issa-skin-rows", ISSA_PRIORS],
    ),
]

INFER_FLAGS = [
    "--cheek-roi",
    "--exposure-scale-skin-mask",
    "--exposure-anchor-from-training",
    "--known-ambient-cct-k",
    "6546",
    "--known-ambient-duv",
    "0.0017",
    "--exclude-trials",
    "P2_T1",
]


def _run(cmd: List[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _train(name: str, extra: List[str]) -> Path:
    out = CAL_ROOT / f"tier3_{name}"
    if name == "baseline":
        return BASE_CAL
    shared = [
        sys.executable,
        str(ROOT / "train_flash_noflash_checker_calibration.py"),
        "--data-root",
        str(DATA_ROOT),
        "--monochromator-bundle",
        str(MONO),
        "--out-dir",
        str(out),
    ]
    _run(shared + extra)
    return out


def _infer(cal_dir: Path, out_name: str) -> Dict[str, Any]:
    out = ROOT / f"flash_tier3_{out_name}"
    cmd = [
        sys.executable,
        str(ROOT / "flash_no_flash_skin_lab.py"),
        "--data-root",
        str(DATA_ROOT),
        "--input-mode",
        "dng",
        "--out-dir",
        str(out),
        "--iphone-calibration",
        str(cal_dir),
    ] + INFER_FLAGS
    _run(cmd)
    summary_path = out / "summary.json"
    with summary_path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    results: Dict[str, Any] = {
        "inference_stack": INFER_FLAGS,
        "reference_baseline_calibration": str(BASE_CAL),
        "tier2_baseline_median_de00": 4.495227943083502,
        "variants": {},
    }

    for name, extra in TIER3_TRAINS:
        print(f"\n=== Tier-3: {name} ===", flush=True)
        cal_dir = _train(name, extra)
        summary = _infer(cal_dir, name)
        med = summary.get("reflectance_cheek_de00_median")
        mean = summary.get("reflectance_cheek_de00_mean")
        ref_med = results["tier2_baseline_median_de00"]
        delta = float(med) - float(ref_med) if med is not None else None
        results["variants"][name] = {
            "calibration_dir": str(cal_dir),
            "train_args": extra,
            "reflectance_cheek_de00_median": med,
            "reflectance_cheek_de00_mean": mean,
            "delta_median_vs_tier2_stack": delta,
            "matrix_fit": summary.get("matrix_fit"),
        }
        print(f"  median ΔE00 = {med}  (Δ vs tier2 {delta:+.3f})" if delta is not None else f"  median = {med}")

    out_json = ROOT / "flash_noflash_tier3_ablation.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
