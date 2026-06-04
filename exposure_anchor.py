"""
Chart-free exposure anchor from offline ColorChecker training.

During ``train_flash_noflash_checker_calibration.py``, each trial stores
``white_patch_scale = Y_D65_white / Y_camera_white_patch`` so underexposed
captures are lifted to canonical white. At inference, multiply reflectance
linear RGB by the participant median scale (no checker in frame).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def participant_key(subject_id: str, participant: str = "") -> str:
    sid = (subject_id or "").strip().upper()
    if sid.startswith("P1"):
        return "P1"
    if sid.startswith("P2"):
        return "P2"
    part = (participant or "").strip()
    m = re.match(r"(?:participant\s*)?(\d+)\s*$", part, re.I)
    if m:
        return f"P{int(m.group(1))}"
    return sid.split("_")[0] if "_" in sid else sid or "unknown"


def aggregate_exposure_anchors(trial_log: List[Dict[str, Any]]) -> Dict[str, float]:
    """Median ``white_patch_scale`` per participant (P1, P2, …)."""
    by_pid: Dict[str, List[float]] = {}
    for row in trial_log:
        scale = row.get("white_patch_scale")
        if scale is None:
            continue
        pid = participant_key(str(row.get("subject_id", "")))
        by_pid.setdefault(pid, []).append(float(scale))
    if not by_pid:
        return {}
    import numpy as np

    return {pid: float(np.median(vals)) for pid, vals in sorted(by_pid.items())}


def load_exposure_anchors(calibration_dir: Path) -> Dict[str, float]:
    """
    Load participant → scale from ``iphone_calibration_bundle.json`` or
    ``exposure_anchor_by_participant.json``.
    """
    cal_dir = Path(calibration_dir)
    sidecar = cal_dir / "exposure_anchor_by_participant.json"
    if sidecar.is_file():
        with sidecar.open(encoding="utf-8") as f:
            d = json.load(f)
        anchors = d.get("by_participant", d)
        return {str(k): float(v) for k, v in anchors.items()}

    bundle_path = cal_dir / "iphone_calibration_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"No exposure anchors under {cal_dir} "
            "(need training bundle or exposure_anchor_by_participant.json)"
        )
    with bundle_path.open(encoding="utf-8") as f:
        bundle = json.load(f)
    if "exposure_anchor_by_participant" in bundle:
        return {str(k): float(v) for k, v in bundle["exposure_anchor_by_participant"].items()}
    trials = bundle.get("training_trials") or []
    anchors = aggregate_exposure_anchors(trials)
    if not anchors:
        raise ValueError(
            f"{bundle_path} has no training_trials[].white_patch_scale; re-run checker training."
        )
    return anchors


def save_exposure_anchors(cal_dir: Path, anchors: Dict[str, float], trial_log: List[Dict[str, Any]]) -> Path:
    cal_dir = Path(cal_dir)
    payload = {
        "by_participant": anchors,
        "training_trials": trial_log,
        "notes": (
            "Multiply chart-free reflectance linear RGB by by_participant[Px] at inference "
            "(--exposure-anchor-from-training). Same scales used to fit camera_rgb_to_xyz."
        ),
    }
    path = cal_dir / "exposure_anchor_by_participant.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
