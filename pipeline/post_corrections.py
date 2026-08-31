"""Post-CAT Lab correctors and optional RGB→XYZ projector loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAB_AFFINE = ROOT / "calibration" / "lab_affine_corrector_pansor" / "lab_affine_4x3.npy"
DEFAULT_MULTI_LAB_AFFINE = ROOT / "calibration" / "multi_illuminant_lab_affine" / "multi_illuminant_lab_affine.json"
DEFAULT_COLOR_PROJECTOR = ROOT / "calibration" / "color_projector_pansor" / "color_projector.npz"
DEFAULT_WARM_AFFINE = ROOT / "calibration" / "tier3_affine_warm_cc"
DEFAULT_D65_RING_AFFINE = ROOT / "calibration" / "tier3_affine_d65_ring_cc"
DEFAULT_ROUTED_AFFINE_BUNDLE = ROOT / "calibration" / "tier3_affine_illuminant_routed.json"


def load_routed_affine_bundle(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_ROUTED_AFFINE_BUNDLE).expanduser().resolve()
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "paths": {
            "default_cool": str(ROOT / "calibration" / "tier3_affine"),
            "warm_f12": str(DEFAULT_WARM_AFFINE),
            "d65_ring": str(DEFAULT_D65_RING_AFFINE),
        }
    }


def select_affine_for_illuminant(
    illuminant: str,
    *,
    M_default: np.ndarray,
    M_warm: Optional[np.ndarray] = None,
    M_d65_ring: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    ill = str(illuminant).upper()
    if ill in ("F12", "D12") and M_warm is not None:
        return M_warm, "warm_f12"
    if ill == "D65" and M_d65_ring is not None:
        return M_d65_ring, "d65_ring"
    return M_default, "tier3_default"


def load_lab_affine(path: Optional[Path] = None) -> np.ndarray:
    p = Path(path or DEFAULT_LAB_AFFINE).expanduser().resolve()
    if p.suffix == ".npy":
        W = np.load(p)
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
        W = np.asarray(payload["lab_affine_4x3"], dtype=np.float64)
    if W.shape != (4, 3):
        raise ValueError(f"lab affine must be 4x3, got {W.shape}")
    return W


def apply_lab_affine(lab: np.ndarray, W: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64).reshape(3)
    return np.array([lab[0], lab[1], lab[2], 1.0], dtype=np.float64) @ W


def load_multi_illuminant_lab_affine(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_MULTI_LAB_AFFINE).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"multi-illuminant lab affine not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def apply_multi_illuminant_lab_affine(
    lab: np.ndarray,
    illuminant: str,
    bundle: Dict[str, Any],
    *,
    mode: str = "routed",
    person_key: Optional[str] = None,
    loo: bool = False,
) -> Tuple[np.ndarray, str]:
    """Apply illuminant-routed or global Lab corrector from training bundle."""
    ill = str(illuminant).upper()
    if ill == "D12":
        ill = "F12"
    if loo and person_key and bundle.get("loo_weights_by_person"):
        loo_map = bundle["loo_weights_by_person"]
        if person_key in loo_map:
            key = "W_f12" if ill == "F12" else "W_d65"
            W = np.asarray(loo_map[person_key][key], dtype=np.float64)
            return apply_lab_affine(lab, W), f"loo_{key}"
    key = mode
    if mode == "routed":
        key = "W_f12" if ill == "F12" else "W_d65"
    elif mode == "global":
        key = "W_global"
    else:
        key = mode
    matrices = bundle.get("matrices") or {}
    if key not in matrices:
        key = "W_global"
    W = np.asarray(matrices[key], dtype=np.float64)
    return apply_lab_affine(lab, W), key


def load_color_projector(path: Optional[Path] = None) -> Any:
    from models.color_projector import load_color_projector_artifact

    p = Path(path or DEFAULT_COLOR_PROJECTOR).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"color projector not found: {p}")
    return load_color_projector_artifact(p)
