"""Spectral helpers for flash/no-flash training and Bradford CAT."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from iphone_camera_calibration import _cie_cmfs_1nm
from scr_awb import _planck_spd_on_wl
from skin_reflectance_priors import SkinReflectancePrior, load_prior

_EPS = 1e-12
D65_CCT_K = 6504.0


def planck_xyz_y1(cct_k: float, duv: float = 0.0) -> np.ndarray:
    """CIE XYZ of Planckian illuminant (Y=1), 10 nm integration."""
    try:
        from src.lu2006_ambient import planck_rgb_from_cct_duv  # type: ignore
    except ImportError:
        raise ImportError("mabl-flash-illumination required for planck_rgb_from_cct_duv")

    rgb = np.maximum(
        np.asarray(planck_rgb_from_cct_duv(float(cct_k), float(duv)), dtype=np.float64), _EPS
    )
    rgb = rgb / np.median(rgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = rgb @ m.T
    return xyz / max(float(xyz[1]), _EPS)


def skin_reflectance_to_xyz_d65(reflectance: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Skin reflectance × D65 illuminant → XYZ (Y=1)."""
    wl, cmf = _cie_cmfs_1nm()
    r = np.asarray(reflectance, dtype=np.float64)
    if r.shape[0] != wl.shape[0]:
        r = np.interp(wl, np.asarray(wavelengths_nm, dtype=np.float64), r)
    spd = _planck_spd_on_wl(wl, D65_CCT_K)
    xyz = (cmf * r[:, None] * spd[:, None]).sum(axis=0)
    return xyz / max(float(xyz[1]), _EPS)


def skin_reflectance_to_camera_rgb(
    reflectance: np.ndarray,
    wavelengths_nm: np.ndarray,
    spectral_sensitivity_rgb: np.ndarray,
) -> np.ndarray:
    """ISSA-style skin × D65 → linear camera RGB (median-normalized)."""
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    s = np.asarray(spectral_sensitivity_rgb, dtype=np.float64)
    r = np.asarray(reflectance, dtype=np.float64)
    if r.shape[0] != s.shape[0]:
        r = np.interp(wl, np.asarray(wavelengths_nm, dtype=np.float64), r)
    spd = _planck_spd_on_wl(wl, D65_CCT_K)
    rgb = (s * r[:, None] * spd[:, None]).sum(axis=0)
    rgb = np.maximum(rgb, _EPS)
    return rgb / max(float(np.median(rgb)), _EPS)


def issa_skin_calibration_rows(
    spectral_sensitivity_rgb: np.ndarray,
    wavelengths_nm: np.ndarray,
    prior_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extra (rgb, xyz) rows for matrix training from ISSA cheek medians."""
    rgbs: List[np.ndarray] = []
    xyzs: List[np.ndarray] = []
    for name in prior_names:
        prior = load_prior(str(name))
        wl = np.asarray(wavelengths_nm, dtype=np.float64)
        r = prior.resample_to(wl)
        rgbs.append(skin_reflectance_to_camera_rgb(r, wl, spectral_sensitivity_rgb))
        xyzs.append(skin_reflectance_to_xyz_d65(r, wl))
    return np.stack(rgbs, axis=0), np.stack(xyzs, axis=0)
