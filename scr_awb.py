"""
SCR-AWB (Zhou et al. 2025) — portrait illuminant from skin median RGB.

Uses monochromator-derived camera spectral sensitivity S_j(lambda), an ISSA (or other)
skin reflectance prior r(lambda), and a three-term illuminant basis (Planckians at
fixed CCTs). Solves M @ alpha = rgb_skin (Eq. 6 spirit), then diagonal WB on no-flash.

Reference: Sicong Zhou et al., Technologies 2025, 13, 232.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from skin_reflectance_priors import SkinReflectancePrior, load_prior

ROOT = Path(__file__).resolve().parent
FLASH_REPO = ROOT.parent / "mabl-flash-illumination"

try:
    from vendor.flash_align.lu2006_ambient import _planck_rgb_linear
except ImportError:
    if FLASH_REPO.is_dir():
        sys.path.insert(0, str(FLASH_REPO))
    try:
        from src.lu2006_ambient import _planck_rgb_linear  # type: ignore
    except ImportError:
        _planck_rgb_linear = None  # type: ignore

if _planck_rgb_linear is None:

    def _planck_rgb_linear(cct_k: float) -> np.ndarray:
        """Fallback Planck → linear sRGB if mabl-flash-illumination is missing."""
        wl = np.arange(400, 701, 10, dtype=np.float64)
        h, c2 = 6.62607015e-34, 1.438776877e-2
        spd = (wl * 1e-9) ** -5 / (np.exp(c2 / (wl * 1e-9 * cct_k)) - 1.0 + 1e-30)
        spd = spd / max(float(np.max(spd)), 1e-12)
        cmf = np.array(
            [
                [0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954],
                [0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910],
                [0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876],
            ],
            dtype=np.float64,
        )
        wl8 = np.linspace(400, 700, 8)
        xyz = cmf @ np.interp(wl8, wl, spd)
        M = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ],
            dtype=np.float64,
        )
        rgb = np.maximum(M @ xyz, 0.0)
        return rgb / max(float(np.linalg.norm(rgb)), 1e-12)


# Default portrait priors for Phase 3 cohort (ISSA cheek medians, 400–700 nm).
# P2 = Liki (Indian) → South Asian ISSA median, not African.
DEFAULT_PARTICIPANT_PRIOR: dict[str, str] = {
    "P1": "issa_median_caucasian",
    "P2": "issa_median_south_asian",
}

# Three-term illuminant basis (K) — Zhou uses SVD daylight bank; Planck triple is a stable v1.
BASIS_CCT_K = (3000.0, 5500.0, 9000.0)
_EPS = 1e-12


@dataclass
class ScrAwbResult:
    skin_rgb_median: np.ndarray  # (3,)
    alpha: np.ndarray  # (3,)
    system_matrix: np.ndarray  # (3, 3)
    illuminant_rgb: np.ndarray  # (3,) relative, max=1
    ambient_cct_k: float
    prior_name: str
    basis_cct_k: Tuple[float, ...]
    residual_norm: float
    white_balanced_linear: np.ndarray  # (H, W, 3)


def _planck_spd_on_wl(wavelengths_nm: np.ndarray, cct_k: float) -> np.ndarray:
    """Relative Planck spectral power on ``wavelengths_nm`` (nm)."""
    wl_m = np.asarray(wavelengths_nm, dtype=np.float64) * 1e-9
    c2 = 1.438776877e-2
    t = max(float(cct_k), 500.0)
    spd = wl_m**-5 / (np.exp(c2 / (wl_m * t)) - 1.0 + 1e-30)
    return spd / max(float(np.max(spd)), _EPS)


def build_illuminant_basis(
    wavelengths_nm: np.ndarray,
    basis_cct_k: Sequence[float] = BASIS_CCT_K,
) -> np.ndarray:
    """(n_basis, n_wl) normalized Planck SPD samples."""
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    rows = []
    for cct in basis_cct_k:
        rows.append(_planck_spd_on_wl(wl, float(cct)))
    return np.stack(rows, axis=0)


def build_scr_system_matrix(
    spectral_sensitivity_rgb: np.ndarray,
    skin_reflectance: np.ndarray,
    illuminant_basis: np.ndarray,
) -> np.ndarray:
    """
    M[c, p] = sum_w S[c, w] * r[w] * E_p[w]  (discrete Zhou Eq. 6).
    """
    s = np.asarray(spectral_sensitivity_rgb, dtype=np.float64)
    r = np.asarray(skin_reflectance, dtype=np.float64)
    e = np.asarray(illuminant_basis, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError(f"spectral_sensitivity_rgb must be (N, 3), got {s.shape}")
    n_wl = s.shape[0]
    if r.shape != (n_wl,) or e.shape[1] != n_wl:
        raise ValueError("skin_reflectance and illuminant_basis length must match sensitivity rows")
    m = np.zeros((3, e.shape[0]), dtype=np.float64)
    for p in range(e.shape[0]):
        m[:, p] = (s * r[:, None] * e[p, :, None]).sum(axis=0)
    return m


def _solve_alpha_nnls(m: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Non-negative least squares for M @ alpha ≈ rgb."""
    rgb = np.asarray(rgb, dtype=np.float64).reshape(3)
    m = np.asarray(m, dtype=np.float64)
    try:
        from scipy.optimize import nnls

        alpha, residual = nnls(m, rgb)
        return alpha, float(residual)
    except ImportError:
        alpha, _, _, _ = np.linalg.lstsq(m, rgb, rcond=None)
        alpha = np.maximum(alpha, 0.0)
        if alpha.sum() > _EPS:
            alpha = alpha / alpha.sum()
        pred = m @ alpha
        return alpha, float(np.linalg.norm(pred - rgb))


def illuminant_rgb_from_alpha(
    spectral_sensitivity_rgb: np.ndarray,
    illuminant_basis: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Channel response to illuminant only (no skin reflectance)."""
    s = np.asarray(spectral_sensitivity_rgb, dtype=np.float64)
    e = np.asarray(illuminant_basis, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    spd = (a[:, None] * e).sum(axis=0)
    rgb = (s * spd[:, None]).sum(axis=0)
    rgb = np.maximum(rgb, _EPS)
    return rgb / max(float(np.max(rgb)), _EPS)


def estimate_ambient_cct_from_alpha(
    alpha: np.ndarray,
    basis_cct_k: Sequence[float] = BASIS_CCT_K,
) -> float:
    a = np.asarray(alpha, dtype=np.float64)
    ccts = np.asarray(basis_cct_k, dtype=np.float64)
    if a.sum() < _EPS:
        return float(ccts[len(ccts) // 2])
    return float(np.dot(a, ccts) / a.sum())


def median_skin_linear_rgb(
    linear_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Per-channel median on masked pixels (Zhou: robust skin aggregate)."""
    m = mask > 0
    if not np.any(m):
        return np.full(3, np.nan, dtype=np.float64)
    px = np.asarray(linear_rgb[m], dtype=np.float64)
    return np.median(px, axis=0)


def white_balance_diagonal(linear_rgb: np.ndarray, illuminant_rgb: np.ndarray) -> np.ndarray:
    """von Kries: divide by illuminant, normalize by 99th percentile (Lu-style display)."""
    e = np.maximum(np.asarray(illuminant_rgb, dtype=np.float64).reshape(1, 1, 3), _EPS)
    out = np.asarray(linear_rgb, dtype=np.float64) / e
    scale = float(np.percentile(out, 99.0)) + _EPS
    return np.clip(out / scale, 0.0, 1.0)


def resolve_prior_name(
    participant: str,
    subject_id: str,
    override: Optional[str] = None,
    priors_dir: Optional[Path] = None,
) -> str:
    if override:
        return override.strip()
    sid = (subject_id or "").strip().upper()
    if sid.startswith("P1"):
        return DEFAULT_PARTICIPANT_PRIOR["P1"]
    if sid.startswith("P2"):
        return DEFAULT_PARTICIPANT_PRIOR["P2"]
    part = (participant or "").strip()
    m = re.match(r"(?:participant\s*)?(\d+)\s*$", part, re.I)
    if m:
        key = f"P{int(m.group(1))}"
        if key in DEFAULT_PARTICIPANT_PRIOR:
            return DEFAULT_PARTICIPANT_PRIOR[key]
    return "issa_median_caucasian"


def estimate_scr_awb(
    noflash_linear: np.ndarray,
    skin_mask: np.ndarray,
    *,
    spectral_sensitivity_rgb: np.ndarray,
    wavelengths_nm: Sequence[float],
    skin_prior: SkinReflectancePrior,
    basis_cct_k: Sequence[float] = BASIS_CCT_K,
    known_ambient_cct_k: Optional[float] = None,
    known_ambient_duv: float = 0.0,
) -> ScrAwbResult:
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    s = np.asarray(spectral_sensitivity_rgb, dtype=np.float64)
    r = skin_prior.resample_to(wl)
    r = r / max(float(np.max(r)), _EPS)
    basis = build_illuminant_basis(wl, basis_cct_k)
    m = build_scr_system_matrix(s, r, basis)
    rgb_med = median_skin_linear_rgb(noflash_linear, skin_mask)
    if not np.all(np.isfinite(rgb_med)):
        raise ValueError("SCR-AWB: no finite skin median RGB")
    alpha, res = _solve_alpha_nnls(m, rgb_med)
    illum_solved = illuminant_rgb_from_alpha(s, basis, alpha)
    cct = estimate_ambient_cct_from_alpha(alpha, basis_cct_k)
    if known_ambient_cct_k is not None and float(known_ambient_cct_k) > 0.0:
        try:
            from src.lu2006_ambient import planck_rgb_from_cct_duv  # type: ignore

            illum = planck_rgb_from_cct_duv(float(known_ambient_cct_k), float(known_ambient_duv))
            cct = float(known_ambient_cct_k)
        except ImportError:
            illum = illum_solved
    else:
        illum = illum_solved
    wb = white_balance_diagonal(noflash_linear, illum)
    return ScrAwbResult(
        skin_rgb_median=rgb_med,
        alpha=alpha,
        system_matrix=m,
        illuminant_rgb=illum,
        ambient_cct_k=cct,
        prior_name=skin_prior.name,
        basis_cct_k=tuple(float(x) for x in basis_cct_k),
        residual_norm=res,
        white_balanced_linear=wb,
    )


def load_scr_awb_prior(
    name_or_path: str,
    priors_dir: Optional[Path] = None,
) -> SkinReflectancePrior:
    return load_prior(name_or_path, priors_dir=priors_dir)
