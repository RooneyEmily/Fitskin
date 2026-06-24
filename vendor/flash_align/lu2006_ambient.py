"""
Lu & Drew, CIC 2006 — ambient illuminant from flash / no-flash pairs.

Implements log-difference geometric-mean chromaticity (Eq. 8–10) and
nearest match to a Planckian reference locus (Eq. 13).

Deviations from the published pipeline (document in methods):
  - **Spectral sharpening** ``M`` is identity unless ``sharpening_matrix`` is supplied
    (Lu et al. train ``M`` per calibrated camera + ColorChecker; we do not guess ``M``).
  - **Flash CCT** defaults to **auto-estimate** from neutral-biased pure-flash pixels,
    with fallback ``FLASH_CCT_FALLBACK_K`` (5500 K). Fixed 5500 K is only a last-resort
    prior when the scene lacks usable flash-only neutrals — not a claim about iPhone flash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .align_pair import AlignResult

_EPS = 1e-6
# Wien constants; λ in metres in paper, we use nm → ek = -c2 / (λ_nm * 1e-9) scaled consistently
_LAMBDA_NM = np.array([600.0, 550.0, 450.0], dtype=np.float64)
_C2 = 1.438e7  # nm·K
_E_VEC = -_C2 / _LAMBDA_NM
_E_MEAN = float(_E_VEC.mean())
# Last-resort Planck flash model when auto-estimate fails (not measured per device).
FLASH_CCT_FALLBACK_K = 5500.0
_FLASH_CCT_K = FLASH_CCT_FALLBACK_K  # backward-compatible alias

# 2×3 projector onto plane orthogonal to u = (1,1,1)/√3 (Eq. 11–12)
_U_CHI = np.array(
    [
        [0.70710678, -0.70710678, 0.0],
        [0.40824829, 0.40824829, -0.81649658],
    ],
    dtype=np.float64,
)


@dataclass
class Lu2006Result:
    """Lu et al. CIC 2006 — illuminant estimate + white-balance variants (comparison path)."""

    ambient_cct_k: float
    ambient_cct_estimated_k: float
    ambient_cct_source: str  # "estimated" | "known_booth"
    ambient_duv: float
    flash_cct_k: float
    flash_cct_source: str  # "auto" | "fixed" | "fallback"
    spectral_sharpening: str  # "none" | "matrix"
    ambient_rgb_planck: np.ndarray  # Planck RGB at classified CCT (paper WB reference)
    ambient_rgb_reflectance: np.ndarray  # median A/F on flash (DiCarlo-style extra step)
    flash_rgb_linear: np.ndarray
    pure_flash_linear: np.ndarray
    chi_mean: np.ndarray
    chi_distance: float
    n_valid_pixels: int
    white_balanced_lu: np.ndarray  # divide by Planck illuminant at estimated CCT (paper-style)
    white_balanced_booth: Optional[np.ndarray]  # optional: WB by known booth CCT/Duv
    white_balanced_reflectance: np.ndarray  # divide by reflectance-inferred illuminant

    @property
    def ambient_rgb_linear(self) -> np.ndarray:
        """Alias: Planck illuminant used for paper-style WB."""
        return self.ambient_rgb_planck

    @property
    def white_balanced_linear(self) -> np.ndarray:
        """Default WB output = paper-style Lu illuminant normalization."""
        return self.white_balanced_lu


def _planck_rgb_linear(cct_k: float) -> np.ndarray:
    """Relative linear sRGB for a Planckian illuminant at cct_k."""
    try:
        import colour

        sd = colour.sd_blackbody(cct_k)
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        xyz = colour.sd_to_XYZ(sd, cmfs)
        rgb = colour.XYZ_to_RGB(
            xyz / np.maximum(xyz[1], 1e-8),
            colour.RGB_COLOURSPACES["sRGB"],
            illuminant=colour.RGB_COLOURSPACES["sRGB"].whitepoint,
            matrix_XYZ_to_RGB=colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )
        rgb = np.maximum(rgb, 1e-6)
        return rgb / np.linalg.norm(rgb)
    except Exception:
        pass
    # Fallback: correlate with D65 / warm / cool anchors
    t = float(np.clip(cct_k, 2000.0, 12000.0))
    warm = np.array([1.0, 0.85, 0.7], dtype=np.float64)
    cool = np.array([0.85, 0.95, 1.15], dtype=np.float64)
    w = (6500.0 - t) / (6500.0 - 3000.0)
    w = float(np.clip(w, 0.0, 1.0))
    rgb = w * warm + (1.0 - w) * cool
    return rgb / np.linalg.norm(rgb)


def planck_rgb_from_cct_duv(cct_k: float, duv: float = 0.0) -> np.ndarray:
    """
    Relative linear sRGB for a Planckian illuminant at ``cct_k`` with CIE Duv offset.

    ``duv`` is applied as a small shift in CIE 1960 u'v' (positive above the locus).
    Suitable for booth spectrometer metadata (e.g. 6546 K, Duv 0.0017).
    """
    try:
        import colour

        uv = np.asarray(colour.temperature.CCT_to_uv(float(cct_k)), dtype=np.float64).reshape(2)
        if abs(float(duv)) > 1e-15:
            uv = uv + np.array([0.0, float(duv)], dtype=np.float64)
        xy = colour.UCS_to_xy(colour.models.UV_to_UCS(uv))
        xyz = colour.xy_to_XYZ(xy)
        xyz = xyz / max(float(xyz[1]), 1e-8)
        rgb = colour.XYZ_to_RGB(
            xyz,
            colour.RGB_COLOURSPACES["sRGB"],
            illuminant=colour.RGB_COLOURSPACES["sRGB"].whitepoint,
            matrix_XYZ_to_RGB=colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )
        rgb = np.maximum(rgb, 1e-6)
        return rgb / np.linalg.norm(rgb)
    except Exception:
        return _planck_rgb_linear(cct_k)


def _reference_chi(cct_ambient_k: float, cct_flash_k: float = _FLASH_CCT_K) -> np.ndarray:
    """Synthetic gray-scene log-diff χ for a given ambient CCT (training-free reference)."""
    la = _planck_rgb_linear(cct_ambient_k)
    lf = _planck_rgb_linear(cct_flash_k)
    d = np.log(la + _EPS) - np.log(lf + _EPS)
    r = d - d.mean()
    return _U_CHI @ r


def _build_cct_table(cct_flash_k: float, step_k: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    temps = np.arange(2500, 10001, step_k, dtype=np.float64)
    chi_refs = np.stack([_reference_chi(t, cct_flash_k) for t in temps], axis=0)
    return temps, chi_refs


_CCT_TABLES: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}


def _cct_table(cct_flash_k: float) -> Tuple[np.ndarray, np.ndarray]:
    key = round(float(cct_flash_k), 1)
    if key not in _CCT_TABLES:
        _CCT_TABLES[key] = _build_cct_table(key)
    return _CCT_TABLES[key]


def apply_spectral_sharpening(
    rgb_linear: np.ndarray,
    sharpening_matrix: Optional[np.ndarray],
) -> Tuple[np.ndarray, str]:
    """
    Lu et al. preprocess: ``rgb' = M @ rgb`` per pixel.

    Default (``None`` or identity): **no sharpening** — required deviation unless a
    camera-specific ``M`` was measured with a ColorChecker training set.
    """
    if sharpening_matrix is None:
        return rgb_linear, "none"
    M = np.asarray(sharpening_matrix, dtype=np.float64).reshape(3, 3)
    if np.allclose(M, np.eye(3), rtol=0.0, atol=1e-12):
        return rgb_linear, "none"
    out = np.einsum("hwc,dc->hwd", np.asarray(rgb_linear, dtype=np.float64), M, optimize=True)
    return np.maximum(out, 0.0), "matrix"


def estimate_flash_cct_from_pure_flash(
    pure_flash: np.ndarray,
    *,
    luma_lo: float = 0.04,
    luma_hi: float = 0.92,
    max_chroma_ratio: float = 0.18,
    min_pixels: int = 80,
) -> Optional[float]:
    """
    Estimate flash Planck CCT from neutral-biased pixels in the pure-flash image.

    Uses high pure-flash luma + low channel spread (proxy for gray/white surfaces
    lit mainly by flash). Returns ``None`` if too few pixels qualify.
    """
    pf = np.asarray(pure_flash, dtype=np.float64)
    luma = 0.2126 * pf[..., 0] + 0.7152 * pf[..., 1] + 0.0722 * pf[..., 2]
    mx = pf.max(axis=-1)
    mn = pf.min(axis=-1)
    chroma_ratio = (mx - mn) / np.maximum(mx, _EPS)
    valid = (
        (luma > luma_lo)
        & (luma < luma_hi)
        & (chroma_ratio < max_chroma_ratio)
        & (np.max(pf, axis=-1) > 0.02)
    )
    if int(valid.sum()) < min_pixels:
        return None
    med = np.median(pf[valid], axis=0)
    med = med / max(float(np.max(med)), _EPS)
    temps = np.arange(2500, 10001, 250, dtype=np.float64)
    best_cct = float(temps[0])
    best_cos = -2.0
    for t in temps:
        ref = _planck_rgb_linear(float(t))
        ref = ref / max(float(np.linalg.norm(ref)), _EPS)
        cos = float(np.dot(med, ref) / (np.linalg.norm(med) + _EPS))
        if cos > best_cos:
            best_cos = cos
            best_cct = float(t)
    return best_cct


def resolve_flash_cct_k(
    pure_flash: np.ndarray,
    *,
    cct_flash_k: Optional[float],
    measured_cct_k: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Resolve flash CCT for the Lu reference locus.

    Priority: explicit ``cct_flash_k`` > 0 → ``fixed``; else ``measured_cct_k`` (e.g. MK350 SPD)
    → ``measured_spd``; else auto-estimate from pure-flash neutrals → ``auto``;
    else ``FLASH_CCT_FALLBACK_K`` → ``fallback``.
    """
    if cct_flash_k is not None and float(cct_flash_k) > 0.0:
        return float(cct_flash_k), "fixed"
    if measured_cct_k is not None and float(measured_cct_k) > 0.0:
        return float(measured_cct_k), "measured_spd"
    est = estimate_flash_cct_from_pure_flash(pure_flash)
    if est is not None:
        return est, "auto"
    return FLASH_CCT_FALLBACK_K, "fallback"


def resolve_flash_rgb_linear(
    flash_cct_k: float,
    *,
    flash_rgb_measured: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Flash illuminant in linear RGB: measured SPD integration or Planck at ``flash_cct_k``."""
    if flash_rgb_measured is not None:
        rgb = np.maximum(np.asarray(flash_rgb_measured, dtype=np.float64).reshape(3), _EPS)
        return rgb / max(float(np.linalg.norm(rgb)), _EPS)
    return _planck_rgb_linear(flash_cct_k)


def pure_flash_image(
    noflash: np.ndarray,
    flash: np.ndarray,
    *,
    nonnegative: bool = True,
) -> np.ndarray:
    f = flash - noflash
    if nonnegative:
        f = np.maximum(f, 0.0)
    return f


def log_diff_geom_chroma_per_pixel(
    noflash: np.ndarray,
    flash_pure: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Per-pixel χ (2,) vectors; returns (N,2) and count."""
    d = np.log(noflash + _EPS) - np.log(flash_pure + _EPS)
    r = d - d.mean(axis=-1, keepdims=True)
    chi = r @ _U_CHI.T
    if mask.ndim == 2:
        valid = mask & np.all(flash_pure > 0.02, axis=-1) & np.all(noflash > 0.02, axis=-1)
    else:
        valid = mask
    chi_v = chi[valid]
    return chi_v, int(valid.sum())


def white_balance_lu_illuminant(
    noflash_linear: np.ndarray,
    illuminant_rgb: np.ndarray,
) -> np.ndarray:
    """
    Paper-style diagonal WB (Lu et al. Sec. White Balance): scale channels by
    inverse of estimated ambient illuminant. Reference white is the illuminant
    vector (training used a ColorChecker white patch per illuminant cluster).
    """
    e = np.maximum(np.asarray(illuminant_rgb, dtype=np.float64).reshape(1, 1, 3), _EPS)
    e = e / np.median(e)
    wb = noflash_linear / e
    scale = float(np.percentile(wb, 99.0))
    return np.clip(wb / max(scale, _EPS), 0.0, 1.0)


def white_balance_reflectance(
    noflash_linear: np.ndarray,
    illuminant_rgb: np.ndarray,
) -> np.ndarray:
    """WB using reflectance-inferred ambient RGB (extension, not in Lu et al.)."""
    e = np.maximum(np.asarray(illuminant_rgb, dtype=np.float64).reshape(1, 1, 3), _EPS)
    e = e / np.median(e)
    wb = noflash_linear / e
    scale = float(np.percentile(wb, 99.0))
    return np.clip(wb / max(scale, _EPS), 0.0, 1.0)


def estimate_ambient_lu2006(
    align: AlignResult,
    *,
    cct_flash_k: Optional[float] = None,
    measured_flash_cct_k: Optional[float] = None,
    flash_rgb_measured: Optional[np.ndarray] = None,
    known_ambient_cct_k: Optional[float] = None,
    known_ambient_duv: float = 0.0,
    sharpening_matrix: Optional[np.ndarray] = None,
    max_sat: float = 0.98,
) -> Lu2006Result:
    a_raw = align.noflash_linear
    b_raw = align.flash_aligned_linear
    a, sharpen_tag = apply_spectral_sharpening(a_raw, sharpening_matrix)
    b, _ = apply_spectral_sharpening(b_raw, sharpening_matrix)
    f = pure_flash_image(a, b)

    flash_k, flash_src = resolve_flash_cct_k(
        f, cct_flash_k=cct_flash_k, measured_cct_k=measured_flash_cct_k
    )
    e_flash = resolve_flash_rgb_linear(flash_k, flash_rgb_measured=flash_rgb_measured)

    luma = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    mask = (luma > 0.04) & (luma < max_sat) & (np.max(f, axis=-1) > 0.02)

    chi_pix, n_valid = log_diff_geom_chroma_per_pixel(a, f, mask)
    if n_valid < 50:
        mask = np.ones(a.shape[:2], dtype=bool)
        chi_pix, n_valid = log_diff_geom_chroma_per_pixel(a, f, mask)

    chi_mean = np.median(chi_pix, axis=0)

    temps, chi_refs = _cct_table(flash_k)
    dists = np.linalg.norm(chi_refs - chi_mean[None, :], axis=1)
    j = int(np.argmin(dists))
    cct_est = float(temps[j])
    dist = float(dists[j])

    e_planck = _planck_rgb_linear(cct_est)

    wb_booth: Optional[np.ndarray] = None
    if known_ambient_cct_k is not None and float(known_ambient_cct_k) > 0.0:
        e_booth = planck_rgb_from_cct_duv(float(known_ambient_cct_k), float(known_ambient_duv))
        wb_booth = white_balance_lu_illuminant(a, e_booth)
        ambient_used = float(known_ambient_cct_k)
        ambient_src = "known_booth"
        ambient_duv = float(known_ambient_duv)
    else:
        ambient_used = cct_est
        ambient_src = "estimated"
        ambient_duv = 0.0

    # Reflectance from pure-flash; recover ambient RGB from no-flash (not in Lu WB section)
    rho = f / (e_flash.reshape(1, 1, 3) + _EPS)
    valid3 = mask[..., None] & (rho > 0.01) & (rho < 0.95)
    rho_med = np.median(rho[valid3[..., 0]], axis=0) if valid3.any() else np.median(rho, axis=(0, 1))
    e_refl = a / (rho_med.reshape(1, 1, 3) + _EPS)
    e_refl = np.median(e_refl[valid3[..., 0]], axis=0) if valid3.any() else e_refl
    e_refl = np.maximum(e_refl, _EPS)
    e_refl = e_refl / np.median(e_refl)

    wb_lu = white_balance_lu_illuminant(a, e_planck)
    wb_refl = white_balance_reflectance(a, e_refl)

    return Lu2006Result(
        ambient_cct_k=ambient_used,
        ambient_cct_estimated_k=cct_est,
        ambient_cct_source=ambient_src,
        ambient_duv=ambient_duv,
        flash_cct_k=flash_k,
        flash_cct_source=flash_src,
        spectral_sharpening=sharpen_tag,
        ambient_rgb_planck=e_planck,
        ambient_rgb_reflectance=e_refl,
        flash_rgb_linear=e_flash,
        pure_flash_linear=f,
        chi_mean=chi_mean,
        chi_distance=dist,
        n_valid_pixels=n_valid,
        white_balanced_lu=wb_lu,
        white_balanced_booth=wb_booth,
        white_balanced_reflectance=wb_refl,
    )
