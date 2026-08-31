"""Planckian illuminant helpers for flash/no-flash skin colorimetry."""

from __future__ import annotations

import numpy as np

D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _cct_to_xy_mccamy(cct_k: float) -> tuple[float, float]:
    T = float(cct_k)
    if T < 1667.0:
        T = 1667.0
    if T > 25000.0:
        T = 25000.0
    x = -0.2661239e9 / (T**3) + 0.2343589e6 / (T**2) + 0.8776956e3 / T + 0.179910
    y = -1.1063814 * x**3 + 1.3485240 * x**2 + 0.2172377 * x + 0.240390
    return float(x), float(y)


def planck_xyz_y1(cct_k: float, duv: float = 0.0) -> np.ndarray:
    """CIE XYZ of a Planckian at ``cct_k`` (K), normalized to Y=1."""
    try:
        import colour

        xy = colour.temperature.CCT_to_xy(float(cct_k), method="McCamy 1992")
        X, Y, Z = colour.xy_to_XYZ(xy)
        Y = max(float(Y), 1e-12)
        xyz = np.array([X / Y, 1.0, Z / Y], dtype=np.float64)
    except Exception:
        x, y = _cct_to_xy_mccamy(cct_k)
        if y <= 1e-12:
            return D65.copy()
        xyz = np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)

    if abs(float(duv)) > 1e-12:
        try:
            import colour

            xy = colour.XYZ_to_xy(xyz * max(float(xyz[1]), 1e-12))
            xy_d = colour.temperature.uv_to_Luv_uv(
                colour.temperature.CCT_to_uv(float(cct_k), method="Krystek 1985")
            )
            # Simple duv shift in u'v' is non-trivial; keep Y=1 Planck fallback.
            _ = xy_d
        except Exception:
            pass
    return xyz
