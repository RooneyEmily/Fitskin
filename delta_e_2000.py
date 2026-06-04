"""
CIEDE2000 (ΔE₀₀) — Sharma et al. / CIE 142-2001.

Ported to NumPy from the same structure as common MATLAB reference implementations
(arithmetic mean SL, SC, SH; hue angle in degrees; same G, T, RC, RT terms as typical ``deltaE00.m``
course handouts). Use for skin Lab vs MST Table I Lab (D65).

``delta_e_2000(lab_t, lab_r)`` accepts broadcastable ``(..., 3)`` L*, a*, b* and returns
the same leading shape of scalar ΔE₀₀ values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _deg2rad(d: np.ndarray) -> np.ndarray:
    return np.radians(d)


def _atan2deg(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Hue angle in degrees [0, 360), matching atan2(b, a) for Lab chroma plane."""
    return np.mod(np.degrees(np.arctan2(y, x)), 360.0)


def delta_e_2000(lab_t: np.ndarray, lab_r: np.ndarray) -> np.ndarray:
    """
    Vectorized ΔE₀₀. ``lab_t``, ``lab_r`` broadcastable to common shape ``(..., 3)`` with L*, a*, b*.

    Returns float array of shape ``np.broadcast_shapes(lab_t.shape[:-1], lab_r.shape[:-1])``.
    """
    Lt, at, bt = np.moveaxis(np.asarray(lab_t, dtype=np.float64), -1, 0)
    Lr, ar, br = np.moveaxis(np.asarray(lab_r, dtype=np.float64), -1, 0)

    C1 = np.sqrt(at * at + bt * bt)
    C2 = np.sqrt(ar * ar + br * br)
    mC = (C1 + C2) * 0.5
    G = 0.5 * (1.0 - np.sqrt(mC**7 / (mC**7 + 25.0**7)))

    a1p = at * (1.0 + G)
    a2p = ar * (1.0 + G)
    b1p = bt
    b2p = br
    C1p = np.sqrt(a1p * a1p + b1p * b1p)
    C2p = np.sqrt(a2p * a2p + b2p * b2p)

    h1p = _atan2deg(b1p, a1p)
    h2p = _atan2deg(b2p, a2p)

    dL = Lt - Lr
    dCp = C2p - C1p

    h1 = np.where((h1p < h2p) & (np.abs(h1p - h2p) > 180.0), h1p + 360.0, h1p)
    h2 = np.where((h1p > h2p) & (np.abs(h1p - h2p) > 180.0), h2p + 360.0, h2p)
    dhp = h2 - h1
    dHp = 2.0 * np.sqrt(np.maximum(C1p * C2p, 0.0)) * np.sin(_deg2rad(dhp * 0.5))

    mLp = (Lt + Lr) * 0.5
    mCp = (C1p + C2p) * 0.5
    mhp = (h1 + h2) * 0.5
    mhp = np.where(np.abs(h1 - h2) > 180.0, mhp + 180.0, mhp)
    mhp = np.mod(mhp, 360.0)

    T = (
        1.0
        - 0.17 * np.cos(_deg2rad(mhp - 30.0))
        + 0.24 * np.cos(_deg2rad(2.0 * mhp))
        + 0.32 * np.cos(_deg2rad(3.0 * mhp + 6.0))
        - 0.20 * np.cos(_deg2rad(4.0 * mhp - 63.0))
    )

    SL = 1.0 + (0.015 * (mLp - 50.0) ** 2) / np.sqrt(20.0 + (mLp - 50.0) ** 2)
    SC = 1.0 + 0.045 * mCp
    SH = 1.0 + 0.015 * mCp * T

    RC = 2.0 * np.sqrt(mCp**7 / (mCp**7 + 25.0**7))
    dtheta = 30.0 * np.exp(-(((mhp - 275.0) / 25.0) ** 2))
    RT = -np.sin(_deg2rad(2.0 * dtheta)) * RC

    kL = kC = kH = 1.0
    v = (
        (dL / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return np.sqrt(np.maximum(v, 0.0))


def mst_de2000_row(
    lab_skin: np.ndarray,
    mst_lab_10x3: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    """
    ``lab_skin`` shape (3,) and ``mst_lab_10x3`` shape (10, 3) rows = monk 1..10.

    Returns ``(de10, nearest_monk_1based, min_de)``.
    """
    skin = np.asarray(lab_skin, dtype=np.float64).reshape(1, 3)
    mst = np.asarray(mst_lab_10x3, dtype=np.float64)
    if mst.shape != (10, 3):
        raise ValueError(f"mst_lab_10x3 must be (10,3), got {mst.shape}")
    de = np.asarray(delta_e_2000(skin, mst), dtype=np.float64).reshape(-1)
    if de.size != 10:
        raise ValueError(f"expected 10 ΔE values, got {de.size}")
    j = int(np.argmin(de))
    return de, j + 1, float(de[j])


def load_mst_lab_matrix_10x3(path: Path) -> np.ndarray:
    """Rows monk 1..10, columns L*, a*, b* from ``mst_reference_cheng2024_table1.csv`` style."""
    import csv

    path = Path(path).expanduser().resolve()
    by_m: dict[int, Tuple[float, float, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mk = int(float(row["monk"]))
            by_m[mk] = (float(row["L"]), float(row["a"]), float(row["b"]))
    for i in range(1, 11):
        if i not in by_m:
            raise ValueError(f"MST CSV missing monk={i}")
    return np.array([by_m[i] for i in range(1, 11)], dtype=np.float64)


def de2000_csv_header() -> list[str]:
    return [f"de2000_mst{k:02d}" for k in range(1, 11)] + [
        "de2000_nearest_mst",
        "de2000_min",
    ]


def de2000_csv_values(de10: np.ndarray, nearest: int, dmin: float) -> list[float]:
    return [float(de10[i]) for i in range(10)] + [float(nearest), float(dmin)]
