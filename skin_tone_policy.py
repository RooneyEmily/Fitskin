"""
Skin-tone policy for chart CC / flash-no-flash pipelines.

Estimates a coarse **light** vs **dark** tier from uncorrected preview cheek L*
(no FitSkin, no chart matrix) and picks ROI + matrix fit settings that generalize
across the Pansor cohort better than a single fixed ROI.

Usage::

    from skin_tone_policy import resolve_chart_cc_policy, probe_skin_tone_tier

    tier, probe_L = probe_skin_tone_tier(preview_bgr, face_mesh)
    policy = resolve_chart_cc_policy(tier)  # roi, affine, label
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import cv2
import numpy as np

import physio_skin_lab_monk as psl

try:
    from chart_cc_fitskin_lib import cheek_mask_from_landmarks
except ImportError:
    from scripts.chart_cc_fitskin_lib import cheek_mask_from_landmarks

SkinToneTier = Literal["light", "dark", "auto"]
ResolvedTier = Literal["light", "dark"]

# Preview cheek L* below this → dark tier (Pansor: P2 ~35, P1 ~47–50).
DEFAULT_DARK_L_STAR_THRESHOLD = 42.0


@dataclass(frozen=True)
class ChartCcPolicy:
    tier: ResolvedTier
    roi: str
    affine: bool
    probe_cheek_L: float
    reason: str


def probe_skin_tone_tier(
    preview_bgr: np.ndarray,
    face_mesh: Any,
    *,
    dark_l_threshold: float = DEFAULT_DARK_L_STAR_THRESHOLD,
) -> Tuple[ResolvedTier, float]:
    """
  Estimate light vs dark from gray-WB preview cheek L* (D65 Lab via monk).

  Returns ``(tier, probe_L*)``. ``probe_L*`` is ``nan`` if face/mask fails.
    """
    h, w = preview_bgr.shape[:2]
    rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return "light", float("nan")
    lm = res.multi_face_landmarks[0].landmark
    mesh_mask, *_ = psl.build_skin_mask_from_mesh(
        h, w, lm, skin_triangulation="tessellation", exclusion_dilate_iod_fraction=0.12
    )
    cheek = cheek_mask_from_landmarks(h, w, lm, mesh_mask)
    L, a, b, npx, *_ = psl.mean_lab_masked(
        preview_bgr, cheek, l_star_trim_lo=0.05, l_star_trim_hi=0.05, min_chroma_ab=2.0
    )
    if npx <= 0 or not np.isfinite(L):
        return "light", float("nan")
    tier: ResolvedTier = "dark" if float(L) < float(dark_l_threshold) else "light"
    return tier, float(L)


def resolve_chart_cc_policy(
    tier: SkinToneTier,
    *,
    probe_L: float = float("nan"),
    dark_l_threshold: float = DEFAULT_DARK_L_STAR_THRESHOLD,
) -> ChartCcPolicy:
    """
    Map skin-tone tier → chart CC settings.

    **dark**  — mesh ROI + 3×4 affine (cheek hull overshoots on darker skin)
    **light** — cheek ROI + 3×3 lstsq (aligned with FitSkin cheek probe)
    """
    if tier == "auto":
        if not np.isfinite(probe_L):
            resolved: ResolvedTier = "light"
            reason = "auto→light (probe failed)"
        elif probe_L < dark_l_threshold:
            resolved = "dark"
            reason = f"auto→dark (probe L*={probe_L:.1f} < {dark_l_threshold:.0f})"
        else:
            resolved = "light"
            reason = f"auto→light (probe L*={probe_L:.1f} ≥ {dark_l_threshold:.0f})"
    else:
        resolved = tier
        reason = f"manual {tier}"

    if resolved == "dark":
        return ChartCcPolicy(
            tier="dark",
            roi="mesh",
            affine=True,
            probe_cheek_L=probe_L,
            reason=reason + "; mesh+affine",
        )
    return ChartCcPolicy(
        tier="light",
        roi="cheek",
        affine=False,
        probe_cheek_L=probe_L,
        reason=reason + "; cheek+3x3",
    )


def fnf_mesh_only_policy() -> dict:
    """Chart-free flash/no-flash: always mesh ROI (never cheek on dark skin)."""
    return {"cheek_roi": False, "reason": "FNF mesh ROI for all skin tones"}


def resolve_calibration_dir(tone_root: Path, tier: ResolvedTier) -> Path:
    """
    Pick ``tone_root/dark`` or ``tone_root/light`` if present, else ``tone_root``.
    """
    tone_root = tone_root.expanduser().resolve()
    sub = tone_root / tier
    if (sub / "iphone_calibration_bundle.json").is_file():
        return sub
    if (tone_root / "iphone_calibration_bundle.json").is_file():
        return tone_root
    return sub


def tier_from_manifest_row(row: dict) -> Optional[ResolvedTier]:
    """Heuristic light/dark from Pansor-style manifest columns."""
    person = str(row.get("person", "")).strip().lower()
    participant = str(row.get("participant", "")).strip().lower()
    sid = str(row.get("subject_id", "")).upper()
    if person in ("liki", "p2") or "participant 2" in participant or sid.startswith("P2"):
        return "dark"
    if person in ("emily", "p1") or "participant 1" in participant or sid.startswith("P1"):
        return "light"
    return None


def resolve_fnf_skin_tone_tier(
    skin_tone: SkinToneTier,
    row: dict,
    *,
    probe_L: float = float("nan"),
    dark_l_threshold: float = DEFAULT_DARK_L_STAR_THRESHOLD,
) -> Tuple[ResolvedTier, str]:
    """Pick calibration tier for chart-free FNF (mesh ROI always)."""
    if skin_tone in ("light", "dark"):
        return skin_tone, f"manual {skin_tone}"
    from_row = tier_from_manifest_row(row)
    if from_row is not None:
        return from_row, f"auto→{from_row} (manifest participant)"
    if np.isfinite(probe_L):
        tier: ResolvedTier = "dark" if probe_L < dark_l_threshold else "light"
        return tier, f"auto→{tier} (probe L*={probe_L:.1f})"
    return "light", "auto→light (fallback)"
