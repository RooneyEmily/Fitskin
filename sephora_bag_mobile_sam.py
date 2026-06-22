"""
MobileSAM bag segmenter — drop-in replacement for sephora_bag_sam2.Sam2BagSegmenter.

Uses TinyViT image encoder (40 MB) with the standard SAM mask decoder; same
box + point-prompt API as original SAM so all existing prompt logic reuses cleanly.

Install: pip install mobile_sam   (already present in this environment)
Weights:  mobile_sam.pt           (39 MB, committed to the project root)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "mobile_sam.pt"


def mobile_sam_available() -> bool:
    try:
        import torch
        from mobile_sam import SamPredictor, sam_model_registry  # noqa: F401

        return bool(torch.cuda.is_available()) and DEFAULT_CHECKPOINT.is_file()
    except ImportError:
        return False


def _score_bag_mask(
    mask: np.ndarray,
    bag_box: Tuple[int, int, int, int],
    rgb_u01: np.ndarray,
) -> float:
    """Prefer masks that span the bag bbox vertically with alternating stripe structure."""
    x0, y0, x1, y1 = bag_box
    local = mask[y0:y1, x0:x1]
    if local.size == 0 or local.sum() < 80:
        return -1.0
    coverage = float(local.mean())
    if coverage < 0.18 or coverage > 0.95:
        return -1.0
    ys = np.where(local)[0]
    vspan = float(ys.max() - ys.min()) / max(y1 - y0, 1)
    if vspan < 0.60:
        return -1.0

    # Reward stripe alternation along the center column
    def _luma(rgb: np.ndarray) -> np.ndarray:
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    cx = local.shape[1] // 2
    half = max(4, local.shape[1] // 8)
    col_rgb = rgb_u01[y0:y1, x0 + cx - half : x0 + cx + half]
    col_msk = local[:, cx - half : cx + half]
    row_l = []
    for ri in range(col_rgb.shape[0]):
        m = col_msk[ri]
        if m.sum() < 2:
            continue
        row_l.append(float(_luma(col_rgb[ri][m]).mean()))
    if len(row_l) < 6:
        return coverage * vspan
    return coverage * vspan * (1.0 + 3.0 * float(np.std(np.diff(row_l))))


class MobileSamBagSegmenter:
    """
    Thin wrapper around ``mobile_sam.SamPredictor`` with the same interface as
    ``sephora_bag_sam2.Sam2BagSegmenter.predict_mask``.
    """

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT) -> None:
        import torch
        from mobile_sam import SamPredictor, sam_model_registry

        self._torch = torch
        model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        self.predictor = SamPredictor(model)

    def predict_mask(
        self,
        rgb_uint8: np.ndarray,
        box_xyxy: Tuple[int, int, int, int],
        *,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        rgb_u01: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return boolean HxW mask; picks best candidate by bag+stripe score."""
        import torch

        self.predictor.set_image(rgb_uint8)
        box = np.asarray(box_xyxy, dtype=np.float32)[None, :]

        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        ref = rgb_u01 if rgb_u01 is not None else (rgb_uint8.astype(np.float64) / 255.0)
        best_idx, best_score = 0, -1.0
        for i in range(masks.shape[0]):
            m = np.asarray(masks[i], dtype=bool)
            s = _score_bag_mask(m, box_xyxy, ref) + 0.05 * float(scores[i])
            if s > best_score:
                best_score, best_idx = s, i
        return np.asarray(masks[best_idx], dtype=bool)
