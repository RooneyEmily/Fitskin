"""export_landmark_json.py

Run this script ONCE locally (requires mediapipe + rawpy).
It processes every flash/no-flash DNG pair, runs MediaPipe Face Mesh,
extracts pixel-coordinate landmark polygons, and writes one JSON per trial to

    landmarks/<participant_id>_<trial_id>.json

The JSON files are then checked into git so the Colab notebook and iPhone app
can use them without requiring MediaPipe on-device.

Usage
-----
    python export_landmark_json.py

Edit RAW_DATASET_DIR and OUTPUT_DIR below if needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Import calibration raw reader that already lives in this repo
try:
    import physio_skin_lab_raw_pr250 as pr250
    USE_PR250 = True
except ImportError:
    pr250 = None
    USE_PR250 = False

import rawpy

from flash_noflash_face_roi import (
    CHEEK_LANDMARKS,
    CHEEK_R_LANDMARKS,
    FOREHEAD_LANDMARKS,
)

# ---------------------------------------------------------------------------
# Configuration — edit these if your paths differ
# ---------------------------------------------------------------------------
RAW_DATASET_DIR = Path("/home/mabl-main/Documents/RAW Dataset-20260531T233644Z-3-001/RAW Dataset")
OUTPUT_DIR      = Path(__file__).parent / "landmarks"

# All trials to process: (participant_id, trial_id, noflash_filename, flash_filename)
TRIALS = [
    ("participant1", "trial1", "Participant 1/Trial 1/IMG_0785_NoFlash.DNG", "Participant 1/Trial 1/IMG_0784_Flash.DNG"),
    ("participant1", "trial2", "Participant 1/Trial 2/IMG_0787_NoFlash.DNG", "Participant 1/Trial 2/IMG_0786_Flash.DNG"),
    ("participant1", "trial3", "Participant 1/Trial 3/IMG_0789_NoFlash.DNG", "Participant 1/Trial 3/IMG_0788_Flash.DNG"),
    ("participant2", "trial2", "Participant 2/Trial 2/IMG_0779_NoFlash.DNG", "Participant 2/Trial 2/IMG_0778_Flash.DNG"),
    ("participant2", "trial3", "Participant 2/Trial 3/IMG_0781_NoFlash.DNG", "Participant 2/Trial 3/IMG_0780_Flash.DNG"),
]
# ---------------------------------------------------------------------------


def load_dng_preview(path: Path, target_width: int = 1280) -> np.ndarray:
    """Demosaic DNG → uint8 sRGB at target_width for MediaPipe.

    Uses rawpy directly (camera white balance, half_size) then rescales.
    The result matches what a photo viewer shows for the DNG.
    """
    with rawpy.imread(str(path)) as raw:
        img_u8 = raw.postprocess(
            use_camera_wb=True,
            half_size=True,
            output_bps=8,
        )
    h, w = img_u8.shape[:2]
    if w > target_width:
        img_u8 = cv2.resize(img_u8, (target_width, int(h * target_width / w)))
    return img_u8


def run_mediapipe(rgb_u8: np.ndarray) -> list | None:
    """Run MediaPipe Face Mesh on a uint8 RGB image. Returns landmark list or None."""
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.3,
        refine_landmarks=True,
    ) as fm:
        result = fm.process(rgb_u8)
    if not result.multi_face_landmarks:
        return None
    return result.multi_face_landmarks[0].landmark


def landmark_px(lm, idx: int, h: int, w: int) -> list[int]:
    """Convert normalised landmark to [col, row] pixel coords."""
    return [int(lm[idx].x * w), int(lm[idx].y * h)]


def polygon_from_indices(lm, indices, h: int, w: int) -> list[list[int]]:
    """Convex hull of landmark indices as [[col,row], ...] list."""
    pts = np.array(
        [[int(lm[i].x * w), int(lm[i].y * h)] for i in indices],
        dtype=np.int32,
    )
    hull = cv2.convexHull(pts, clockwise=False)  # (N,1,2)
    return hull[:, 0, :].tolist()


def eye_centre(lm, eye_indices: list[int], h: int, w: int) -> list[int]:
    """Mean position of eye landmark indices → [col, row]."""
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in eye_indices])
    c   = pts.mean(axis=0)
    return [int(c[0]), int(c[1])]


# MediaPipe index constants
_LEFT_EYE_INDICES  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_NOSE_TIP          = 4
_MOUTH_CENTRE      = 13


def export_trial(
    participant_id: str,
    trial_id: str,
    noflash_rel: str,
    flash_rel: str,
) -> None:
    noflash_path = RAW_DATASET_DIR / noflash_rel
    if not noflash_path.exists():
        print(f"  SKIP — file not found: {noflash_path}")
        return

    print(f"  Loading {noflash_path.name} ...", end=" ", flush=True)
    rgb_u8 = load_dng_preview(noflash_path)
    h, w   = rgb_u8.shape[:2]
    print(f"{w}×{h}", end="  ")

    lm = run_mediapipe(rgb_u8)
    if lm is None:
        print("NO FACE DETECTED — skipped")
        return

    all_cheek = list(set(list(CHEEK_LANDMARKS) + list(CHEEK_R_LANDMARKS)))
    data = {
        "image":               noflash_path.name,
        "flash_image":         (RAW_DATASET_DIR / flash_rel).name,
        "participant":         participant_id,
        "trial":               trial_id,
        "image_hw":            [h, w],
        # Keypoints (single points)
        "left_eye":            eye_centre(lm, _LEFT_EYE_INDICES,  h, w),
        "right_eye":           eye_centre(lm, _RIGHT_EYE_INDICES, h, w),
        "nose":                landmark_px(lm, _NOSE_TIP,          h, w),
        "mouth":               landmark_px(lm, _MOUTH_CENTRE,      h, w),
        # Polygon ROIs (convex hulls as [col, row] lists)
        "left_cheek_polygon":  polygon_from_indices(lm, list(CHEEK_LANDMARKS),   h, w),
        "right_cheek_polygon": polygon_from_indices(lm, list(CHEEK_R_LANDMARKS), h, w),
        "forehead_polygon":    polygon_from_indices(lm, list(FOREHEAD_LANDMARKS), h, w),
    }

    out_path = OUTPUT_DIR / f"{participant_id}_{trial_id}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"→ {out_path.name}")

    # Quick verification plot
    _verify_plot(rgb_u8, data, out_path.with_suffix(".png"))


def _verify_plot(rgb_u8: np.ndarray, data: dict, out_png: Path) -> None:
    """Save a small verification image showing detected ROIs."""
    from helpers.skin_regions import make_cheek_mask, make_forehead_mask, draw_overlay
    img_lin = rgb_u8.astype(np.float64) / 255.0
    h, w    = img_lin.shape[:2]
    cheek   = make_cheek_mask(h, w, data)
    fore    = make_forehead_mask(h, w, data)
    vis_bgr = draw_overlay(img_lin, data, cheek_mask=cheek, forehead_mask=fore)
    vis_sm  = cv2.resize(vis_bgr, (w // 2, h // 2))
    cv2.imwrite(str(out_png), vis_sm)
    print(f"    Verification image → {out_png.name}")


def main() -> None:
    print(f"Exporting landmarks for {len(TRIALS)} trials → {OUTPUT_DIR}\n")
    for participant_id, trial_id, noflash_rel, flash_rel in TRIALS:
        print(f"[{participant_id} / {trial_id}]")
        export_trial(participant_id, trial_id, noflash_rel, flash_rel)
    print("\nDone. Commit the landmarks/ folder to git.")


if __name__ == "__main__":
    main()
