"""FairFace race classifier for deployment (no demographics ethnicity labels).

Uses pretrained ResNet-34 from https://github.com/dchen236/FairFace
on an sRGB face crop. Output is mapped to Pansor demographics labels so
existing ``specular_tone`` ROI rules can run without race metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAIRFACE_DIR = ROOT / "calibration" / "fairface"

RACE4 = ("White", "Black", "Asian", "Indian")
RACE7 = (
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
)

# Map FairFace labels → Pansor demographics ethnicity (for specular_tone).
FAIRFACE_TO_PANSOR = {
    "White": "White",
    "Black": "Black",
    "Asian": "Asian",
    "Indian": "Indian",
    "East Asian": "Asian",
    "Southeast Asian": "Asian",
    "Middle Eastern": "Iranian",
    # Intermediate / ambiguous → medium-skin Asian ROI branch
    "Latino_Hispanic": "Asian",
}


def _load_resnet34_fairface(weights_path: Path, device: torch.device) -> nn.Module:
    # Match FairFace predict.py: ResNet-34 with 18-D head (race + gender + age).
    try:
        model = torchvision.models.resnet34(weights=None)
    except TypeError:
        model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 18)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@dataclass
class FairFacePredictor:
    mode: str  # "4" or "7"
    model: nn.Module
    device: torch.device
    transform: Any

    @classmethod
    def load(
        cls,
        *,
        mode: str = "7",
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> "FairFacePredictor":
        mode = str(mode)
        if mode not in ("4", "7"):
            raise ValueError("mode must be '4' or '7'")
        wdir = Path(weights_dir or DEFAULT_FAIRFACE_DIR)
        if mode == "4":
            path = wdir / "fairface_alldata_4race_20191111.pt"
        else:
            # Prefer newer all-data 7-class name if present; else align multi-7.
            cand = [
                wdir / "fairface_alldata_20191111.pt",
                wdir / "res34_fair_align_multi_7_20190809.pt",
            ]
            path = next((p for p in cand if p.is_file()), cand[-1])
        if not path.is_file():
            raise FileNotFoundError(
                f"FairFace weights missing: {path}. "
                "Download from FairFace Google Drive into calibration/fairface/."
            )
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = _load_resnet34_fairface(path, dev)
        trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return cls(mode=mode, model=model, device=dev, transform=trans)

    @torch.inference_mode()
    def predict_rgb(self, rgb_u8: np.ndarray) -> Dict[str, Any]:
        """Predict from HxWx3 uint8 RGB face crop (FairFace-aligned style)."""
        if rgb_u8.dtype != np.uint8:
            raise TypeError("rgb_u8 must be uint8")
        if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {rgb_u8.shape}")
        x = self.transform(rgb_u8)
        x = x.view(1, 3, 224, 224).to(self.device)
        outputs = self.model(x).detach().cpu().numpy().squeeze()
        n_race = 4 if self.mode == "4" else 7
        race_logits = outputs[:n_race]
        e = np.exp(race_logits - np.max(race_logits))
        race_prob = e / np.sum(e)
        idx = int(np.argmax(race_prob))
        labels = RACE4 if self.mode == "4" else RACE7
        fair_label = labels[idx]
        pansor = FAIRFACE_TO_PANSOR.get(fair_label, "Asian")
        return {
            "fairface_label": fair_label,
            "predicted_ethnicity": pansor,
            "confidence": float(race_prob[idx]),
            "race_probs": {labels[i]: float(race_prob[i]) for i in range(n_race)},
            "mode": self.mode,
        }


def face_rgb_crop_from_landmarks(
    preview_bgr: np.ndarray,
    landmarks: dict,
    *,
    padding: float = 0.35,
    out_size: int = 224,
) -> np.ndarray:
    """Axis-aligned face crop from Apple Vision landmarks → RGB uint8 square.

    Approximation of FairFace's dlib face chip (no 5-point alignment).
    """
    h, w = preview_bgr.shape[:2]
    faces = landmarks.get("faces") or []
    if not faces:
        raise RuntimeError("No faces in landmark JSON")
    regions = {
        str(r.get("name")): r for r in (faces[0].get("regions") or []) if r.get("name")
    }
    isize = landmarks.get("imageSize") or {}
    src_w = int(isize.get("width") or w)
    src_h = int(isize.get("height") or h)
    origin_ll = "lower" in str(landmarks.get("coordinateOrigin") or "lowerLeft").lower()

    def region_xy(name: str) -> np.ndarray:
        region = regions.get(name)
        if not region:
            return np.zeros((0, 2), dtype=np.float64)
        pts = region.get("imagePoints")
        if pts:
            xy = np.array([[float(p["x"]), float(p["y"])] for p in pts], dtype=np.float64)
        else:
            npts = region.get("normalizedImagePoints") or []
            if not npts:
                return np.zeros((0, 2), dtype=np.float64)
            xy = np.array(
                [[float(p["x"]) * src_w, float(p["y"]) * src_h] for p in npts],
                dtype=np.float64,
            )
        if origin_ll:
            xy[:, 1] = float(src_h) - xy[:, 1]
        if src_w != w or src_h != h:
            xy[:, 0] *= w / float(src_w)
            xy[:, 1] *= h / float(src_h)
        return xy

    parts = []
    for name in ("faceContour", "leftEyebrow", "rightEyebrow"):
        xy = region_xy(name)
        if len(xy):
            parts.append(xy)
    if not parts:
        xy = region_xy("allPoints")
        if len(xy):
            parts.append(xy)
    if not parts:
        for name, region in regions.items():
            xy = region_xy(str(name))
            if len(xy):
                parts.append(xy)
                break
    if not parts:
        raise RuntimeError("Insufficient landmark points for face crop")
    hull_pts = np.vstack(parts)
    x0, y0 = float(hull_pts[:, 0].min()), float(hull_pts[:, 1].min())
    x1, y1 = float(hull_pts[:, 0].max()), float(hull_pts[:, 1].max())
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    pad_x, pad_y = padding * bw, padding * bh
    x0, y0 = x0 - pad_x, y0 - pad_y
    x1, y1 = x1 + pad_x, y1 + pad_y
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0)
    half = 0.5 * side
    x0s = int(np.clip(cx - half, 0, w - 1))
    y0s = int(np.clip(cy - half, 0, h - 1))
    x1s = int(np.clip(cx + half, 0, w))
    y1s = int(np.clip(cy + half, 0, h))
    crop = preview_bgr[y0s:y1s, x0s:x1s]
    if crop.size == 0:
        raise RuntimeError("Empty face crop")
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
