"""FitSkin-free cheek tone/chroma → ethnicity classifier for ROI sampling.

Trained only on demographics ethnicity labels + frozen preAWB+5500 cheek
features (L*, a*, b*, C*, ITA, L percentiles). Never sees FitSkin Lab.

Used by ``--l-sampling tone_chroma`` so specular_tone rules run without
reading demographics ethnicity at inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_NAMES = ("L", "a", "b", "C", "ITA", "Lp10", "Lp50", "Lp90")


def cheek_tone_features(lab: np.ndarray) -> np.ndarray:
    """8-D tone/chroma features from chroma-filtered cheek Lab pixels."""
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim != 2 or lab.shape[1] != 3 or len(lab) == 0:
        raise ValueError(f"lab must be (N,3), got {getattr(lab, 'shape', None)}")
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    C = np.hypot(a, b)
    mean_L = float(np.mean(L))
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_C = float(np.mean(C))
    ita = float(np.degrees(np.arctan2(mean_L - 50.0, mean_b)))
    lp10, lp50, lp90 = [float(x) for x in np.percentile(L, [10, 50, 90])]
    return np.array(
        [mean_L, mean_a, mean_b, mean_C, ita, lp10, lp50, lp90], dtype=np.float64
    )


@dataclass
class RoiToneClassifier:
    """Logistic ethnicity classifier on cheek tone features."""

    classes: List[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (n_classes, n_features)
    intercept: np.ndarray

    def predict(self, feat: np.ndarray) -> str:
        x = (np.asarray(feat, dtype=np.float64).reshape(-1) - self.mean) / self.scale
        logits = self.coef @ x + self.intercept
        return str(self.classes[int(np.argmax(logits))])

    def predict_proba(self, feat: np.ndarray) -> Dict[str, float]:
        x = (np.asarray(feat, dtype=np.float64).reshape(-1) - self.mean) / self.scale
        logits = self.coef @ x + self.intercept
        e = np.exp(logits - np.max(logits))
        p = e / np.sum(e)
        return {c: float(pi) for c, pi in zip(self.classes, p)}

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "roi_tone_logreg",
            "feature_names": list(FEATURE_NAMES),
            "classes": list(self.classes),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "label_source": "demographics_ethnicity",
            "fitskin_used": False,
            "notes": (
                "Predicts ethnicity from frozen-path cheek tone/chroma features; "
                "pair with specular_tone ROI sampling. Not a colorimetric model."
            ),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RoiToneClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            classes=[str(c) for c in payload["classes"]],
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coef=np.asarray(payload["coef"], dtype=np.float64),
            intercept=np.asarray(payload["intercept"], dtype=np.float64),
        )


def train_roi_tone_classifier(
    features: Sequence[np.ndarray],
    ethnicities: Sequence[str],
    *,
    max_iter: int = 4000,
    random_state: int = 0,
) -> RoiToneClassifier:
    """Train multinomial logistic regression (sklearn) → JSON-serializable weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.stack([np.asarray(f, dtype=np.float64) for f in features], axis=0)
    y = [str(e).strip() for e in ethnicities]
    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(Xs, y)
    classes = [str(c) for c in clf.classes_]
    return RoiToneClassifier(
        classes=classes,
        mean=np.asarray(sc.mean_, dtype=np.float64),
        scale=np.asarray(sc.scale_, dtype=np.float64),
        coef=np.asarray(clf.coef_, dtype=np.float64),
        intercept=np.asarray(clf.intercept_, dtype=np.float64),
    )


DEFAULT_TONES_PATH = (
    Path(__file__).resolve().parents[1]
    / "calibration"
    / "roi_tone_chroma"
    / "ethnicity_logreg.json"
)
