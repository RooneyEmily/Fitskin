"""
Residual Fourier MLP camera-RGB → D65 XYZ projector.

Inspired by Color Pass-Through's compact pixel-wise projector, adapted to
Fitskin's camera-only reflectance → XYZ objective. The affine tier3 baseline
is frozen; the network predicts a bounded XYZ residual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # inference-only environments
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass(frozen=True)
class ColorProjectorSettings:
    input_dim: int = 4  # RGB + green context
    num_frequencies: int = 4
    hidden_dim: int = 64
    hidden_layers: int = 2
    use_fourier: bool = True
    residual_scale: float = 0.15  # tanh * scale bound on XYZ residual


def fourier_encode_numpy(x: np.ndarray, num_frequencies: int) -> np.ndarray:
    """Map ``(..., D)`` → ``(..., D * (1 + 2*F))`` with sin/cos positional encoding."""
    x = np.asarray(x, dtype=np.float64)
    if num_frequencies <= 0:
        return x
    freqs = (2.0 ** np.arange(num_frequencies, dtype=np.float64)) * np.pi
    angles = x[..., :, None] * freqs[None, :]
    sin = np.sin(angles)
    cos = np.cos(angles)
    enc = np.concatenate([x[..., None], sin, cos], axis=-1)
    return enc.reshape(*x.shape[:-1], -1)


def affine_rgb_to_xyz(rgb: np.ndarray, affine_4x3: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    M = np.asarray(affine_4x3, dtype=np.float64)
    flat = rgb.reshape(-1, 3)
    aug = np.column_stack([flat, np.ones(len(flat), dtype=np.float64)])
    xyz = aug @ M
    return xyz.reshape(rgb.shape)


def _mlp_forward_numpy(
    features: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
) -> np.ndarray:
    h = features
    last = len(weights) - 1
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W.T + b
        if i < last:
            h = np.maximum(h, 0.0)
    return h


@dataclass
class ColorProjectorArtifact:
    settings: ColorProjectorSettings
    affine_4x3: np.ndarray
    weights: List[np.ndarray]  # each (out, in)
    biases: List[np.ndarray]
    rgb_mean: np.ndarray
    rgb_std: np.ndarray
    meta: Dict[str, Any]

    def encode_features(
        self, rgb: np.ndarray, green_context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        flat = rgb.reshape(-1, 3)
        if green_context is None:
            gctx = flat[:, 1:2]
        else:
            gctx = np.asarray(green_context, dtype=np.float64).reshape(-1, 1)
        feat = np.concatenate([flat, gctx], axis=1)
        feat = (feat - self.rgb_mean) / np.maximum(self.rgb_std, 1e-6)
        if self.settings.use_fourier:
            return fourier_encode_numpy(feat, self.settings.num_frequencies)
        return feat

    def residual_xyz(
        self, rgb: np.ndarray, green_context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        feat = self.encode_features(rgb, green_context)
        raw = _mlp_forward_numpy(feat, self.weights, self.biases)
        return np.tanh(raw) * float(self.settings.residual_scale)

    def __call__(
        self, rgb: np.ndarray, green_context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        base = affine_rgb_to_xyz(rgb, self.affine_4x3)
        resid = self.residual_xyz(rgb, green_context).reshape(base.shape)
        return base + resid


def apply_color_projector_rgb(
    rgb: np.ndarray,
    artifact: ColorProjectorArtifact,
    *,
    use_green_blur: bool = True,
) -> np.ndarray:
    """Apply projector to ``(H,W,3)`` or ``(N,3)`` linear camera RGB."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim == 2 and rgb.shape[1] == 3:
        return artifact(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) or (N,3), got {rgb.shape}")
    green_context = None
    if use_green_blur and artifact.settings.input_dim >= 4:
        try:
            import cv2

            g = rgb[:, :, 1]
            green_context = cv2.blur(g, (3, 3)).reshape(-1, 1)
        except Exception:
            green_context = rgb[:, :, 1].reshape(-1, 1)
    xyz = artifact(rgb, green_context)
    return xyz.reshape(rgb.shape)


def save_color_projector_artifact(path: Path, artifact: ColorProjectorArtifact) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {
        "affine_4x3": np.asarray(artifact.affine_4x3, dtype=np.float64),
        "rgb_mean": np.asarray(artifact.rgb_mean, dtype=np.float64),
        "rgb_std": np.asarray(artifact.rgb_std, dtype=np.float64),
        "n_layers": np.asarray([len(artifact.weights)], dtype=np.int32),
    }
    for i, (w, b) in enumerate(zip(artifact.weights, artifact.biases)):
        arrays[f"W{i}"] = np.asarray(w, dtype=np.float64)
        arrays[f"b{i}"] = np.asarray(b, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"settings": asdict(artifact.settings), "meta": artifact.meta}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def load_color_projector_artifact(path: Path) -> ColorProjectorArtifact:
    import json

    path = Path(path)
    data = np.load(path)
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    settings = ColorProjectorSettings(**payload["settings"])
    n = int(data["n_layers"][0])
    weights = [np.asarray(data[f"W{i}"], dtype=np.float64) for i in range(n)]
    biases = [np.asarray(data[f"b{i}"], dtype=np.float64) for i in range(n)]
    return ColorProjectorArtifact(
        settings=settings,
        affine_4x3=np.asarray(data["affine_4x3"], dtype=np.float64),
        weights=weights,
        biases=biases,
        rgb_mean=np.asarray(data["rgb_mean"], dtype=np.float64),
        rgb_std=np.asarray(data["rgb_std"], dtype=np.float64),
        meta=payload.get("meta") or {},
    )


if torch is not None:

    class ResidualColorProjector(nn.Module):
        def __init__(self, settings: ColorProjectorSettings):
            super().__init__()
            self.settings = settings
            in_dim = settings.input_dim
            if settings.use_fourier:
                in_dim = settings.input_dim * (1 + 2 * settings.num_frequencies)
            layers: List[nn.Module] = []
            prev = in_dim
            for _ in range(settings.hidden_layers):
                layers.append(nn.Linear(prev, settings.hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                prev = settings.hidden_dim
            layers.append(nn.Linear(prev, 3))
            self.mlp = nn.Sequential(*layers)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            raw = self.mlp(features)
            return torch.tanh(raw) * float(self.settings.residual_scale)

        def export_numpy_layers(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
            weights: List[np.ndarray] = []
            biases: List[np.ndarray] = []
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    weights.append(module.weight.detach().cpu().numpy().astype(np.float64))
                    biases.append(module.bias.detach().cpu().numpy().astype(np.float64))
            return weights, biases
