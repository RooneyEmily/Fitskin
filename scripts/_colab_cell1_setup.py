# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — SETUP
# ══════════════════════════════════════════════════════════════════════════════
# Install deps (IPython magics — Colab). Safe if already installed.
%pip install -q rawpy opencv-python-headless numpy matplotlib gdown

import json, sys, zipfile, subprocess, textwrap
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    %pip install -q torch torchvision

IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

REPO_URL = "https://github.com/RooneyEmily/Fitskin.git"


def _sh(cmd: str) -> None:
    """Run a shell command (works in Colab and local)."""
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=False)


if Path("Fitskin").is_dir():
    _sh("cd Fitskin && git pull --ff-only 2>/dev/null || true")
    REPO = Path("Fitskin").resolve()
elif (Path.cwd() / "pipeline" / "d65_fairface7_roi.py").is_file():
    REPO = Path.cwd().resolve()
else:
    _sh(f"git clone -q {REPO_URL}")
    REPO = Path("Fitskin").resolve()

# Mount Drive (needed for Parker WB-sweep zips in Cell 2b). Do NOT deep-scan MyDrive here
# (rglob over all of Drive can hang for minutes and look like the cell is stuck).
if IN_COLAB:
    if not Path("/content/drive/MyDrive").is_dir():
        print("Mounting Google Drive …")
        drive.mount("/content/drive")
    md = Path("/content/drive/MyDrive")
    if md.is_dir():
        print("Drive OK:", md)
        top = sorted(md.glob("Variable Lighting*"))[:8]
        if top:
            print("Top-level lighting folders (My Drive):")
            for h in top:
                print(" ", h)
        else:
            print(
                "NOTE: no 'Variable Lighting*' folder in My Drive root.\n"
                "  If the booth data is under **Shared with me**, Colab only sees it after you\n"
                "  **Add shortcut to Drive** (Drive web → right-click folder → Add shortcut → My Drive).\n"
                "  Or put flat zip folders in My Drive and set EXTRA_ZIP_DIRS in Cell 2b."
            )
        sd = Path("/content/drive/Shareddrives")
        if sd.is_dir() and any(sd.iterdir()):
            print("Shared drives mounted:", sd)
            for h in sorted(sd.glob("Variable Lighting*"))[:4]:
                print(" ", h)
    else:
        print("WARN: Drive not mounted — authorize when prompted, then re-run this cell.")

sys.path = [str(REPO)] + [p for p in sys.path if Path(p).resolve() != REPO]

MULTI_LAB = REPO / "calibration" / "multi_illuminant_lab_affine" / "multi_illuminant_lab_affine.json"
PIPE = REPO / "pipeline" / "d65_fairface7_roi.py"
SKIN_ROI = REPO / "pipeline" / "skin_roi.py"
ASSET_ZIP = REPO / "colab_assets" / "ringlight_best_stack.zip"

if ASSET_ZIP.is_file():
    print("Extracting colab_assets/ringlight_best_stack.zip …")
    with zipfile.ZipFile(ASSET_ZIP) as zf:
        zf.extractall(REPO)
elif (not MULTI_LAB.is_file()) or (not PIPE.is_file()):
    print("WARN: missing assets — push colab_assets/ringlight_best_stack.zip to the repo")

# Colab git tip may still import mediapipe via physio_skin_lab_monk in the eval script.
# Ensure chart-free forehead path uses inline Lab binning (no MediaPipe).
EVAL_PY = REPO / "scripts" / "evaluate_pansor20_chartfree_d65.py"
_TRIM_HELPERS = """
def _clip_skin_trim_q(q: float) -> float:
    if q <= 0.0:
        return 0.0
    return min(float(q), 0.45)


def _apply_channel_quantile_trim(
    sel: np.ndarray,
    channel: np.ndarray,
    trim_lo: float,
    trim_hi: float,
) -> np.ndarray:
    out = sel.copy()
    tlo = _clip_skin_trim_q(trim_lo)
    thi = _clip_skin_trim_q(trim_hi)
    if tlo > 0.0:
        out &= channel >= float(np.quantile(channel, tlo))
    if thi > 0.0:
        out &= channel <= float(np.quantile(channel, 1.0 - thi))
    return out
"""

if EVAL_PY.is_file():
    ev = EVAL_PY.read_text(encoding="utf-8")
    if "from physio_skin_lab_monk import" in ev or (
        "def _apply_channel_quantile_trim" not in ev and "apply_skin_lab_binning" in ev
    ):
        print("Patching scripts/evaluate_pansor20_chartfree_d65.py (drop mediapipe import) …")
        ev = ev.replace(
            "    from physio_skin_lab_monk import _apply_channel_quantile_trim\n", ""
        )
        if "def _clip_skin_trim_q" not in ev and "def apply_skin_lab_binning" in ev:
            ev = ev.replace(
                "def apply_skin_lab_binning(",
                _TRIM_HELPERS.strip() + "\n\n\ndef apply_skin_lab_binning(",
                1,
            )
        EVAL_PY.write_text(ev, encoding="utf-8")
    if "from physio_skin_lab_monk import" in EVAL_PY.read_text(encoding="utf-8"):
        raise RuntimeError(f"Failed to patch {EVAL_PY} — still imports physio_skin_lab_monk")
else:
    print("WARN: missing", EVAL_PY)

# Self-heal forehead∪cheek pool helper if git/assets still lack it.
_POOL_LINES = [
    "def apple_face_forehead_lab_pool_mask(",
    "    landmarks: dict,",
    "    dst_h: int,",
    "    dst_w: int,",
    "    *,",
    "    linear_rgb=None,",
    "):",
    '    """Forehead union cheek mask for specular_tone Lab pooling."""',
    "    forehead = apple_face_skin_roi_mask(",
    '        landmarks, dst_h, dst_w, roi="forehead", linear_rgb=linear_rgb',
    "    )",
    "    _, cheek = apple_face_cheek_masks(landmarks, dst_h, dst_w)",
    "    return (forehead | (cheek > 0)).astype(bool)",
    "",
]
_POOL_FN = "\n".join(_POOL_LINES)

src = SKIN_ROI.read_text(encoding="utf-8") if SKIN_ROI.is_file() else ""
if "def apple_face_forehead_lab_pool_mask" not in src:
    print("Patching pipeline/skin_roi.py with apple_face_forehead_lab_pool_mask …")
    if not SKIN_ROI.is_file():
        raise FileNotFoundError(f"Missing {SKIN_ROI}")
    SKIN_ROI.write_text(src.rstrip() + "\n\n" + _POOL_FN + "\n", encoding="utf-8")

CAL_DIR = REPO / "calibration" / "tier3_affine"
FAIRFACE_DIR = REPO / "calibration" / "fairface"
FAIRFACE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("/content/ringlight_best_stack_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("/content/ringlight_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TORCH_DIR = Path("/content/drive/MyDrive/Torch_meas")  # optional

FF7 = FAIRFACE_DIR / "res34_fair_align_multi_7_20190809.pt"
if not FF7.is_file():
    print("Downloading FairFace-7 weights (~82 MB)…")
    _sh(f'gdown 11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB -O "{FF7}"')
assert FF7.is_file(), "FairFace weights missing — check gdown / network"

# Drop stale modules so patched files are loaded in later cells
for mod in list(sys.modules):
    if (
        mod == "pipeline"
        or mod.startswith("pipeline.")
        or mod == "scripts"
        or mod.startswith("scripts.")
    ):
        del sys.modules[mod]

import torch
from pipeline.d65_fairface7_roi import D65FairFace7ROIPipeline, write_result_json
from pipeline.skin_roi import apple_face_forehead_lab_pool_mask  # verify import

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("REPO:", REPO)
print("tier3 affine:", (CAL_DIR / "camera_rgb_to_xyz_affine.npy").is_file())
print("multi-lab corrector:", MULTI_LAB.is_file())
print("n=84 eval JSON:", (REPO / "data" / "ring_light" / "eval_n84_by_wb_cell.json").is_file())
print("forehead pool mask: OK")

# ── Torch flash SPD (used by hybrid_deploy Lu estimate) ───────────────────────
import matplotlib.pyplot as plt
import numpy as np
from pipeline.illuminant_estimation import load_torch_prior, load_torch_prior_from_cal_bundle

_torch_measured = False
try:
    _tp = load_torch_prior(TORCH_DIR)
    _torch_measured = True
    _torch_label = f"MK350 measured — {TORCH_DIR}"
except FileNotFoundError:
    _tp = load_torch_prior_from_cal_bundle(CAL_DIR)
    _torch_label = f"Calibration bundle fallback — {CAL_DIR.name}/iphone_calibration_bundle.json"

print(f"\nTorch flash prior: {_torch_label}")
print(f"  CCT ≈ {_tp.torch_cct_k:.0f} K  |  files={_tp.files}")

_spd = np.asarray(_tp.mean_spd, dtype=float)
_spd = _spd / max(float(np.nanmax(_spd)), 1e-8)
_fig, _ax = plt.subplots(figsize=(9, 3.2))
_ax.plot(_tp.wavelengths_nm, _spd, color="#d84a2b", lw=2.5)
_ax.set_xlabel("Wavelength (nm)")
_ax.set_ylabel("Normalized SPD")
_title = "iPhone torch SPD — used in Lu / hybrid_deploy CAT"
if not _torch_measured:
    _title += "\n(upload Torch_meas/ on Drive for measured ESPD; bundle fallback shown)"
_ax.set_title(_title, fontsize=10)
_ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("Setup OK.")
