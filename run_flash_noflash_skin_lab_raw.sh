#!/usr/bin/env bash
# Flash/no-flash skin Lab on iPhone DNG pairs (auto-discovered under --data-root).
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA_ROOT:?Set DATA_ROOT to the RAW Dataset root (Participant */Trial */*.DNG)}"
OUT_DIR="${OUT_DIR:-./flash_noflash_dng_output}"
# Prefer checker-trained bundle; fall back to monochromator-only bundle
if [[ -f "./calibration/tier3_affine/camera_rgb_to_xyz_affine.npy" ]]; then
  CAL_DIR="${IPHONE_CALIBRATION:-./calibration/tier3_affine}"
elif [[ -f "./calibration/iphone17pro_trained/iphone_calibration_bundle.json" ]]; then
  CAL_DIR="${IPHONE_CALIBRATION:-./calibration/iphone17pro_trained}"
elif [[ -f "./calibration/iphone17pro_camera_color/iphone_calibration_bundle.json" ]]; then
  CAL_DIR="${IPHONE_CALIBRATION:-./calibration/iphone17pro_camera_color}"
else
  CAL_DIR="${IPHONE_CALIBRATION:-}"
fi
EXTRA_CALIB=()
SCR_AWB_ARGS=()
EXPOSURE_ANCHOR_ARGS=()
IMPROVED_ARGS=()
if [[ -n "${CAL_DIR}" && -f "${CAL_DIR}/iphone_calibration_bundle.json" ]]; then
  EXTRA_CALIB=(--iphone-calibration "$CAL_DIR")
  if [[ "${SCR_AWB:-1}" != "0" ]]; then
    SCR_AWB_ARGS=(--scr-awb)
  fi
  if [[ "${EXPOSURE_ANCHOR:-1}" != "0" ]]; then
    EXPOSURE_ANCHOR_ARGS=(--exposure-anchor-from-training)
  fi
fi
if [[ "${IMPROVED_PIPELINE:-1}" != "0" ]]; then
  IMPROVED_ARGS=(--cheek-roi --exposure-scale-skin-mask)
  if [[ "${KNOWN_AMBIENT_CCT_K:-6546}" != "0" ]]; then
    IMPROVED_ARGS+=(
      --known-ambient-cct-k "${KNOWN_AMBIENT_CCT_K:-6546}"
      --known-ambient-duv "${KNOWN_AMBIENT_DUV:-0.0017}"
    )
  fi
  if [[ "${REFLECTANCE_PRE_WB:-0}" == "booth" ]]; then
    IMPROVED_ARGS+=(--reflectance-pre-wb booth)
  fi
fi
BAG_CAT02_ARGS=()
if [[ "${BAG_CAT02:-auto}" != "off" ]]; then
  BAG_CAT02_ARGS=(--bag-cat02 "${BAG_CAT02:-auto}")
  if [[ "${MOBILE_SAM:-1}" != "0" ]]; then
    BAG_CAT02_ARGS+=(--mobile-sam)
  fi
fi
PRODUCTION_ARGS=()
if [[ "${PRODUCTION:-0}" != "0" ]]; then
  PRODUCTION_ARGS=(--production)
fi
RAW_WB_ARGS=(--raw-camera-wb)
python3 flash_no_flash_skin_lab.py \
  "${EXTRA_CALIB[@]}" \
  "${SCR_AWB_ARGS[@]}" \
  "${EXPOSURE_ANCHOR_ARGS[@]}" \
  "${IMPROVED_ARGS[@]}" \
  "${BAG_CAT02_ARGS[@]}" \
  "${PRODUCTION_ARGS[@]}" \
  "${RAW_WB_ARGS[@]}" \
  --data-root "$DATA_ROOT" \
  --input-mode dng \
  --out-dir "$OUT_DIR" \
  --exclude-trials P2_T1 \
  --write-overlays \
  "$@"
python3 plot_flash_noflash_skin_vs_mst.py \
  --csv "$OUT_DIR/flash_noflash_skin_lab.csv" \
  --out-dir "$OUT_DIR/figures"
# Phase 3 RAW extension (paper figures): same MST views as JPEG pilot
PHASE3_FIG_DIR="${PHASE3_FIG_DIR:-./figures/phase3_raw}"
python3 plot_flash_noflash_skin_vs_mst.py \
  --csv "$OUT_DIR/flash_noflash_skin_lab.csv" \
  --out-dir "$PHASE3_FIG_DIR" \
  --prefix phase3_raw \
  --title "Phase 3: FitSkin & flash/no-flash reflectance (DNG) vs Monk Skin Tone"
