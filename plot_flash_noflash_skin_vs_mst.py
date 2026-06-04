#!/usr/bin/env python3
"""
Plot FitSkin cheek Lab + flash/no-flash skin Lab vs Cheng MST Table I.

Same three views as ``plot_fitskin_scanner_vs_mst.py``: a*–b*, L*–b*, L*–C* (C* = hypot(a*, b*)).

Reads ``flash_noflash_skin_lab.csv``; plots only:
  - **FitSkin cheek** (scanner reference, diamonds)
  - **Flash/no-flash skin** (geometric-mean reflectance from the image pair, circles)

Marker **color**: blue = Participant 1, red = Participant 2 (legend explains both).

Writes ``flash_noflash_vs_mst_{ab,Lb,LC}.png`` and ``flash_noflash_reflectance_de00_by_trial.png``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from plot_skin_lab_vs_mst_reference import (
    _annotate_label_contrast,
    _axis_max_with_padding,
    _discrete_mst_colorbar,
    _mst_segment_linecollection,
    load_mst_lab_csv,
    resolve_mst_display_rgb,
)

ROOT = Path(__file__).resolve().parent

PARTICIPANT_COLORS = {"1": "#2166ac", "2": "#b2182b"}
DEFAULT_FIGURE_TITLE = (
    "FitSkin & flash/no-flash skin vs Monk Skin Tone (Cheng Table I)"
)
DEFAULT_PREFIX = "flash_noflash"
MARKER_SIZE = 52.0


def _pid_from_subject(subject_id: str) -> str:
    if subject_id.startswith("P1"):
        return "1"
    if subject_id.startswith("P2"):
        return "2"
    return subject_id


def load_flash_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def _lab_from_row(row: Dict[str, str], prefix: str) -> Tuple[float, float, float]:
    if prefix == "fitskin_cheek":
        L = float(row["fitskin_cheek_L"])
        a = float(row["fitskin_cheek_a"])
        b = float(row["fitskin_cheek_b"])
    else:
        L = float(row[f"{prefix}_L"])
        a = float(row[f"{prefix}_a"])
        b = float(row[f"{prefix}_b"])
    return L, a, b


def _xy(L: float, a: float, b: float, xkey: str, ykey: str) -> Tuple[float, float]:
    if xkey == "L":
        x = L
    elif xkey == "a":
        x = a
    elif xkey == "b":
        x = b
    elif xkey == "C":
        x = float(np.hypot(a, b))
    else:
        raise ValueError(xkey)
    if ykey == "L":
        y = L
    elif ykey == "a":
        y = a
    elif ykey == "b":
        y = b
    elif ykey == "C":
        y = float(np.hypot(a, b))
    else:
        raise ValueError(ykey)
    return x, y


def _draw_mst_locus(
    ax,
    mst_x: np.ndarray,
    mst_y: np.ndarray,
    swatch_rgb: np.ndarray,
    mk_idx: np.ndarray,
) -> None:
    ax.add_collection(_mst_segment_linecollection(mst_x, mst_y, swatch_rgb))
    ax.scatter(
        mst_x,
        mst_y,
        s=145,
        c=swatch_rgb,
        edgecolors="k",
        linewidths=1.05,
        zorder=4,
    )
    for i in range(10):
        rgb = swatch_rgb[i]
        ax.annotate(
            str(int(mk_idx[i])),
            (mst_x[i], mst_y[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=11,
            fontweight="bold",
            color=_annotate_label_contrast(rgb),
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=rgb,
                edgecolor="k",
                linewidth=1.0,
            ),
            zorder=5,
        )


def _scatter_fitskin_and_flash(
    ax,
    rows: List[Dict[str, str]],
    xkey: str,
    ykey: str,
) -> None:
    """FitSkin diamonds + flash/no-flash reflectance circles; color = participant."""
    for prefix, marker in (("fitskin_cheek", "D"), ("reflectance", "o")):
        for row in rows:
            sid = row["subject_id"]
            pid = _pid_from_subject(sid)
            color = PARTICIPANT_COLORS.get(pid, "0.35")
            L, a, b = _lab_from_row(row, prefix)
            x, y = _xy(L, a, b, xkey, ykey)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            size = MARKER_SIZE * (1.12 if marker == "D" else 1.0)
            ax.scatter(
                x,
                y,
                s=size,
                c=color,
                marker=marker,
                edgecolors="k",
                linewidths=0.55,
                zorder=6,
            )
            ax.annotate(
                sid,
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="0.08",
                path_effects=[mpe.withStroke(linewidth=2.0, foreground="white")],
                zorder=7,
            )


def _add_legend(ax) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markersize=9,
            markerfacecolor="0.82",
            markeredgecolor="k",
            label="FitSkin cheek",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=9,
            markerfacecolor="0.82",
            markeredgecolor="k",
            label="Flash/no-flash skin",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=9,
            markerfacecolor=PARTICIPANT_COLORS["1"],
            markeredgecolor="k",
            label="Participant 1",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=9,
            markerfacecolor=PARTICIPANT_COLORS["2"],
            markeredgecolor="k",
            label="Participant 2",
        ),
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.92, fontsize=9)


def _channel(rows: List[Dict[str, str]], prefix: str, key: str) -> np.ndarray:
    out = []
    for row in rows:
        L, a, b = _lab_from_row(row, prefix)
        if key == "L":
            out.append(L)
        elif key == "a":
            out.append(a)
        elif key == "b":
            out.append(b)
        elif key == "C":
            out.append(float(np.hypot(a, b)))
    return np.array(out, dtype=np.float64)


def plot_ab(
    out_path: Path,
    rows: List[Dict[str, str]],
    mst_a: np.ndarray,
    mst_b: np.ndarray,
    swatch_rgb: np.ndarray,
    mk_idx: np.ndarray,
    *,
    figure_title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    _draw_mst_locus(ax, mst_a, mst_b, swatch_rgb, mk_idx)
    _scatter_fitskin_and_flash(ax, rows, "a", "b")
    pa = np.concatenate(
        [_channel(rows, "fitskin_cheek", "a"), _channel(rows, "reflectance", "a")]
    )
    pb = np.concatenate(
        [_channel(rows, "fitskin_cheek", "b"), _channel(rows, "reflectance", "b")]
    )
    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_title(figure_title)
    ax.set_xlim(0.0, max(20.0, float(np.nanmax(pa)) + 2.0))
    ax.set_ylim(0.0, _axis_max_with_padding(mst_b, pb))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_Lb(
    out_path: Path,
    rows: List[Dict[str, str]],
    mst_L: np.ndarray,
    mst_b: np.ndarray,
    swatch_rgb: np.ndarray,
    mk_idx: np.ndarray,
    *,
    figure_title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    _draw_mst_locus(ax, mst_L, mst_b, swatch_rgb, mk_idx)
    _scatter_fitskin_and_flash(ax, rows, "L", "b")
    pL = np.concatenate(
        [_channel(rows, "fitskin_cheek", "L"), _channel(rows, "reflectance", "L")]
    )
    pb = np.concatenate(
        [_channel(rows, "fitskin_cheek", "b"), _channel(rows, "reflectance", "b")]
    )
    ax.set_xlabel("L*")
    ax.set_ylabel("b*")
    ax.set_title(figure_title)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    lo_L = min(float(np.min(mst_L)), float(np.min(pL))) - 5.0
    hi_L = max(float(np.max(mst_L)), float(np.max(pL))) + 5.0
    lo_b = min(0.0, float(np.min(mst_b)), float(np.min(pb))) - 1.0
    hi_b = max(float(np.max(mst_b)), float(np.max(pb))) + 2.0
    ax.set_xlim(lo_L, hi_L)
    ax.set_ylim(lo_b, hi_b)
    _discrete_mst_colorbar(fig, ax, swatch_rgb)
    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_LC(
    out_path: Path,
    rows: List[Dict[str, str]],
    mst_L: np.ndarray,
    mst_a: np.ndarray,
    mst_b: np.ndarray,
    swatch_rgb: np.ndarray,
    mk_idx: np.ndarray,
    *,
    figure_title: str,
    dpi: int,
) -> None:
    mst_C = np.hypot(mst_a, mst_b)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    _draw_mst_locus(ax, mst_C, mst_L, swatch_rgb, mk_idx)
    _scatter_fitskin_and_flash(ax, rows, "C", "L")
    pL = np.concatenate(
        [_channel(rows, "fitskin_cheek", "L"), _channel(rows, "reflectance", "L")]
    )
    pC = np.concatenate(
        [_channel(rows, "fitskin_cheek", "C"), _channel(rows, "reflectance", "C")]
    )
    ax.set_xlabel("C*")
    ax.set_ylabel("L*")
    ax.set_title(figure_title)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    lo_L = min(float(np.min(mst_L)), float(np.min(pL))) - 5.0
    hi_L = max(float(np.max(mst_L)), float(np.max(pL))) + 5.0
    ax.set_xlim(0.0, _axis_max_with_padding(mst_C, pC))
    ax.set_ylim(lo_L, hi_L)
    _discrete_mst_colorbar(fig, ax, swatch_rgb)
    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_de00_bars(out_path: Path, rows: List[Dict[str, str]], *, dpi: int) -> None:
    labels = [r["subject_id"] for r in rows]
    de = [float(r["reflectance_cheek_de00"]) for r in rows]
    colors = [PARTICIPANT_COLORS.get(_pid_from_subject(lb), "0.5") for lb in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(x, de, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ΔE₀₀ vs FitSkin cheek")
    ax.set_title("Flash/no-flash skin vs FitSkin (reflectance Lab)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "flash_noflash_skin_output" / "flash_noflash_skin_lab.csv",
    )
    ap.add_argument(
        "--mst-csv",
        type=Path,
        default=ROOT / "mst_reference_cheng2024_table1.csv",
    )
    ap.add_argument(
        "--mst-swatch-dir",
        type=Path,
        default=Path("/home/mabl-main/Downloads/MST Swatches"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "flash_noflash_skin_output" / "figures",
    )
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help=f"Figure title (default: {DEFAULT_FIGURE_TITLE!r}).",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=f"Output PNG stem prefix (default: {DEFAULT_PREFIX}).",
    )
    args = ap.parse_args()
    figure_title = args.title or DEFAULT_FIGURE_TITLE
    prefix = args.prefix.strip() or DEFAULT_PREFIX

    if not args.csv.is_file():
        raise SystemExit(f"Missing --csv (run flash_no_flash_skin_lab.py first): {args.csv}")

    rows = load_flash_csv(args.csv)
    mk_idx, mst_L, mst_a, mst_b, rgb_table = load_mst_lab_csv(args.mst_csv)
    swatch_rgb, color_src = resolve_mst_display_rgb(
        args.mst_swatch_dir if args.mst_swatch_dir.is_dir() else None,
        rgb_table,
    )
    print(f"MST colors: {color_src}")
    print(f"Plotting {len(rows)} trial(s): FitSkin + flash/no-flash reflectance only")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_ab(
        args.out_dir / f"{prefix}_vs_mst_ab.png",
        rows,
        mst_a,
        mst_b,
        swatch_rgb,
        mk_idx,
        figure_title=figure_title,
        dpi=args.dpi,
    )
    plot_Lb(
        args.out_dir / f"{prefix}_vs_mst_Lb.png",
        rows,
        mst_L,
        mst_b,
        swatch_rgb,
        mk_idx,
        figure_title=figure_title,
        dpi=args.dpi,
    )
    plot_LC(
        args.out_dir / f"{prefix}_vs_mst_LC.png",
        rows,
        mst_L,
        mst_a,
        mst_b,
        swatch_rgb,
        mk_idx,
        figure_title=figure_title,
        dpi=args.dpi,
    )
    plot_de00_bars(
        args.out_dir / f"{prefix}_reflectance_de00_by_trial.png",
        rows,
        dpi=args.dpi,
    )
    print(
        f"Wrote {prefix}_vs_mst_ab.png, {prefix}_vs_mst_Lb.png, "
        f"{prefix}_vs_mst_LC.png, {prefix}_reflectance_de00_by_trial.png"
    )


if __name__ == "__main__":
    main()
