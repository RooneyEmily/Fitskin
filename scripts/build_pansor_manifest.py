#!/usr/bin/env python3
"""Build manifest for Pansor iPhone DNG exports (Color Checker + Sephora Bag, 2026-06-16).

Deduplicates nested duplicate DNGs (e.g. Liki bag folders re-exporting Color Checker T1).
Optionally links FitSkin cheek Lab from a scan-sessions CSV when provided.

Example::

    python3 scripts/build_pansor_manifest.py

    python3 scripts/build_pansor_manifest.py \\
        --fitskin-csv /path/to/scan-sessions-2026-06-16.csv \\
        --fitskin-mapping data/pansor_fitskin_mapping.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = Path(
    os.environ.get("PANSOR_DATA_ROOT", "~/Downloads/Pansor Images")
).expanduser()

PARTICIPANT_MAP = {
    "Emily": "Participant 1",
    "Liki": "Participant 2",
}

PARTICIPANT_NUM = {
    "Emily": 1,
    "Liki": 2,
}

CONDITION_PREFIX = {
    "Color Checker": "CC",
    "Sephora Bag": "BAG",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_session_stamp(folder_name: str) -> Tuple[str, Optional[int]]:
    """Return ISO-ish local timestamp string and optional trial number from folder name."""
    m = re.match(r"^(\d{8})_(\d{6})_(\d{3})(?:_T(\d+))?$", folder_name)
    if not m:
        return folder_name, None
    ymd, hms, ms, trial = m.group(1), m.group(2), m.group(3), m.group(4)
    dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    # Folder stamps are local capture time; store as UTC-agnostic string for sorting.
    stamp = f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{ms}"
    return stamp, int(trial) if trial else None


def _discover_sessions(data_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for person_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        person = person_dir.name
        if person not in PARTICIPANT_MAP:
            continue
        for condition_dir in sorted(person_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name
            if condition not in CONDITION_PREFIX:
                continue
            for session_dir in sorted(condition_dir.rglob("*")):
                if not session_dir.is_dir():
                    continue
                dngs = list(session_dir.glob("*.dng"))
                if not dngs:
                    continue
                noflash = flash = None
                for p in dngs:
                    name = p.name.lower()
                    if name.startswith("raw_flash"):
                        flash = p
                    elif name.startswith("raw_"):
                        noflash = p
                if noflash is None or flash is None:
                    continue
                rel = session_dir.relative_to(data_root)
                stamp, trial_hint = _parse_session_stamp(session_dir.name)
                face_lm = sorted(session_dir.glob("face_landmarks_*.json"))
                flash_lm = sorted(session_dir.glob("flash_face_landmarks_*.json"))
                rows.append(
                    {
                        "person": person,
                        "participant": PARTICIPANT_MAP[person],
                        "participant_num": PARTICIPANT_NUM[person],
                        "condition": condition,
                        "condition_code": CONDITION_PREFIX[condition],
                        "session_folder": str(rel),
                        "session_stamp": stamp,
                        "trial_hint": trial_hint,
                        "path_noflash": str(noflash.resolve()),
                        "path_flash": str(flash.resolve()),
                        "path_face_landmarks": str(face_lm[0].resolve()) if face_lm else "",
                        "path_flash_face_landmarks": str(flash_lm[0].resolve()) if flash_lm else "",
                        "pair_hash": _md5(noflash) + ":" + _md5(flash),
                    }
                )
    return rows


def _dedupe_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for row in sessions:
        key = row["pair_hash"]
        if key in seen:
            row["duplicate_of"] = seen[key]["session_folder"]
            row["include_in_eval"] = "no"
        else:
            seen[key] = row
            row["duplicate_of"] = ""
            row["include_in_eval"] = "yes"
            out.append(row)
    return out


def _assign_trials(rows: List[Dict[str, Any]]) -> None:
    """Assign trial index within (participant, condition), sorted by session_stamp."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("include_in_eval") != "yes":
            continue
        key = (row["participant"], row["condition"])
        groups.setdefault(key, []).append(row)
    for key in groups:
        groups[key].sort(key=lambda r: (r["session_stamp"], r["session_folder"]))
        for i, row in enumerate(groups[key], start=1):
            trial = row["trial_hint"] if row["trial_hint"] is not None else i
            row["trial"] = trial
            pnum = row["participant_num"]
            row["subject_id"] = f"P{pnum}_{row['condition_code']}_T{trial}"


def _load_fitskin_by_session(scan_csv: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with scan_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("id", "")).strip()
            if sid:
                out[sid] = row
    return out


def _load_fitskin_mapping(mapping_csv: Path) -> Dict[Tuple[str, str], str]:
    """(participant, trial) -> scan_session_id."""
    out: Dict[Tuple[str, str], str] = {}
    for row in _load_fitskin_mapping_rows(mapping_csv):
        part = str(row.get("participant", "")).strip()
        trial = str(row.get("trial", "")).strip()
        sid = str(row.get("scan_session_id", "")).strip()
        if part and trial and sid:
            out[(part, trial)] = sid
    return out


def _load_fitskin_mapping_rows(mapping_csv: Path) -> List[dict]:
    if not mapping_csv.is_file():
        return []
    with mapping_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_fitskin_by_participant(mapping_csv: Path) -> Dict[str, str]:
    """participant -> scan_session_id for session-level (one scan, all trials) mapping."""
    out: Dict[str, str] = {}
    for row in _load_fitskin_mapping_rows(mapping_csv):
        part = str(row.get("participant", "")).strip()
        sid = str(row.get("scan_session_id", "")).strip()
        scope = str(row.get("scope", "per_trial")).strip().lower()
        if part and sid and scope == "per_participant":
            out[part] = sid
    return out


def _inline_lab_from_mapping_row(row: dict) -> Optional[Dict[str, str]]:
    """Manual June-16 XPOL / inline Lab when scan-sessions CSV is unavailable."""
    keys = ("fitskin_cheek_L", "fitskin_cheek_a", "fitskin_cheek_b")
    vals = {k: str(row.get(k, "")).strip() for k in keys}
    if not all(vals.values()):
        return None
    return {
        **vals,
        "fitskin_forehead_L": str(row.get("fitskin_forehead_L", "")).strip(),
        "fitskin_forehead_a": str(row.get("fitskin_forehead_a", "")).strip(),
        "fitskin_forehead_b": str(row.get("fitskin_forehead_b", "")).strip(),
        "skin_color_source": str(row.get("skin_color_source", "")).strip(),
        "scan_timestamp_utc": str(row.get("scan_timestamp_utc", "")).strip(),
    }


def _attach_fitskin(
    rows: List[Dict[str, Any]],
    scan_csv: Optional[Path],
    mapping_csv: Optional[Path],
) -> None:
    by_session: Dict[str, dict] = {}
    if scan_csv is not None and scan_csv.is_file():
        by_session = _load_fitskin_by_session(scan_csv)

    per_trial: Dict[Tuple[str, str], str] = {}
    per_participant_sid: Dict[str, str] = {}
    inline_by_participant: Dict[str, dict] = {}
    if mapping_csv and mapping_csv.is_file():
        per_trial = _load_fitskin_mapping(mapping_csv)
        per_participant_sid = _load_fitskin_by_participant(mapping_csv)
        for mrow in _load_fitskin_mapping_rows(mapping_csv):
            part = str(mrow.get("participant", "")).strip()
            scope = str(mrow.get("scope", "per_trial")).strip().lower()
            if part and scope == "per_participant":
                inline = _inline_lab_from_mapping_row(mrow)
                if inline:
                    inline_by_participant[part] = {
                        "sid": str(mrow.get("scan_session_id", "")).strip(),
                        "scan_date": str(mrow.get("scan_date", "")).strip(),
                        **inline,
                    }

    for row in rows:
        inst: Dict[str, str] = {}
        ts = ""
        sid = ""
        linked = "no"
        if row.get("include_in_eval") == "yes":
            sid = per_trial.get((row["participant"], str(row.get("trial", ""))), "")
            if not sid:
                sid = per_participant_sid.get(row["participant"], "")
            if sid and sid in by_session:
                r = by_session[sid]
                ts = str(r.get("timestamp_utc", "") or r.get("scan_date", ""))
                inst = {
                    "fitskin_cheek_L": r.get("cheek_lab_d65_l_1", ""),
                    "fitskin_cheek_a": r.get("cheek_lab_d65_a_1", ""),
                    "fitskin_cheek_b": r.get("cheek_lab_d65_b_1", ""),
                    "fitskin_forehead_L": r.get("forehead_lab_d65_l_1", ""),
                    "fitskin_forehead_a": r.get("forehead_lab_d65_a_1", ""),
                    "fitskin_forehead_b": r.get("forehead_lab_d65_b_1", ""),
                }
                linked = (
                    "yes"
                    if all(str(inst[k]).strip() for k in ("fitskin_cheek_L", "fitskin_cheek_a", "fitskin_cheek_b"))
                    else "partial"
                )
            elif row["participant"] in inline_by_participant:
                inline = inline_by_participant[row["participant"]]
                sid = inline.get("sid", sid)
                ts = inline.get("scan_timestamp_utc", "")
                inst = {
                    "fitskin_cheek_L": inline["fitskin_cheek_L"],
                    "fitskin_cheek_a": inline["fitskin_cheek_a"],
                    "fitskin_cheek_b": inline["fitskin_cheek_b"],
                    "fitskin_forehead_L": inline.get("fitskin_forehead_L", ""),
                    "fitskin_forehead_a": inline.get("fitskin_forehead_a", ""),
                    "fitskin_forehead_b": inline.get("fitskin_forehead_b", ""),
                }
                scan_date = inline.get("scan_date", "")
                src = inline.get("skin_color_source", "")
                if scan_date == "2026-05-20":
                    linked = "yes_may20_cross_session"
                elif src:
                    linked = f"yes_june16_{src}"
                else:
                    linked = "yes_manual"
        row["scan_session_id"] = sid
        row["scan_timestamp_utc"] = ts
        row["fitskin_csv_linked"] = linked
        row.update(inst)


def _search_fitskin_exports(extra_paths: List[Path]) -> Dict[str, Any]:
    patterns = ("*scan*session*", "*scan-sessions*", "*fitskin*")
    hits: List[str] = []
    roots = [
        Path("/home/mabl-main/Downloads"),
        Path("/home/mabl-main/color_space_research"),
        ROOT / "data",
    ]
    roots.extend(extra_paths)
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                s = str(p.resolve())
                if s in seen:
                    continue
                seen.add(s)
                hits.append(s)
    june16 = [h for h in hits if "2026-06-16" in h or "20260616" in h]
    return {
        "searched_roots": [str(r) for r in roots if r.exists()],
        "all_scan_exports_found": sorted(hits),
        "june_16_candidates": sorted(june16),
        "june_16_found": len(june16) > 0,
        "note": (
            "No scan-sessions export for 2026-06-16 was found on disk. "
            "Export from FitSkin app/backend before running same-session ΔE₀₀."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--fitskin-csv",
        type=Path,
        default=None,
        help="FitSkin scan-sessions CSV (e.g. scan-sessions-2026-06-16.csv)",
    )
    ap.add_argument(
        "--fitskin-mapping",
        type=Path,
        default=ROOT / "data" / "pansor" / "pansor_fitskin_mapping.csv",
        help="Maps participant (+ optional trial) to scan_session_id",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "pansor" / "manifest_pansor_fitskin.csv",
    )
    ap.add_argument(
        "--search-report",
        type=Path,
        default=ROOT / "data" / "pansor" / "pansor_fitskin_scan_search.json",
    )
    args = ap.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(f"Data root not found: {args.data_root}")

    all_sessions = _discover_sessions(args.data_root)
    eval_rows = _dedupe_sessions(all_sessions)
    _assign_trials(eval_rows)
    _attach_fitskin(eval_rows, args.fitskin_csv, args.fitskin_mapping)

    # Also write duplicate rows for audit trail
    dup_rows = [r for r in all_sessions if r.get("include_in_eval") == "no"]

    fields = [
        "subject_id",
        "person",
        "participant",
        "condition",
        "condition_code",
        "trial",
        "session_stamp",
        "session_folder",
        "capture_date",
        "path_noflash",
        "path_flash",
        "path_face_landmarks",
        "path_flash_face_landmarks",
        "scan_session_id",
        "scan_timestamp_utc",
        "fitskin_csv_linked",
        "fitskin_cheek_L",
        "fitskin_cheek_a",
        "fitskin_cheek_b",
        "fitskin_forehead_L",
        "fitskin_forehead_a",
        "fitskin_forehead_b",
        "include_in_eval",
        "duplicate_of",
        "pair_hash",
    ]

    for row in eval_rows + dup_rows:
        row["capture_date"] = "2026-06-16"
        if "subject_id" not in row:
            row["subject_id"] = f"DUP_{row['session_folder']}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(eval_rows, key=lambda r: (r["participant"], r["condition"], r.get("trial", 0))):
            w.writerow(row)
        for row in dup_rows:
            w.writerow(row)

    search = _search_fitskin_exports([args.data_root.parent])
    search["manifest_out"] = str(args.out.resolve())
    search["eval_pairs"] = sum(1 for r in eval_rows if r.get("include_in_eval") == "yes")
    search["duplicate_pairs"] = len(dup_rows)
    search["fitskin_linked_eval_rows"] = sum(
        1 for r in eval_rows if str(r.get("fitskin_csv_linked", "")).startswith("yes")
    )
    args.search_report.write_text(json.dumps(search, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out} ({search['eval_pairs']} eval pairs, {search['duplicate_pairs']} duplicates)")
    print(f"FitSkin linked eval rows: {search['fitskin_linked_eval_rows']}")
    print(f"Search report: {args.search_report}")
    if not search["june_16_found"]:
        print("WARN: No June 16 scan-sessions CSV found — add export and re-run with --fitskin-csv")


if __name__ == "__main__":
    main()
