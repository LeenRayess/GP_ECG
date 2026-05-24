"""
Build SIMULATED qualitative review spreadsheets (Excel) for C1–C4.

Reads each condition's selected_cases.csv (for case_id, h5_index, bucket),
reorders rows like gallery.html (buckets alphabetically), then fills checklist
columns so bucket-level Present rates match scripted targets (for demos only).

Outputs under --out-dir (default reports/qualitative_error_analysis_simulated/):
  - review_labels_SIMULATED_C1.xlsx ... C4.xlsx
  - simulated_prevalence_targets.json  (declared target fractions)
  - simulated_prevalence_achieved.csv    (k/n per condition, bucket, checklist item)

Requires: openpyxl (pip install openpyxl)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Checklist columns (must match export_qualitative_review_patches.py / template)
CHK = [
    "chk_tissue_scarcity_Present_Absent_Unclear",
    "chk_artifact_burden_Present_Absent_Unclear",
    "chk_borderline_morphology_Present_Absent_Unclear",
    "chk_small_focus_lesion_Present_Absent_Unclear",
    "chk_color_stain_atypia_Present_Absent_Unclear",
    "chk_patch_context_limit_Present_Absent_Unclear",
]

# Internal keys -> column names
KEYS = [
    "tissue",
    "artifact",
    "borderline",
    "small_focus",
    "stain",
    "context",
]
KEY_TO_COL = dict(zip(KEYS, CHK))

# --- Target Present fractions (0–1) per (condition, bucket). Realistic story:
# C1 PCam in-domain: moderate FN focus/context; FP more artifact/stain.
# C2 PCam->CAM17: more stain/context on FNs (domain shift).
# C3 CAM17 in-domain: similar to C1 but slightly messier FPs.
# C4 CAM17->PCam: hardest FN story (context, small focus, stain); HE errors very borderline.

TARGETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "C1": {
        "FN": {
            "small_focus": 0.50,
            "context": 0.80,
            "borderline": 0.40,
            "tissue": 0.20,
            "artifact": 0.20,
            "stain": 0.15,
        },
        "FP": {
            "artifact": 0.50,
            "stain": 0.30,
            "borderline": 0.35,
            "small_focus": 0.20,
            "context": 0.15,
            "tissue": 0.10,
        },
        "high_entropy_error": {
            "borderline": 0.70,
            "small_focus": 0.45,
            "context": 0.40,
            "artifact": 0.25,
            "stain": 0.20,
            "tissue": 0.15,
        },
        "high_entropy_correct": {
            "borderline": 0.60,
            "small_focus": 0.25,
            "context": 0.25,
            "artifact": 0.15,
            "stain": 0.15,
            "tissue": 0.10,
        },
        "confident_error": {
            "artifact": 0.45,
            "stain": 0.35,
            "borderline": 0.30,
            "small_focus": 0.25,
            "context": 0.20,
            "tissue": 0.10,
        },
    },
    "C2": {
        "FN": {
            "stain": 0.55,
            "context": 0.55,
            "small_focus": 0.45,
            "borderline": 0.40,
            "artifact": 0.30,
            "tissue": 0.20,
        },
        "FP": {
            "stain": 0.60,
            "artifact": 0.45,
            "borderline": 0.35,
            "small_focus": 0.25,
            "context": 0.25,
            "tissue": 0.15,
        },
        "high_entropy_error": {
            "borderline": 0.75,
            "stain": 0.45,
            "context": 0.45,
            "small_focus": 0.40,
            "artifact": 0.30,
            "tissue": 0.15,
        },
        "high_entropy_correct": {
            "borderline": 0.55,
            "stain": 0.35,
            "context": 0.30,
            "small_focus": 0.20,
            "artifact": 0.20,
            "tissue": 0.10,
        },
        "confident_error": {
            "stain": 0.50,
            "artifact": 0.40,
            "borderline": 0.35,
            "small_focus": 0.30,
            "context": 0.25,
            "tissue": 0.10,
        },
    },
    "C3": {
        "FN": {
            "small_focus": 0.45,
            "context": 0.45,
            "borderline": 0.45,
            "artifact": 0.25,
            "stain": 0.25,
            "tissue": 0.20,
        },
        "FP": {
            "artifact": 0.55,
            "borderline": 0.40,
            "stain": 0.30,
            "small_focus": 0.25,
            "context": 0.20,
            "tissue": 0.15,
        },
        "high_entropy_error": {
            "borderline": 0.70,
            "small_focus": 0.50,
            "context": 0.40,
            "artifact": 0.30,
            "stain": 0.22,
            "tissue": 0.15,
        },
        "high_entropy_correct": {
            "borderline": 0.58,
            "small_focus": 0.22,
            "context": 0.22,
            "artifact": 0.18,
            "stain": 0.18,
            "tissue": 0.12,
        },
        "confident_error": {
            "artifact": 0.50,
            "borderline": 0.35,
            "stain": 0.32,
            "small_focus": 0.28,
            "context": 0.22,
            "tissue": 0.12,
        },
    },
    "C4": {
        "FN": {
            "context": 0.60,
            "small_focus": 0.55,
            "stain": 0.45,
            "borderline": 0.45,
            "artifact": 0.25,
            "tissue": 0.25,
        },
        "FP": {
            "stain": 0.45,
            "borderline": 0.40,
            "artifact": 0.35,
            "small_focus": 0.30,
            "context": 0.25,
            "tissue": 0.15,
        },
        "high_entropy_error": {
            "borderline": 0.80,
            "context": 0.50,
            "small_focus": 0.45,
            "stain": 0.35,
            "artifact": 0.28,
            "tissue": 0.18,
        },
        "high_entropy_correct": {
            "borderline": 0.62,
            "context": 0.30,
            "small_focus": 0.28,
            "stain": 0.25,
            "artifact": 0.15,
            "tissue": 0.12,
        },
        "confident_error": {
            "borderline": 0.40,
            "stain": 0.40,
            "artifact": 0.38,
            "small_focus": 0.30,
            "context": 0.25,
            "tissue": 0.12,
        },
    },
}

CONDITION_DIRS = {
    "C1": "C1_pcam_trained_on_pcam_test",
    "C2": "C2_pcam_trained_on_cam17_test",
    "C3": "C3_cam17_trained_on_cam17_test",
    "C4": "C4_cam17_trained_on_pcam_test",
}


def gallery_order_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)
    out: List[Dict[str, str]] = []
    for bname in sorted(buckets.keys()):
        out.extend(buckets[bname])
    return out


def _k_present(n: int, p: float) -> int:
    k = int(round(p * n))
    return max(0, min(n, k))


def assign_bucket_checklist(
    n: int,
    targets: Dict[str, float],
    rng,
) -> List[Dict[str, str]]:
    """Independent columns: exactly k Present each, shuffled rows (simulation)."""
    rows: List[Dict[str, str]] = [{c: "Absent" for c in CHK} for _ in range(n)]
    for key in KEYS:
        p = float(targets.get(key, 0.0))
        k = _k_present(n, p)
        col = KEY_TO_COL[key]
        idx = list(range(n))
        rng.shuffle(idx)
        for j in idx[:k]:
            rows[j][col] = "Present"
    return rows


def load_selected_cases(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_workbook_rows(
    ordered_cases: List[Dict[str, str]],
    condition: str,
    rng,
) -> Tuple[List[Dict[str, object]], List[dict]]:
    """Return (flat rows for sheet1, achieved summary records)."""
    by_bucket: Dict[str, List[Dict[str, str]]] = {}
    for r in ordered_cases:
        by_bucket.setdefault(r["bucket"], []).append(r)

    out_rows: List[Dict[str, object]] = []
    achieved: List[dict] = []
    review_order = 0

    for bname in sorted(by_bucket.keys()):
        cases = by_bucket[bname]
        n = len(cases)
        tgt = TARGETS[condition][bname]
        chk_rows = assign_bucket_checklist(n, tgt, rng)

        for i, case in enumerate(cases):
            review_order += 1
            chk = chk_rows[i]
            row = {
                "case_id": case["case_id"],
                "h5_index": case.get("h5_index", ""),
                "bucket": bname,
                "review_order": review_order,
                **chk,
                "free_text_note": f"[SIMULATED] {condition} {bname}",
            }
            out_rows.append(row)

        # achieved k per column for this bucket
        for key in KEYS:
            col = KEY_TO_COL[key]
            k = sum(1 for i in range(n) if chk_rows[i][col] == "Present")
            achieved.append(
                {
                    "condition": condition,
                    "bucket": bname,
                    "checklist_key": key,
                    "target_present_frac": tgt.get(key, 0.0),
                    "achieved_present_n": k,
                    "n": n,
                    "achieved_present_frac": k / n if n else 0.0,
                }
            )

    return out_rows, achieved


def write_xlsx(rows: List[Dict[str, object]], path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "review_labels"
    headers = [
        "case_id",
        "h5_index",
        "bucket",
        "review_order",
        *CHK,
        "free_text_note",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h, "") for h in headers])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qual-root",
        type=str,
        default="reports/qualitative_error_analysis/virchow_c1_c4",
        help="folder containing C1_*/selected_cases.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="reports/qualitative_error_analysis_simulated",
        help="output directory for xlsx + summary",
    )
    ap.add_argument("--seed", type=int, default=20260212)
    args = ap.parse_args()

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise SystemExit("Install openpyxl: pip install openpyxl")

    repo = Path(__file__).resolve().parents[1]
    qual_root = repo / args.qual_root
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "simulated_prevalence_targets.json", "w", encoding="utf-8") as f:
        json.dump(TARGETS, f, indent=2)

    all_achieved: List[dict] = []
    for cid, sub in CONDITION_DIRS.items():
        sc = qual_root / sub / "selected_cases.csv"
        if not sc.is_file():
            print("skip (no selected_cases.csv):", sc)
            continue
        raw_rows = load_selected_cases(sc)
        ordered = gallery_order_rows(raw_rows)
        rng = __import__("random").Random(args.seed + sum(ord(c) for c in cid))
        wb_rows, achieved = build_workbook_rows(ordered, cid, rng)
        all_achieved.extend(achieved)
        xlsx_path = out_dir / f"review_labels_SIMULATED_{cid}.xlsx"
        write_xlsx(wb_rows, xlsx_path)
        print("Wrote", xlsx_path, "rows", len(wb_rows))

    ach_path = out_dir / "simulated_prevalence_achieved.csv"
    if all_achieved:
        keys = list(all_achieved[0].keys())
        with open(ach_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_achieved)
        print("Wrote", ach_path)

    readme = out_dir / "README_SIMULATION.txt"
    readme.write_text(
        "\n".join(
            [
                "SIMULATED qualitative labels for thesis / slide practice only.",
                "Not human review. Do not cite as empirical results.",
                "Targets: simulated_prevalence_targets.json; achieved k/n: simulated_prevalence_achieved.csv",
                "",
                f"Source case lists: {qual_root}/*/selected_cases.csv",
            ]
        ),
        encoding="utf-8",
    )
    print("Wrote", readme)


if __name__ == "__main__":
    main()
