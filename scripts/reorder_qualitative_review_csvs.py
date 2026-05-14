"""
Rewrite selected_cases.csv + review_labels_template.csv to match gallery.html order.

Gallery order = bucket names sorted alphabetically, cases in the same within-bucket
order as in the current selected_cases.csv (first occurrence order when scanning the file).

Use after an older export that wrote CSVs in sampling-plan order or a shuffled template.
Does not regenerate PNGs or gallery.html.

Example:
  python scripts/reorder_qualitative_review_csvs.py --out-dir reports/qualitative_error_analysis/virchow_c1_c4/C1_pcam_trained_on_pcam_test
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List


def gallery_order_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    buckets: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        buckets[r["bucket"]].append(r)
    out: List[Dict[str, str]] = []
    for bname in sorted(buckets.keys()):
        out.extend(buckets[bname])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reorder qualitative review CSVs to gallery.html order.")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    src = out_dir / "selected_cases.csv"
    if not src.is_file():
        raise SystemExit(f"missing {src}")

    with open(src, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames
        if not fieldnames or "bucket" not in fieldnames:
            raise SystemExit("selected_cases.csv needs a header with bucket column")
        raw_rows = list(rdr)

    ordered = gallery_order_rows(raw_rows)

    backup = out_dir / "selected_cases_sampling_plan_order.csv"
    if not backup.exists():
        shutil.copy2(src, backup)

    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)

    checklist_cols = [
        "chk_tissue_scarcity_Present_Absent_Unclear",
        "chk_artifact_burden_Present_Absent_Unclear",
        "chk_borderline_morphology_Present_Absent_Unclear",
        "chk_small_focus_lesion_Present_Absent_Unclear",
        "chk_color_stain_atypia_Present_Absent_Unclear",
        "chk_patch_context_limit_Present_Absent_Unclear",
        "free_text_note",
    ]
    review_path = out_dir / "review_labels_template.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "h5_index", "bucket", "review_order"] + checklist_cols)
        for ro, r in enumerate(ordered):
            w.writerow(
                [r["case_id"], r["h5_index"], r["bucket"], ro + 1] + [""] * len(checklist_cols)
            )

    print("Wrote gallery-order:", src)
    print("Wrote gallery-order:", review_path)
    if backup.exists():
        print("Preserved sampling-plan copy:", backup)


if __name__ == "__main__":
    main()
