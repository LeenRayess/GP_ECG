"""
Extract and organize the PatchCamelyon (PCam) dataset after downloading from Google Drive.

Default paths (relative to GP_ECG project root):
  - Source: pcamv1/   (downloaded .gz and .csv files)
  - Output: pcam_data/
             training/   <- train x, y .h5 + train meta .csv
             val/        <- valid x, y .h5 + valid meta .csv
             test/       <- test x, y .h5 + test meta .csv

Usage:
  # From GP_ECG project root (recommended):
  python pcam-master/keras_pcam/dataset/extract_pcam.py

  # Or with explicit paths:
  python extract_pcam.py --source "c:/GP_ECG/pcamv1" --out "c:/GP_ECG/pcam_data"

  # Remove .gz files after extraction to save space:
  python extract_pcam.py --remove-gz

After extraction, use in code:
  load_data(data_dir=r"c:/GP_ECG/pcam_data")
"""

from __future__ import print_function

import argparse
import gzip
import os
import shutil

# Expected PCam files (from official README) -> (split folder name, base filename)
SPLIT_FILES = [
    ("training", "camelyonpatch_level_2_split_train_x.h5.gz", "camelyonpatch_level_2_split_train_x.h5"),
    ("training", "camelyonpatch_level_2_split_train_y.h5.gz", "camelyonpatch_level_2_split_train_y.h5"),
    ("val", "camelyonpatch_level_2_split_valid_x.h5.gz", "camelyonpatch_level_2_split_valid_x.h5"),
    ("val", "camelyonpatch_level_2_split_valid_y.h5.gz", "camelyonpatch_level_2_split_valid_y.h5"),
    ("test", "camelyonpatch_level_2_split_test_x.h5.gz", "camelyonpatch_level_2_split_test_x.h5"),
    ("test", "camelyonpatch_level_2_split_test_y.h5.gz", "camelyonpatch_level_2_split_test_y.h5"),
]
META_FILES = [
    ("training", "camelyonpatch_level_2_split_train_meta.csv"),
    ("val", "camelyonpatch_level_2_split_valid_meta.csv"),
    ("test", "camelyonpatch_level_2_split_test_meta.csv"),
]


def extract_gz(gz_path, out_path):
    """Decompress .gz to out_path."""
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def find_gz_in_dir(source_dir, expected_gz_name):
    """Return path to .gz file, using exact name or Google Drive–style variant (e.g. 'name.gz (1)')."""
    exact = os.path.join(source_dir, expected_gz_name)
    if os.path.isfile(exact):
        return exact
    prefix = expected_gz_name.replace(".gz", "")  # e.g. camelyonpatch_level_2_split_train_x.h5
    try:
        for f in os.listdir(source_dir):
            if not f.endswith(".gz") and ".gz" not in f:
                continue
            if f == expected_gz_name or f.startswith(prefix):
                p = os.path.join(source_dir, f)
                if os.path.isfile(p):
                    return p
    except OSError:
        pass
    return None


def main():
    # Default paths relative to GP_ECG project root (parent of pcam-master)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset -> keras_pcam -> pcam-master -> GP_ECG (3 levels up)
    project_root = os.path.normpath(os.path.join(script_dir, "..", "..", ".."))
    default_source = os.path.join(project_root, "pcamv1")
    default_out = os.path.join(project_root, "pcam_data")

    parser = argparse.ArgumentParser(
        description="Extract PCam .gz files into pcam_data/training, pcam_data/val, pcam_data/test."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=default_source,
        help="Directory containing the downloaded .gz and .csv files (default: GP_ECG/pcamv1).",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=default_out,
        help="Root output directory (default: GP_ECG/pcam_data). Creates training/, val/, test/ inside.",
    )
    parser.add_argument(
        "--remove-gz",
        action="store_true",
        help="Remove .gz files after successful extraction to save disk space.",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    out_root = os.path.abspath(args.out)

    if not os.path.isdir(source_dir):
        print("Error: source directory does not exist:", source_dir)
        return 1

    # Debug: show exactly where we're looking and what's there (so we can see path/name issues)
    train_x_gz = os.path.join(source_dir, "camelyonpatch_level_2_split_train_x.h5.gz")
    print("Source directory:", source_dir)
    print("Train X .gz path:", train_x_gz)
    print("  exists:", os.path.exists(train_x_gz), "  isfile:", os.path.isfile(train_x_gz))
    try:
        all_files = os.listdir(source_dir)
        print("  Files in folder ({} total):".format(len(all_files)))
        for f in sorted(all_files):
            print("    ", repr(f))
    except OSError as e:
        print("  listdir error:", e)
    print()

    for sub in ("training", "val", "test"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    found_any = False

    # Extract each .gz into the correct split folder
    for split_name, gz_name, h5_name in SPLIT_FILES:
        gz_path = find_gz_in_dir(source_dir, gz_name)
        out_path = os.path.join(out_root, split_name, h5_name)

        if os.path.exists(out_path):
            print("Skip (already exists):", os.path.join(split_name, h5_name))
            found_any = True
            continue
        if gz_path is None:
            print("Missing:", gz_name, "in", source_dir)
            if split_name == "training" and gz_name == "camelyonpatch_level_2_split_train_x.h5.gz":
                try:
                    contents = os.listdir(source_dir)
                    print("  Contents of source folder ({} items):".format(len(contents)))
                    for c in sorted(contents)[:50]:
                        print("    ", c)
                    if len(contents) > 50:
                        print("    ... and", len(contents) - 50, "more")
                except OSError:
                    pass
            continue

        print("Extracting:", os.path.basename(gz_path), "->", out_path)
        extract_gz(gz_path, out_path)
        found_any = True

        if args.remove_gz and os.path.isfile(gz_path):
            try:
                os.remove(gz_path)
                print("  Removed", os.path.basename(gz_path))
            except OSError as e:
                print("  Warning: could not remove", gz_path, e)

    # Copy meta CSVs into the correct split folder
    for split_name, csv_name in META_FILES:
        src = os.path.join(source_dir, csv_name)
        dst = os.path.join(out_root, split_name, csv_name)
        if os.path.exists(src) and not os.path.exists(dst):
            print("Copying:", csv_name, "->", os.path.join(split_name, csv_name))
            shutil.copy2(src, dst)
            found_any = True
        elif os.path.exists(dst):
            print("Skip (already exists):", os.path.join(split_name, csv_name))
            found_any = True

    if not found_any:
        print("No expected .gz or .csv files found in", source_dir)
        return 1

    print("Done. Layout:")
    print("  ", out_root)
    print("    training/  (train x, y .h5 + meta.csv)")
    print("    val/       (valid x, y .h5 + meta.csv)")
    print("    test/      (test x, y .h5 + meta.csv)")
    print("\nTo load for training, use:")
    print('  load_data(data_dir=r"{}")'.format(out_root))
    return 0


if __name__ == "__main__":
    exit(main())
