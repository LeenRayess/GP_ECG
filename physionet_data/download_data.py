"""
Full mirror of PhysioNet Challenge 2021 training data (no wget).
Downloads all files under training/: RECORDS in each folder + every .hea and .mat.
https://physionet.org/content/challenge-2021/1.0.3/training/
"""
import os
import time
import requests

BASE_FILES_URL = "https://physionet.org/files/challenge-2021/1.0.3/"
BASE_CONTENT_URL = "https://physionet.org/content/challenge-2021/1.0.3/"
ROOT_RECORDS_URL = BASE_CONTENT_URL + "RECORDS"
OUTPUT_DIR = "training"


def get_training_folder_paths():
    """Root RECORDS lists folder paths like 'training/cpsc_2018/g1/'."""
    r = requests.get(ROOT_RECORDS_URL)
    r.raise_for_status()
    paths = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("training/") and line.endswith("/"):
            paths.append(line)
    return paths


def get_record_ids_from_folder(folder_path):
    """Each folder has a RECORDS file listing record IDs (e.g. A0001, A0002)."""
    # Use files URL for raw text; content URL returns HTML and breaks parsing
    url = BASE_FILES_URL + folder_path + "RECORDS"
    r = requests.get(url)
    r.raise_for_status()
    ids = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip HTML or non-record lines (record IDs are alphanumeric, no spaces/angle brackets)
        if "<" in line or ">" in line or " " in line or len(line) > 32:
            continue
        ids.append(line)
    return ids


def download_file(url, local_path, session=None):
    """Download a single file; raise on failure."""
    req = session.get(url, stream=True) if session else requests.get(url, stream=True)
    req.raise_for_status()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in req.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    folder_paths = get_training_folder_paths()
    print(f"Full mirror: {len(folder_paths)} folder(s) under training/")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    session = requests.Session()
    total_files = 0
    failed = []
    start_time = time.time()
    update_interval = 100  # print progress every N files

    for folder_idx, folder_path in enumerate(folder_paths, 1):
        # e.g. "training/cpsc_2018/g1/" -> path under OUTPUT_DIR: "cpsc_2018/g1"
        path_from_training = folder_path[len("training/"):].rstrip("/")
        print(f"[{folder_idx}/{len(folder_paths)}] Folder: {path_from_training}")

        # 1. Download this folder's RECORDS file (part of full mirror)
        records_url = BASE_FILES_URL + folder_path + "RECORDS"
        records_local = os.path.join(OUTPUT_DIR, path_from_training, "RECORDS")
        if not os.path.exists(records_local):
            try:
                download_file(records_url, records_local, session)
                total_files += 1
                elapsed = time.time() - start_time
                print(f"  RECORDS saved | {total_files} files | {elapsed:.0f}s elapsed")
            except requests.RequestException as e:
                failed.append((records_url, str(e)))
        # Get record IDs (from content URL; we may have just saved RECORDS from files URL)
        try:
            record_ids = get_record_ids_from_folder(folder_path)
        except requests.RequestException as e:
            print(f"  Skip {folder_path}: {e}")
            failed.append((folder_path, str(e)))
            continue

        # 2. Download every .hea and .mat for this folder
        for record_id in record_ids:
            for ext in [".hea", ".mat"]:
                url = BASE_FILES_URL + folder_path + record_id + ext
                local_path = os.path.join(OUTPUT_DIR, path_from_training, record_id + ext)
                if os.path.exists(local_path):
                    continue
                try:
                    download_file(url, local_path, session)
                    total_files += 1
                    if total_files % update_interval == 0:
                        elapsed = time.time() - start_time
                        print(f"  ... {total_files} files | {elapsed:.0f}s elapsed")
                except requests.RequestException as e:
                    failed.append((url, str(e)))

    elapsed = time.time() - start_time
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for url, err in failed[:20]:
            print(f"  {url}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    print(f"\nDone. {total_files} files downloaded in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
