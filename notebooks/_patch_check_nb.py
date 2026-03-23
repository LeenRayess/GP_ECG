"""One-off patch for check_normalization_results.ipynb — run from repo root."""
import json

p = r"c:\GP_ECG\notebooks\check_normalization_results.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][0]["source"])
src = src.replace(
    'PREPROC_DIR = os.path.join(DATA_DIR, "preprocessed")',
    'PREPROC_DIR = os.path.join(DATA_DIR, "preprocessed_multi_ref")  # use "preprocessed" for older single-ref run',
)
for a, b in [
    (
        "pcam_data/preprocessed/test_x.h5",
        "pcam_data/preprocessed_multi_ref/test_x.h5",
    ),
    (
        "pcam_data/preprocessed/preprocess_report.json",
        "pcam_data/preprocessed_multi_ref/preprocess_report.json",
    ),
]:
    src = src.replace(a, b)

lines = src.split("\n")
nb["cells"][0]["source"] = [ln + "\n" for ln in lines[:-1]]
if lines:
    nb["cells"][0]["source"].append(lines[-1] + ("\n" if lines[-1] else ""))

for c in nb["cells"]:
    c["outputs"] = []
    c["execution_count"] = None

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("patched", p)
