"""Add 'more at 0.35' cells to the tissue investigation notebook."""
import json
import sys

nb_path = "notebooks/temp_tissue_ratio_investigation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find index of the code cell that contains "thresholds_to_try"
insert_at = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "source" in cell:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "thresholds_to_try = [" in src and "least low 24 shown" in src:
            insert_at = i + 1
            break
if insert_at is None:
    # Fallback: insert before "Summary: total removed"
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "source" in cell:
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "Summary: total removed" in src:
                insert_at = i
                break
if insert_at is None:
    insert_at = 17

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## More patches removed at threshold 0.35\n",
        "\n",
        "Larger sample of what would be removed when using threshold 0.35 (least low first)."
    ]
}
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Threshold 0.35: show 60 removed patches (6x10 grid)\n",
        "thresh_035 = 0.35\n",
        "low_035 = data[data[:, final_col] < thresh_035]\n",
        "low_035_sorted = low_035[np.argsort(-low_035[:, final_col])]\n",
        "to_show_035 = low_035_sorted[:min(60, len(low_035_sorted))]\n",
        "fig_035, axes_035 = plt.subplots(6, 10, figsize=(14, 8))\n",
        "for k in range(60):\n",
        "    ax = axes_035.flat[k]\n",
        "    if k < len(to_show_035):\n",
        "        idx = int(to_show_035[k, idx_col])\n",
        "        patch = np.asarray(train_x[idx])\n",
        "        if patch.max() > 1.0: patch = np.clip(patch / 255.0, 0, 1)\n",
        "        ax.imshow(patch)\n",
        "        fin_p = to_show_035[k, final_col]\n",
        "        lab = \"abn\" if int(train_y[idx]) == 1 else \"norm\"\n",
        "        ax.set_title(f\"{idx} {fin_p:.2f}\", fontsize=6)\n",
        "    ax.axis(\"off\")\n",
        "plt.suptitle(f\"Threshold 0.35: {len(low_035)} removed — showing {len(to_show_035)} (least low first)\", fontsize=11)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

nb["cells"].insert(insert_at, code_cell)
nb["cells"].insert(insert_at, markdown_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Inserted markdown + code at index", insert_at)
