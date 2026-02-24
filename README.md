# GP_ECG

Project for ECG and histopathology (PatchCamelyon) experiments: data setup, dataset investigation, and model training.

## Layout

```
GP_ECG/
├── pcam-master/           # PatchCamelyon loader & extraction (from basveeling/pcam)
│   └── keras_pcam/
│       └── dataset/
│           ├── pcam.py          # load_data(data_dir=...) — load train/val/test from HDF5
│           └── extract_pcam.py  # extract .gz → pcam_data/training, val, test
├── pcamv1/                # Downloaded PCam .gz and .csv (not in git; add locally)
├── pcam_data/              # Extracted PCam data (not in git)
│   ├── training/           # train x,y .h5 + meta.csv
│   ├── val/
│   └── test/
├── notebooks/              # Jupyter notebooks
│   └── pcam_data_investigation.ipynb   # Dataset EDA: files, sizes, labels, meta, samples
├── experiments/            # Model outputs, logs, predictions
├── .gitignore
└── README.md
```

## PatchCamelyon (PCam) setup

1. **Download** the dataset from [Google Drive](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB). Put all files into `pcamv1/` (in the project root).

2. **Extract** into `pcam_data/training`, `val`, `test`:
   ```bash
   cd GP_ECG
   python pcam-master/keras_pcam/dataset/extract_pcam.py
   ```
   Optional: `--remove-gz` to delete `.gz` files after extraction.

3. **Load in code**:
   ```python
   import sys
   sys.path.insert(0, "pcam-master")
   from keras_pcam.dataset.pcam import load_data

   (train_x, train_y, meta_train), (valid_x, valid_y, meta_valid), (test_x, test_y, meta_test) = load_data(
       data_dir="pcam_data"
   )
   ```

4. **Notebook**: Open `notebooks/pcam_data_investigation.ipynb` for file inventory, label counts, metadata, and sample images. Run from project root or from `notebooks/`.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib (for notebooks)
- h5py (for HDF5 loading when using Keras 3)
- keras or tensorflow (for training; loader works with either or h5py fallback)

Create a venv and install as needed:
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install pandas numpy matplotlib h5py
# pip install tensorflow  # or keras
```

## Data not in git

The following are ignored (see `.gitignore`):

- `pcamv1/` — downloaded PCam archives (~7.7 GB compressed)
- `pcam_data/` — extracted HDF5 and CSV (~tens of GB)
- `experiments/` contents (keep folder via `experiments/.gitkeep`)

Clone the repo then download PCam and run the extraction script locally.

## License

PCam data: [CC0](https://choosealicense.com/licenses/cc0-1.0/). Code in `pcam-master` is from [basveeling/pcam](https://github.com/basveeling/pcam) (MIT). Rest of the project: add your preferred license.
