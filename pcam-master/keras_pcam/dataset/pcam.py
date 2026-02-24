"""PatchCamelyon(PCam) dataset
Small 96x96 patches from histopathology slides from the Camelyon16 dataset.

Please consider citing [1] when used in your publication:
- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv [cs.CV] (2018), (available at http://arxiv.org/abs/1806.03962).


Author: Bastiaan Veeling
Source: https://github.com/basveeling/pcam
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd

# HDF5Matrix was removed in Keras 3; use h5py-based fallback when needed
try:
    from keras.utils import HDF5Matrix
    from keras.utils.data_utils import get_file
    from keras import backend as K
except ImportError:
    try:
        from tensorflow.keras.utils import HDF5Matrix
        from tensorflow.keras.utils import get_file
        from tensorflow.keras import backend as K
    except ImportError:
        HDF5Matrix = None
        get_file = None
        K = None

if HDF5Matrix is None:
    import h5py
    import numpy as np

    class HDF5Matrix(object):
        """Thin wrapper around h5py dataset (Keras 3 removed HDF5Matrix)."""
        def __init__(self, filepath, dataname):
            self._file = h5py.File(filepath, "r")
            self._data = self._file[dataname]

        def __len__(self):
            return self._data.shape[0]

        def __getitem__(self, key):
            return np.array(self._data[key])

        @property
        def shape(self):
            return self._data.shape

        def __del__(self):
            if hasattr(self, "_file") and self._file is not None:
                self._file.close()
                self._file = None


def get_unzip_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    import gzip
    import shutil
    get_file()
    with open('file.txt', 'rb') as f_in, gzip.open('file.txt.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def _load_from_dir(data_dir):
    """Load PCam from a directory of extracted .h5 and .csv files.

    Supports two layouts:
    - Organized: data_dir/training/, data_dir/val/, data_dir/test/ (each with .h5 and .csv)
    - Flat: all .h5 and .csv directly in data_dir
    """
    data_dir = os.path.abspath(data_dir)

    def path_in(subdir, name):
        return os.path.join(data_dir, subdir, name)

    # Detect organized layout (pcam_data/training, val, test)
    training_dir = os.path.join(data_dir, "training")
    if os.path.isdir(training_dir) and os.path.exists(
        path_in("training", "camelyonpatch_level_2_split_train_x.h5")
    ):
        x_train = HDF5Matrix(path_in("training", "camelyonpatch_level_2_split_train_x.h5"), "x")
        y_train = HDF5Matrix(path_in("training", "camelyonpatch_level_2_split_train_y.h5"), "y")
        x_valid = HDF5Matrix(path_in("val", "camelyonpatch_level_2_split_valid_x.h5"), "x")
        y_valid = HDF5Matrix(path_in("val", "camelyonpatch_level_2_split_valid_y.h5"), "y")
        x_test = HDF5Matrix(path_in("test", "camelyonpatch_level_2_split_test_x.h5"), "x")
        y_test = HDF5Matrix(path_in("test", "camelyonpatch_level_2_split_test_y.h5"), "y")
        meta_train = pd.read_csv(path_in("training", "camelyonpatch_level_2_split_train_meta.csv"))
        meta_valid = pd.read_csv(path_in("val", "camelyonpatch_level_2_split_valid_meta.csv"))
        meta_test = pd.read_csv(path_in("test", "camelyonpatch_level_2_split_test_meta.csv"))
    else:
        # Flat layout: all files in data_dir
        def h5_path(name):
            return os.path.join(data_dir, name)
        def csv_path(name):
            return os.path.join(data_dir, name)
        x_train = HDF5Matrix(h5_path("camelyonpatch_level_2_split_train_x.h5"), "x")
        y_train = HDF5Matrix(h5_path("camelyonpatch_level_2_split_train_y.h5"), "y")
        x_valid = HDF5Matrix(h5_path("camelyonpatch_level_2_split_valid_x.h5"), "x")
        y_valid = HDF5Matrix(h5_path("camelyonpatch_level_2_split_valid_y.h5"), "y")
        x_test = HDF5Matrix(h5_path("camelyonpatch_level_2_split_test_x.h5"), "x")
        y_test = HDF5Matrix(h5_path("camelyonpatch_level_2_split_test_y.h5"), "y")
        meta_train = pd.read_csv(csv_path("camelyonpatch_level_2_split_train_meta.csv"))
        meta_valid = pd.read_csv(csv_path("camelyonpatch_level_2_split_valid_meta.csv"))
        meta_test = pd.read_csv(csv_path("camelyonpatch_level_2_split_test_meta.csv"))

    if K is not None and K.image_data_format() == "channels_first":
        raise NotImplementedError()

    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)


def load_data(data_dir=None):
    """Loads PCam dataset.

    # Arguments
        data_dir: Optional path to a directory containing already-downloaded
                  and extracted PCam files (.h5 and meta .csv). If None, attempts
                  download via Keras get_file (may fail for Google Drive).

    # Returns
        Tuple of three splits: (train), (valid), (test).
        Each split is (x, y, meta) with HDF5Matrix x/y and pandas DataFrame meta.
    """
    if data_dir is not None:
        data_dir = os.path.abspath(data_dir)
        if not os.path.isdir(data_dir):
            raise ValueError("data_dir is not a directory: {}".format(data_dir))
        return _load_from_dir(data_dir)

    dirname = os.path.join('datasets', 'pcam')
    base = 'https://drive.google.com/uc?export=download&id='
    try:
        y_train = HDF5Matrix(get_file('camelyonpatch_level_2_split_train_y.h5', origin= base+ '1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG', cache_subdir=dirname, archive_format='gzip'), 'y')
        x_valid = HDF5Matrix(get_file('camelyonpatch_level_2_split_valid_x.h5', origin= base+ '1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3', cache_subdir=dirname, archive_format='gzip'), 'x')
        y_valid = HDF5Matrix(get_file('camelyonpatch_level_2_split_valid_y.h5', origin= base+ '1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO', cache_subdir=dirname, archive_format='gzip'), 'y')
        x_test = HDF5Matrix(get_file('camelyonpatch_level_2_split_test_x.h5', origin= base+ '1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_', cache_subdir=dirname, archive_format='gzip'), 'x')
        y_test = HDF5Matrix(get_file('camelyonpatch_level_2_split_test_y.h5', origin= base+ '17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP', cache_subdir=dirname, archive_format='gzip'), 'y')

        meta_train = pd.read_csv(get_file('camelyonpatch_level_2_split_train_meta.csv', origin= base+ '1XoaGG3ek26YLFvGzmkKeOz54INW0fruR', cache_subdir=dirname))
        meta_valid = pd.read_csv(get_file('camelyonpatch_level_2_split_valid_meta.csv', origin= base+ '16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg', cache_subdir=dirname))
        meta_test = pd.read_csv(get_file('camelyonpatch_level_2_split_test_meta.csv', origin= base+ '19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4', cache_subdir=dirname))
        x_train = HDF5Matrix(get_file('camelyonpatch_level_2_split_train_x.h5', origin= base+ '1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2', cache_subdir=dirname, archive_format='gzip'), 'x')
    except OSError:
        raise NotImplementedError('Direct download currently not working. Please go to https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB and press download all. Then place files (ungzipped) in ~/.keras/datasets/pcam.')
        
    if K is not None and K.image_data_format() == "channels_first":
        raise NotImplementedError()

    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)


if __name__ == '__main__':
    import sys
    data_dir = os.environ.get('PCAM_DATA_DIR', None)
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data(data_dir=data_dir)
    print('Train:', len(x_train), 'Valid:', len(x_valid), 'Test:', len(x_test))
