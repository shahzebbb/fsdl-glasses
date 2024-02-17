import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split

import glasses_detector.metadata.glasses as metadata
from glasses_detector.data.base_data_module import BaseDataModule
from glasses_detector.data.utils import download_url

RAW_DATA_DIRNAME = metadata.RAW_DATA_DIRNAME
RAW_DATA_IMAGE_DIRNAME = metadata.RAW_DATA_IMAGE_DIRNAME
RAW_DATA_FILENAME = metadata.RAW_DATA_FILENAME
RAW_DATA_LABELS_FILENAME = metadata.RAW_DATA_LABELS_FILENAME
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME
PROCESSED_DATA_FILENAME = metadata.PROCESSED_DATA_FILENAME
ESSENTIALS_FILENAME = metadata.ESSENTIALS_FILENAME

IMAGES_URL = metadata.IMAGES_URL
LABELS_URL = metadata.LABELS_URL

TRAIN_FRAC = 0.8


class Glasses(BaseDataModule):
    """Glasses Dataset"""

    def __init__(self, args=None):
        super().__init__(args)

        self.transform = ...
        self.mapping = metadata.MAPPING
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.input_dims = metadata.INPUT_DIMS
        self.output_dims = metadata.OUTPUT_DIMS

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exits(PROCESSED_DATA_FILENAME):
            _download_and_processed_glasses_data()


def _download_and_processed_glasses_data():
    _download_raw_dataset()

def _download_raw_dataset():
    RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILENAME.exists() and RAW_DATA_LABELS_FILENAME.exists():
        return RAW_DATA_FILENAME
    print(f"Downloading raw dataset from {IMAGES_URL} to {RAW_DATA_FILENAME}")
    download_url(IMAGES_URL, RAW_DATA_FILENAME)
    print(f"Downloading raw dataset from {LABELS_URL} to {RAW_DATA_LABELS_FILENAME}")
    download_url(LABELS_URL, RAW_DATA_LABELS_FILENAME)
    return RAW_DATA_FILENAME


def _process_raw_dataset():
    with zipfile.ZipFile(RAW_DATA_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIRNAME)
    
    labels = pd.read_csv(RAW_DATA_LABELS_FILENAME, sep=' ', header=None, names=['Filename', 'Label'])
    X = []
    y = []

    # Iterate through each row of the CSV file
    for index, row in labels.iterrows():
        filename = row['filename']
        label = row['label']
        
        image = np.array(Image.open(RAW_DATA_IMAGE_DIRNAME / filename))
        
        # Check if the image was successfully loaded
        if image is not None:
            X.append(image)
            y.append(label)
        else:
            print(f"Error loading image: {filename}")

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_FRAC, random_state=42)
    print("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("X_test", data=X_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")
