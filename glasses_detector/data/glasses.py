import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

import glasses_detector.metadata.glasses as metadata
from glasses_detector.data.base_data_module import BaseDataModule
from glasses_detector.data.utils import download_url, BaseDataset, split_dataset
from glasses_detector.stems.data_augmentation import DataAugmentation

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

        
        self.transform = DataAugmentation()
        self.mapping = metadata.MAPPING
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.input_dims = metadata.INPUT_DIMS
        self.output_dims = metadata.OUTPUT_DIMS

    
    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_processed_glasses_data()

    
    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["X_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)
            self.data_val.transform = None

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["X_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=None)
    
    
    def __repr__(self):
        basic = f"Glasses Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.input_dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data
    


def _download_and_processed_glasses_data():
    _download_raw_dataset()
    _process_raw_dataset()


def _download_raw_dataset():
    RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILENAME.exists() and RAW_DATA_LABELS_FILENAME.exists():
        return RAW_DATA_FILENAME
    print(f"Downloading raw dataset from {IMAGES_URL} to {RAW_DATA_FILENAME}")
    download_url(IMAGES_URL, RAW_DATA_FILENAME)
    print(f"Downloading raw dataset from {LABELS_URL} to {RAW_DATA_LABELS_FILENAME}")
    download_url(LABELS_URL, RAW_DATA_LABELS_FILENAME)
    return RAW_DATA_FILENAME


def _process_raw_dataset(cpu_count=8):
    with zipfile.ZipFile(RAW_DATA_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIRNAME)
    
    labels = pd.read_csv(RAW_DATA_LABELS_FILENAME, sep=' ', header=None, names=['filename', 'label'])
    print('hi')

    num_images = len(labels)
    image_shape = np.array(Image.open(RAW_DATA_IMAGE_DIRNAME / labels.iloc[0]['filename'])).shape

    # Use multiprocessing to load images
    with Pool(cpu_count) as p:
        results = p.map(_load_image, labels.values)

    # Unpack results into X and y
    images, labels = zip(*results)
    X = np.array(images)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_FRAC, random_state=42)
    print("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("X_test", data=X_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")


def _load_image(row):
    filename, label = row
    image = np.array(Image.open(RAW_DATA_IMAGE_DIRNAME / filename))
    return image, label
