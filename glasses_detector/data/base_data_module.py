"""Base DataModule class"""

import argparse
import os
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from glasses_detector.data.utils import BaseDataset

BATCH_SIZE = 128
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


class BaseDataModule(pl.LightningDataModule):
    """Base class for all LightningDataModule classes."""

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variable below in subclasses
        self.input_dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        pass
        
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type = int,
            default=BATCH_SIZE,
            help = f"Number of examples to operate on per forward step. Default is {BATCH_SIZE}."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Number of additional processed to load data. Default is {DEFAULT_NUM_WORKERS}."
        )
        return parser
    
    def config(self):
        """Return import settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.input_dims, "output_dims": self.output_dims, "mapping": self.mapping}
    
    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use
        
        Use this method to do thing that might write to disk"""

    def setup(self, stage: Optional[str] = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.
        
        Here is where we typically split into train, validation and test.
        Should assign 'torch Dataset' objects to self.data_train, self.data_val, and optionally self.data_test
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
    

