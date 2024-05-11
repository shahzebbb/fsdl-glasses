from typing import Any, Callable, Dict, Sequence, Tuple, Union
from urllib.request import urlretrieve

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processed data and targest through optional transforms.
    
    Parameters
    ----------
        - data: 
                commonly these are torch tensors, numpy arrays, or PIL Imagess
        - targest:
                commonly these are torch tensors or numpy arrays
        - transform:
                function that takes a datum and applies a transformation on it
        - target_transform:
                function that takes a target and applies a transofrmation on it
    """

    def __init__(
            self,
            data: SequenceOrTensor,
            targets: SequenceOrTensor,
            transform: Callable = None,
            target_transform: Callable = None
    ) -> None:
        
        if len(data) != len(targets):
            raise ValueError('Data and targets must be of equal lenghts')
        
        super().__init__()

        self.data = data
        self.targets = targets
        self.tensor_transform = transforms.ToTensor()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.
        
        Parameters
        ----------
        index
            index of the datum and target
            
        Returns
        ----------
        (datum, target)
        """

        datum, target = self.data[index], self.targets[index]


        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target
    
def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
    other of size (1 - fraction) * size of the base_dataset.
    """
    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size
    return torch.utils.data.random_split(  # type: ignore
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )
    
class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # noqa: S310