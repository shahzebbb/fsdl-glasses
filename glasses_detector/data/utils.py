from typing import Any, Callable, Dict, Sequence, Tuple, Union

from PIL import Image
import torch


SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.Dataset):
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