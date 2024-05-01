import torch
from torchvision import transforms
from metadata.glasses import MEAN


class DataAugmentation:
    """XXXX
    """

    def __init__(self, p=0.5):
        self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Normalize(mean=list(MEAN), std=[1.0, 1.0, 1.0]),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.8),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=p),
    ])

    def __call__(self, img):
        img = self.transforms(img)

        return img