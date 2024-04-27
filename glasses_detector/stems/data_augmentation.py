import torch
from torchvision import transforms


class DataAugmentation:
    """XXXX
    """

    def __init__(self, p=0.5):
        self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=p),
            transforms.RandomVerticalFlip(p=p),
            transforms.RandomRotation(degrees=(-45, 45), p=p),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=p),
    ])

    def __call__(self, img):
        img = self.transforms(img)

        return img