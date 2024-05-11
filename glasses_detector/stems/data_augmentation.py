import torch
from torchvision import transforms
from glasses_detector.metadata.glasses import MEAN


class Normalize:
    """XXXX
    """

    def __init__(self):
        self.transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(MEAN), std=[1.0, 1.0, 1.0])
    ])

    def __call__(self, img):
        img = self.transforms(img)

        return img


class DataAugmentation:
    """XXXX
    """

    def __init__(self, p=0.5):
        self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.8),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=p)
    ])
        self.normalize = Normalize()

    def __call__(self, img):
        img = self.transforms(img)
        img = self.normalize(img)

        return img