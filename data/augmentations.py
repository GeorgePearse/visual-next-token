"""
Data augmentation utilities for Self-Supervised Learning.

Strong augmentations are crucial for contrastive learning methods like SimCLR, MoCo, etc.
"""

import random

from PIL import ImageFilter
from torchvision import transforms


class GaussianBlur:
    """Gaussian blur augmentation from SimCLR."""

    def __init__(self, sigma=None):
        self.sigma = sigma if sigma is not None else [0.1, 2.0]

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_simclr_augmentation(size=224):
    """
    Returns SimCLR-style augmentation pipeline.

    Args:
        size: Target image size

    Returns:
        torchvision.transforms composition
    """
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[0.1, 2.0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_moco_augmentation(size=224):
    """
    Returns MoCo-style augmentation pipeline.
    Similar to SimCLR but typically less aggressive.

    Args:
        size: Target image size

    Returns:
        torchvision.transforms composition
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_byol_augmentation(size=224):
    """
    Returns BYOL-style augmentation pipeline.

    Args:
        size: Target image size

    Returns:
        torchvision.transforms composition
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[0.1, 2.0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TwoCropsTransform:
    """
    Create two augmented views of the same image.
    Used for contrastive learning methods.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return [view1, view2]
