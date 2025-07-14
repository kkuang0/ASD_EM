import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def load_image(path: str) -> Image.Image:
    """Load an image from disk."""
    with Image.open(path) as img:
        return img.convert('L')


def normalize(img: torch.Tensor) -> torch.Tensor:
    """Normalize an image tensor to 0-1 range."""
    img = img.float()
    if img.max() > 1:
        img = img / 255.0
    return img


def random_augment(img: torch.Tensor) -> torch.Tensor:
    """Apply rotation and flip augmentations."""
    if torch.rand(1) < 0.5:
        img = TF.hflip(img)
    if torch.rand(1) < 0.5:
        img = TF.vflip(img)
    # rotations in multiples of 90 degrees
    k = torch.randint(0, 4, (1,)).item()
    img = torch.rot90(img, k, [1, 2])
    return img


class PatchDataset(Dataset):
    """Dataset that extracts patches from SEM images."""

    def __init__(self, image_paths: List[str], patch_size: int = 64, overlap: int = 16,
                 augment: bool = True):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment = augment
        self.patches: List[Tuple[str, Tuple[int, int]]] = []
        self._index_patches()

    def _index_patches(self):
        for path in self.image_paths:
            img = load_image(path)
            w, h = img.size
            step = self.patch_size - self.overlap
            for y in range(0, h - self.patch_size + 1, step):
                for x in range(0, w - self.patch_size + 1, step):
                    self.patches.append((path, (x, y)))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, (x, y) = self.patches[idx]
        img = load_image(path)
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        patch_t = normalize(TF.to_tensor(patch))
        if self.augment:
            patch_t = random_augment(patch_t)
        return patch_t
