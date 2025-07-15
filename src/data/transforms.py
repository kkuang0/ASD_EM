import torchvision.transforms as transforms
from typing import List
import torch
from PIL import Image
import numpy as np
from skimage import exposure, restoration


class RandomGaussianNoise:
    """Add Gaussian noise to a tensor with probability ``p``."""

    def __init__(self, std: float = 0.01, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return tensor + torch.randn_like(tensor) * self.std
        return tensor


class GrayscaleNormalize(transforms.Normalize):
    """Custom normalization for grayscale images converted to 3-channel"""

    def __init__(self):
        # ImageNet mean/std repeated for 3 channels, but adjusted for grayscale
        # Using grayscale conversion weights: 0.299*R + 0.587*G + 0.114*B
        # For grayscale images repeated across channels, we use the same normalization
        super().__init__(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])


class CLAHEAndNLM:
    """Apply CLAHE and Non-local Means filtering."""

    def __init__(self, clip_limit: float = 0.02, h: float = 0.15):
        self.clip_limit = clip_limit
        self.h = h

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to grayscale numpy array
        arr = np.array(img.convert("L"), dtype=np.uint8)
        arr = arr.astype(np.float32) / 255.0
        # CLAHE with clip limit
        arr = exposure.equalize_adapthist(arr, clip_limit=self.clip_limit)
        # Non-local Means filtering
        arr = restoration.denoise_nl_means(arr, h=self.h, fast_mode=True)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        arr_rgb = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr_rgb)


def get_em_transforms(
    image_size: int = 224, is_training: bool = True
) -> transforms.Compose:
    """
    Transforms optimized for 8-bit grayscale electron microscopy images of axon cross-sections.

    Key considerations for EM axon images:
    - Grayscale images with fine structural details
    - Circular/elliptical structures (axons) that are rotation-invariant
    - Scale: 10nm per pixel, so geometric relationships are important
    - High contrast between myelin sheaths and axon interiors
    - Images are 1024x1024 originally

    Args:
        image_size: Target size for model input (depends on backbone)
        is_training: Whether to apply training augmentations
    """

    preprocess = CLAHEAndNLM(clip_limit=0.02, h=0.15)

    if not is_training:
        # Validation/test transforms - preserve image quality
        return transforms.Compose(
            [
                preprocess,
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert to 3-channel for pretrained models
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        )

    # Training transforms optimized for EM axon cross-sections
    training_transforms = [
        preprocess,
        # Resize with slight upsampling to allow for random cropping
        transforms.Resize(
            int(image_size * 1.05), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        # Random crop to introduce spatial variations
        transforms.RandomCrop(image_size),
        # Rotation augmentation - axons are roughly circular, so rotation should be invariant
        # Limited to 180° since EM images don't have a natural "up" orientation
        transforms.RandomRotation(
            180, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        # Horizontal flip - EM cross-sections are symmetric
        transforms.RandomHorizontalFlip(p=0.5),
        # Vertical flip - also symmetric for axon cross-sections
        transforms.RandomVerticalFlip(p=0.5),
        # Brightness/contrast adjustments for varying EM imaging conditions
        # Conservative values to preserve fine structural details
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.1,  # Slight brightness variation for different imaging conditions
                    contrast=0.15,  # Moderate contrast changes to simulate different EM settings
                    saturation=0,  # No saturation change for grayscale
                    hue=0,  # No hue change for grayscale
                )
            ],
            p=0.4,
        ),
        # Gaussian noise to simulate imaging noise (applied after tensor conversion)
        # This will be handled by a custom transform below
        # Convert grayscale to 3-channel for compatibility with pretrained models
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # Custom normalization for grayscale images
        GrayscaleNormalize(),
        # Add slight Gaussian noise to simulate EM imaging conditions
        RandomGaussianNoise(std=0.01, p=0.3),
    ]

    return transforms.Compose(training_transforms)


def get_test_time_augmentation(image_size: int = 224) -> List[transforms.Compose]:
    """
    Test-time augmentation transforms for EM axon cross-sections.

    Uses multiple views of the same image to improve prediction robustness:
    - Original image
    - 90° rotation
    - 180° rotation
    - Horizontal flip
    - Vertical flip

    Args:
        image_size: Target size for model input
    """

    base_transforms = [
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        GrayscaleNormalize(),
    ]

    return [
        # Original image
        transforms.Compose(base_transforms),
        # 90° rotation
        transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomRotation(
                    (90, 90), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        ),
        # 180° rotation
        transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomRotation(
                    (180, 180), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        ),
        # 270° rotation
        transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomRotation(
                    (270, 270), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        ),
        # Horizontal flip
        transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        ),
        # Vertical flip
        transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        ),
    ]


def get_backbone_image_size(backbone_name: str) -> int:
    """Get the recommended image size for different backbone architectures"""

    backbone_sizes = {
        # EfficientNet family
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 260,
        "efficientnet_b3": 300,
        "efficientnet_b4": 380,
        "efficientnet_b5": 456,
        "efficientnet_b6": 528,
        "efficientnet_b7": 600,  # Added B7 support
        # ResNet family
        "resnet18": 224,
        "resnet34": 224,
        "resnet50": 224,
        "resnet101": 224,
        "resnet152": 224,
        # Vision Transformers
        "vit_base_patch16_224": 224,
        "vit_large_patch16_224": 224,
        "vit_base_patch16_384": 384,
        # Other models
        "densenet121": 224,
        "densenet161": 224,
        "densenet169": 224,
        "densenet201": 224,
    }

    return backbone_sizes.get(backbone_name, 224)  # Default to 224 if not found
