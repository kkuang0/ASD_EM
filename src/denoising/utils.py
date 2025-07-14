import torch
import torch.nn.functional as F
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF


def generate_n2v_mask(shape: Tuple[int, int], mask_ratio: float = 0.03, radius: int = 2) -> torch.Tensor:
    """Generate random mask for Noise2Void."""
    mask = torch.zeros(shape, dtype=torch.bool)
    num_pixels = int(mask.numel() * mask_ratio)
    coords = torch.stack(
        [torch.randint(0, s, (num_pixels,)) for s in shape], dim=1
    )
    for y, x in coords:
        y0 = max(y - radius, 0)
        y1 = min(y + radius + 1, shape[0])
        x0 = max(x - radius, 0)
        x1 = min(x + radius + 1, shape[1])
        mask[y0:y1, x0:x1] = True
    return mask


def apply_n2v_mask(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Noise2Void masking to an image batch."""
    mask = mask.unsqueeze(1)  # add channel dimension for indexing
    noisy_img = img.clone()
    noise = torch.randn_like(noisy_img)
    noisy_img[mask] = noise[mask]
    return noisy_img, img[mask]


def n2v_loss(pred: torch.Tensor, target_pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss computed on masked pixel locations."""
    mask = mask.unsqueeze(1)
    return F.mse_loss(pred[mask], target_pixels)


def save_comparison(original: torch.Tensor, denoised: torch.Tensor, path: str) -> None:
    """Save side-by-side original and denoised images."""
    orig_img = TF.to_pil_image(original.cpu().clamp(0, 1))
    den_img = TF.to_pil_image(denoised.cpu().clamp(0, 1))
    w, h = orig_img.size
    canvas = Image.new('L', (w * 2, h))
    canvas.paste(orig_img, (0, 0))
    canvas.paste(den_img, (w, 0))
    canvas.save(path)
