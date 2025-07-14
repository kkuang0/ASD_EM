import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from typing import Tuple


def edge_preservation(original: torch.Tensor, denoised: torch.Tensor) -> float:
    """Compute edge preservation using Sobel gradients."""
    sobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    sobel_x = sobel.view(1, 1, 3, 3)
    sobel_y = sobel.t().view(1, 1, 3, 3)
    grad_orig = F.conv2d(original, sobel_x, padding=1) ** 2 + F.conv2d(original, sobel_y, padding=1) ** 2
    grad_deno = F.conv2d(denoised, sobel_x, padding=1) ** 2 + F.conv2d(denoised, sobel_y, padding=1) ** 2
    return F.cosine_similarity(grad_orig.flatten(), grad_deno.flatten(), dim=0).item()


def texture_similarity(original: torch.Tensor, denoised: torch.Tensor) -> float:
    """Evaluate texture preservation using SSIM."""
    orig_np = original.squeeze().cpu().numpy()
    den_np = denoised.squeeze().cpu().numpy()
    return ssim(orig_np, den_np, data_range=1.0)
