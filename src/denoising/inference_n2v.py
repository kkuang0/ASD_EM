import argparse
import os
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from .dataset import PatchDataset, load_image
from .unet import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise2Void inference")
    p.add_argument('--image', required=True, help='Image to denoise')
    p.add_argument('--model', required=True, help='Path to trained model')
    p.add_argument('--patch-size', type=int, default=64)
    p.add_argument('--overlap', type=int, default=16)
    p.add_argument('--output', default='denoised.tif')
    return p.parse_args()


def reconstruct_from_patches(patches: List[torch.Tensor], img_size: tuple, patch_size: int, overlap: int) -> torch.Tensor:
    step = patch_size - overlap
    out = torch.zeros(img_size)
    count = torch.zeros(img_size)
    idx = 0
    h, w = img_size
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            out[:, y:y+patch_size, x:x+patch_size] += patches[idx]
            count[:, y:y+patch_size, x:x+patch_size] += 1
            idx += 1
    out /= count
    return out


def inference(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    img = load_image(args.image)
    w, h = img.size
    dataset = PatchDataset([args.image], patch_size=args.patch_size, overlap=args.overlap, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    patches = []
    with torch.no_grad():
        for patch in loader:
            patch = patch.to(device)
            pred = model(patch)
            patches.append(pred.cpu())
    denoised = reconstruct_from_patches(patches, (1, h, w), args.patch_size, args.overlap)
    out_img = TF.to_pil_image(denoised.clamp(0, 1))
    out_img.save(args.output)
    print(f"Saved denoised image to {args.output}")


if __name__ == '__main__':
    inference(parse_args())
