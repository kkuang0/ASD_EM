import argparse
import csv
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import PatchDataset
from .unet import UNet
from .utils import generate_n2v_mask, apply_n2v_mask, n2v_loss, save_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise2Void training")
    p.add_argument('--images', nargs='+', help='Image file paths')
    p.add_argument('--image-csv', help='CSV file containing a column named "filepath" with image paths')
    p.add_argument('--output-dir', default='n2v_runs', help='Directory to save checkpoints')
    p.add_argument('--patch-size', type=int, default=64)
    p.add_argument('--overlap', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--mask-ratio', type=float, default=0.03)
    args = p.parse_args()
    if not args.images and not args.image_csv:
        p.error('Specify --images or --image-csv')
    return args


def train(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.image_csv:
        with open(args.image_csv, newline='') as f:
            reader = csv.DictReader(f)
            image_paths = [row['filepath'] for row in reader]
    else:
        image_paths = args.images
    dataset = PatchDataset(image_paths, patch_size=args.patch_size, overlap=args.overlap)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, patience=5)

    best_loss = float('inf')
    no_improve = 0

    # select example patch for visualization
    aug_flag = dataset.augment
    dataset.augment = False
    example_patch = dataset[0].unsqueeze(0).to(device)
    dataset.augment = aug_flag
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            mask = torch.stack([generate_n2v_mask(batch.shape[-2:], args.mask_ratio) for _ in range(batch.size(0))])
            noisy, target_pixels = apply_n2v_mask(batch, mask)
            pred = model(noisy)
            loss = n2v_loss(pred, target_pixels, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1} Loss {epoch_loss:.4f}")

        # log example denoising result
        model.eval()
        with torch.no_grad():
            example_pred = model(example_patch)
        save_comparison(
            example_patch.squeeze(0),
            example_pred.squeeze(0).cpu(),
            os.path.join(args.output_dir, f"epoch_{epoch+1}.png"),
        )
        model.train()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pt'))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print('Early stopping')
                break


if __name__ == '__main__':
    args = parse_args()
    train(args)
