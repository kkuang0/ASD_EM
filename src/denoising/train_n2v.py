import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import PatchDataset
from .unet import UNet
from .utils import generate_n2v_mask, apply_n2v_mask, n2v_loss, save_comparison


def get_image_files(directory_path: str) -> list[str]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    image_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")
    
    return image_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise2Void training")
    p.add_argument(
        "--images",
        nargs="+",
        help="Image file paths. Use this, --csv, or --directory to specify training images.",
    )
    p.add_argument(
        "--csv",
        help=(
            "CSV file containing a 'filepath' column with image paths. "
            "Provides an alternative to --images or --directory."
        ),
    )
    p.add_argument(
        "--directory",
        help=(
            "Directory containing image files. 20 random images will be selected. "
            "Provides an alternative to --images or --csv."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="n2v_runs",
        help="Directory to save checkpoints",
    )
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--mask-ratio", type=float, default=0.03)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    # Count how many input methods are specified
    input_methods = sum([
        bool(args.images),
        bool(args.csv),
        bool(args.directory)
    ])
    
    if input_methods == 0:
        p.error("One of --images, --csv, or --directory must be specified")
    elif input_methods > 1:
        p.error("Only one of --images, --csv, or --directory can be specified")

    # Set random seed for reproducibility
    random.seed(args.random_state)

    # Load image paths from CSV if provided
    if args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        if "filepath" not in df.columns:
            raise ValueError("CSV must contain a 'filepath' column")
        # Limit training to 20 samples when a CSV is provided
        args.images = (
            df["filepath"]
            .sample(n=min(20, len(df)), random_state=args.random_state)
            .astype(str)
            .tolist()
        )
    
    # Load image paths from directory if provided
    elif args.directory:
        all_images = get_image_files(args.directory)
        # Randomly select up to 20 images
        num_images = min(20, len(all_images))
        args.images = random.sample(all_images, num_images)
        print(f"Selected {num_images} random images from directory: {args.directory}")

    return args


def train(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = PatchDataset(
        args.images, patch_size=args.patch_size, overlap=args.overlap
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, patience=5)

    best_loss = float("inf")
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
            mask = torch.stack(
                [
                    generate_n2v_mask(batch.shape[-2:], args.mask_ratio)
                    for _ in range(batch.size(0))
                ]
            )
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
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best.pt"),
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping")
                break


if __name__ == "__main__":
    args = parse_args()
    train(args)
    