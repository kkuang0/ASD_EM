import os
import argparse
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models


class PathologyDataset(Dataset):
    """Simple dataset for binary pathology classification."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"Control": 0, "ASD": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_map.get(row["pathology"], 0)
        return img, label


def build_transforms(image_size=224, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def setup_ddp(rank, world_size, backend, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def run(rank, world_size, args):
    setup_ddp(rank, world_size, args.backend, args.port)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    if "patient_id" not in df.columns:
        raise ValueError("CSV must contain a 'patient_id' column for grouping")

    gkf = GroupKFold(n_splits=args.n_splits)
    groups = df["patient_id"]
    splits = list(gkf.split(df, groups=groups))
    if args.fold >= len(splits):
        raise ValueError(f"Fold index {args.fold} out of range for {len(splits)} splits")

    train_idx, val_idx = splits[args.fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = PathologyDataset(train_df, transform=build_transforms(train=True))
    val_dataset = PathologyDataset(val_df, transform=build_transforms(train=False))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if rank == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt"))

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Binary pathology classification with DDP")
    parser.add_argument("--csv", required=True, help="Metadata CSV with 'filepath' and 'pathology' columns")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--n-splits", type=int, default=5, help="Number of group CV folds")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold to use for validation (0-indexed)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    world_size = torch.cuda.device_count() or 1
    mp.spawn(run, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
