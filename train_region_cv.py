import argparse
import pandas as pd
import torch
import torch.multiprocessing as mp
from types import SimpleNamespace
import tempfile
import os

from src.data.utils import get_class_weight_tensors
from src.training.ddp_trainer import DDPTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validation per region using DDP")
    parser.add_argument('--csv', required=True, help='Path to metadata CSV')
    parser.add_argument('--output-dir', default='runs', help='Base directory for models and logs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--holdout-frac', type=float, default=0.2,
                        help='Fraction of data reserved for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training init')
    return parser.parse_args()


def build_config(args, rank, world_size, class_weights, csv_path, region_name):
    cfg = SimpleNamespace()
    cfg.csv_path = csv_path  # region-specific temporary CSV
    cfg.output_dir = os.path.join(args.output_dir, f"region_{region_name}")
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.lr = args.lr
    cfg.scheduler = args.scheduler
    cfg.backbone = args.backbone
    cfg.n_splits = args.n_splits
    cfg.holdout_frac = args.holdout_frac
    cfg.random_seed = args.seed
    cfg.dropout_rate = 0.2
    cfg.feature_dim = 512

    # Only pathology and depth tasks
    cfg.num_classes = {'pathology': 2, 'depth': 2}
    cfg.task_weights = {'pathology': 1.0, 'depth': 1.0}

    # DDP parameters
    cfg.distributed = world_size > 1
    cfg.world_size = world_size
    cfg.rank = rank
    cfg.local_rank = rank
    if torch.cuda.is_available() and world_size > 0:
        cfg.dist_backend = 'nccl'
        cfg.device = 'cuda'
    else:
        cfg.dist_backend = 'gloo'
        cfg.device = 'cpu'
    cfg.dist_url = f'tcp://127.0.0.1:{args.port}'

    # Other options
    cfg.mixed_precision = True
    cfg.log_every_n_batches = 50
    cfg.save_best_model = True
    cfg.class_weights = class_weights

    return cfg


def main_worker(rank, args, csv_path, world_size, class_weights, region_name):
    config = build_config(args, rank, world_size, class_weights, csv_path, region_name)
    trainer = DDPTrainer(config)
    trainer.train_cross_validation()
    trainer.cleanup()


def run_region(args, region_name, df_region):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        world_size = 1

    tasks = ['pathology', 'depth']
    class_weights = get_class_weight_tensors(df_region, tasks)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        df_region.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    mp.spawn(main_worker, nprocs=world_size,
             args=(args, csv_path, world_size, class_weights, region_name))

    os.remove(csv_path)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    if 'region' not in df.columns:
        raise ValueError("CSV must contain a 'region' column")

    regions = sorted(df['region'].unique())

    for region in regions:
        print(f"\n===== Training region: {region} =====")
        df_region = df[df['region'] == region].reset_index(drop=True)
        run_region(args, region, df_region)


if __name__ == '__main__':
    main()
