import argparse
import pandas as pd
import torch
import torch.multiprocessing as mp
from types import SimpleNamespace
import tempfile
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.data import PathDepthPairDataset
from src.training.ddp_trainer import DDPTrainer


PAIR_MAPPING = {
    ('Control', 'DWM'): 0,
    ('Control', 'SWM'): 1,
    ('ASD', 'DWM'): 2,
    ('ASD', 'SWM'): 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validation per region for single-task CNN"
    )
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


def compute_pair_class_weights(df: pd.DataFrame) -> dict:
    labels = df[['pathology', 'depth']].apply(lambda r: PAIR_MAPPING[(r['pathology'], r['depth'])], axis=1)
    classes = np.arange(len(PAIR_MAPPING))
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {'pair': torch.tensor(weights, dtype=torch.float32)}


def build_config(args, rank, world_size, class_weights, csv_path, region_name):
    cfg = SimpleNamespace()
    cfg.csv_path = csv_path
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

    cfg.num_classes = {'pair': len(PAIR_MAPPING)}
    cfg.task_weights = {'pair': 1.0}
    cfg.model_type = 'single'
    cfg.dataset_cls = PathDepthPairDataset

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

    class_weights = compute_pair_class_weights(df_region)

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
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

