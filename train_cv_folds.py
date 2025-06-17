import argparse
import pandas as pd
import torch.multiprocessing as mp
import torch
from types import SimpleNamespace

from src.training.ddp_trainer import DDPTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDP cross-validation using pre-assigned folds")
    parser.add_argument('--csv', required=True, help='Metadata CSV with fold column')
    parser.add_argument('--output-dir', default='runs', help='Directory for models and logs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training init')
    return parser.parse_args()


def build_config(args, rank, n_splits, world_size):
    cfg = SimpleNamespace()
    cfg.csv_path = args.csv  # for logging/reference
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.lr = args.lr
    cfg.scheduler = args.scheduler
    cfg.backbone = args.backbone
    cfg.dropout_rate = 0.2
    cfg.feature_dim = 512
    cfg.n_splits = n_splits

    # Multi-task setup
    cfg.num_classes = {'pathology': 2, 'region': 3, 'depth': 2}
    cfg.task_weights = {'pathology': 1.0, 'region': 1.0, 'depth': 1.0}

    # DDP settings
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

    return cfg


def main_worker(rank, args, df, fold_ids, world_size):
    config = build_config(args, rank, len(fold_ids), world_size)
    trainer = DDPTrainer(config)

    # Prepare folds only on rank 0 then broadcast
    if rank == 0:
        folds = [(
            df[df['fold'] != k].reset_index(drop=True),
            df[df['fold'] == k].reset_index(drop=True)
        ) for k in fold_ids]
    else:
        folds = None

    folds, _ = trainer._broadcast_dataframes(folds, df)

    all_results = []
    for fold_idx, (train_df, val_df) in enumerate(folds):
        test_df = val_df  # no separate test set
        fold_res = trainer.train_fold(fold_idx, train_df, val_df, test_df)
        all_results.append(fold_res)

    if rank == 0:
        trainer._print_cv_results(all_results)

    trainer.cleanup()


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size == 0:
        world_size = 1
    df = pd.read_csv(args.csv)
    if 'fold' not in df.columns:
        raise ValueError("CSV must contain a 'fold' column")
    fold_ids = sorted(df['fold'].unique())
    mp.spawn(main_worker, nprocs=world_size, args=(args, df, fold_ids, world_size))


if __name__ == '__main__':
    main()
