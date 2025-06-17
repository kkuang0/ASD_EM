import argparse
import torch
import torch.multiprocessing as mp
from types import SimpleNamespace

from src.training.ddp_trainer import DDPTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training script")
    parser.add_argument('--csv', required=True, help='Path to metadata CSV')
    parser.add_argument('--output-dir', default='runs', help='Directory for models and logs')
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
                        help='Fraction of data to reserve for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training init')
    return parser.parse_args()


def build_config(args, rank):
    cfg = SimpleNamespace()
    # Dataset and model settings
    cfg.csv_path = args.csv
    cfg.output_dir = args.output_dir
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

    # Multi-task setup
    cfg.num_classes = {'pathology': 2, 'region': 3, 'depth': 2}
    cfg.task_weights = {'pathology': 1.0, 'region': 1.0, 'depth': 1.0}

    # DDP parameters
    cfg.distributed = True
    cfg.world_size = 4
    cfg.rank = rank
    cfg.local_rank = rank
    cfg.dist_backend = 'nccl'
    cfg.dist_url = f'tcp://127.0.0.1:{args.port}'
    cfg.device = 'cuda'

    # Other options
    cfg.mixed_precision = True
    cfg.log_every_n_batches = 50
    cfg.save_best_model = True

    return cfg


def main_worker(rank, args):
    config = build_config(args, rank)
    trainer = DDPTrainer(config)
    trainer.train_cross_validation()
    trainer.cleanup()


def main():
    args = parse_args()
    mp.spawn(main_worker, nprocs=4, args=(args,))


if __name__ == '__main__':
    main()
