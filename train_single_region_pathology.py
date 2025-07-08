import argparse
import pandas as pd
import torch
import torch.multiprocessing as mp
from types import SimpleNamespace
import tempfile
import os
import numpy as np

from src.data.utils import get_class_weight_tensors
from src.training.ddp_trainer import DDPTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validation for a single region using pathology only"
    )
    parser.add_argument('--train-val-csv', required=True,
                        help='CSV with training and validation data')
    parser.add_argument('--test-csv', required=True,
                        help='CSV with held-out test data')
    parser.add_argument('--region', required=True, help='Region name to train on')
    parser.add_argument('--output-dir', default='runs', help='Base directory for models and logs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for the CNN')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Number of random search trials')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='Lower bound for learning rate search')
    parser.add_argument('--lr-max', type=float, default=1e-3,
                        help='Upper bound for learning rate search')
    parser.add_argument('--dropout-min', type=float, default=0.1,
                        help='Lower bound for dropout search')
    parser.add_argument('--dropout-max', type=float, default=0.5,
                        help='Upper bound for dropout search')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training init')
    return parser.parse_args()


def build_config(args, rank, world_size, class_weights,
                 train_val_path, test_path, region_name,
                 lr=None, dropout=None):
    cfg = SimpleNamespace()
    cfg.train_val_csv_path = train_val_path
    cfg.test_csv_path = test_path
    cfg.output_dir = os.path.join(args.output_dir, f"region_{region_name}")
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.lr = args.lr if lr is None else lr
    cfg.scheduler = args.scheduler
    cfg.backbone = args.backbone
    cfg.n_splits = args.n_splits
    cfg.random_seed = args.seed
    cfg.dropout_rate = args.dropout if dropout is None else dropout
    cfg.feature_dim = 512

    cfg.num_classes = {'pathology': 2}
    cfg.task_weights = {'pathology': 1.0}

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


def main_worker(rank, args, train_val_path, test_path, world_size, class_weights,
                region_name, lr, dropout, result_queue):
    config = build_config(args, rank, world_size, class_weights,
                          train_val_path, test_path, region_name,
                          lr=lr, dropout=dropout)
    trainer = DDPTrainer(config)
    results = trainer.train_cross_validation()
    trainer.cleanup()
    if rank == 0 and result_queue is not None:
        avg_score = float(np.mean([fold['pathology'] for fold in results]))
        result_queue.put(avg_score)


def run_region(args, region_name, df_trainval_reg, df_test_reg, lr, dropout):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        world_size = 1

    tasks = ['pathology']
    class_weights = get_class_weight_tensors(df_trainval_reg, tasks)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_trainval, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_test:
        df_trainval_reg.to_csv(tmp_trainval.name, index=False)
        df_test_reg.to_csv(tmp_test.name, index=False)
        train_path = tmp_trainval.name
        test_path = tmp_test.name

    result_queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(args, train_path, test_path, world_size, class_weights, region_name,
              lr, dropout, result_queue)
    )

    os.remove(train_path)
    os.remove(test_path)
    return result_queue.get()


def main():
    args = parse_args()
    df_trainval = pd.read_csv(args.train_val_csv)
    df_test = pd.read_csv(args.test_csv)

    for df in (df_trainval, df_test):
        if 'region' not in df.columns:
            raise ValueError("CSVs must contain a 'region' column")

    available_regions = set(df_trainval['region'].unique()) | set(df_test['region'].unique())
    if args.region not in available_regions:
        raise ValueError(f"Region '{args.region}' not found in provided CSVs")

    df_trainval_reg = df_trainval[df_trainval['region'] == args.region].reset_index(drop=True)
    df_test_reg = df_test[df_test['region'] == args.region].reset_index(drop=True)

    def sample_params():
        lr = 10 ** np.random.uniform(np.log10(args.lr_min), np.log10(args.lr_max))
        dropout = np.random.uniform(args.dropout_min, args.dropout_max)
        return lr, dropout

    best_score = -1.0
    best_params = None

    for trial in range(args.num_trials):
        if args.num_trials > 1:
            lr, dropout = sample_params()
        else:
            lr, dropout = args.lr, args.dropout
        print(f"\n===== Trial {trial + 1}/{args.num_trials}: lr={lr:.2e}, dropout={dropout:.2f} =====")
        score = run_region(args, args.region, df_trainval_reg, df_test_reg, lr,
                           dropout)
        print(f"Trial {trial + 1} mean accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = {'lr': lr, 'dropout': dropout}

    print(f"\nBest params: {best_params} -- accuracy {best_score:.4f}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
