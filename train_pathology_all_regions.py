import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from src.data.utils import create_splits, get_class_weight_tensors
from src.training.ddp_trainer import DDPTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Random search on the first fold followed by full cross-"
            "validation for pathology classification"
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Metadata CSV file. When --test-csv is provided this should "
            "point to the train/validation CSV"
        ),
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Optional CSV containing a held-out test set",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Base directory for models and logs",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate for the CNN"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "step", "none"],
        help="Learning rate scheduler type",
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction reserved for the test set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Search trials",
    )
    parser.add_argument(
        "--lr-min", type=float, default=1e-5, help="Lower bound for LR search"
    )
    parser.add_argument(
        "--lr-max", type=float, default=1e-3, help="Upper bound for LR search"
    )
    parser.add_argument(
        "--dropout-min", type=float, default=0.1, help="Lower dropout bound"
    )
    parser.add_argument(
        "--dropout-max", type=float, default=0.5, help="Upper dropout bound"
    )
    parser.add_argument(
        "--port", type=int, default=12355, help="Port for distributed init"
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable PyTorch AMP mixed precision",
    )
    parser.set_defaults(mixed_precision=True)
    return parser.parse_args()


def build_config(
    args,
    rank,
    world_size,
    class_weights,
    csv_path=None,
    test_csv_path=None,
    lr=None,
    dropout=None,
):
    cfg = SimpleNamespace()
    # When a separate test CSV is provided, use train_val/test paths.
    if test_csv_path is not None:
        cfg.train_val_csv_path = csv_path
        cfg.test_csv_path = test_csv_path
        cfg.csv_path = None
    else:
        cfg.csv_path = csv_path
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.lr = args.lr if lr is None else lr
    cfg.scheduler = args.scheduler
    cfg.backbone = args.backbone
    cfg.n_splits = args.n_splits
    cfg.holdout_frac = args.holdout_frac
    cfg.random_seed = args.seed
    cfg.dropout_rate = args.dropout if dropout is None else dropout
    cfg.feature_dim = 512

    cfg.num_classes = {"pathology": 2}
    cfg.task_weights = {"pathology": 1.0}

    cfg.distributed = world_size > 1
    cfg.world_size = world_size
    cfg.rank = rank
    cfg.local_rank = rank
    if torch.cuda.is_available() and world_size > 0:
        cfg.dist_backend = "nccl"
        cfg.device = "cuda"
    else:
        cfg.dist_backend = "gloo"
        cfg.device = "cpu"
    cfg.dist_url = f"tcp://127.0.0.1:{args.port}"

    cfg.mixed_precision = args.mixed_precision
    cfg.log_every_n_batches = 50
    cfg.save_best_model = True
    cfg.class_weights = class_weights

    return cfg


def main_worker_single_fold(
    rank,
    args,
    train_df,
    val_df,
    test_df,
    world_size,
    class_weights,
    lr,
    dropout,
    result_queue,
):
    config = build_config(
        args,
        rank,
        world_size,
        class_weights,
        None,
        lr,
        dropout,
    )
    trainer = DDPTrainer(config)
    fold_results = trainer.train_fold(0, train_df, val_df, test_df)
    trainer.cleanup()
    if rank == 0 and result_queue is not None:
        result_queue.put(float(fold_results["pathology"]))


def run_single_fold(args, train_df, val_df, test_df, df_trainval, lr, dropout):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        world_size = 1

    tasks = ["pathology"]
    class_weights = get_class_weight_tensors(df_trainval, tasks)

    result_queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        main_worker_single_fold,
        nprocs=world_size,
        args=(
            args,
            train_df,
            val_df,
            test_df,
            world_size,
            class_weights,
            lr,
            dropout,
            result_queue,
        ),
    )

    return result_queue.get()


def main_worker_cv(
    rank,
    args,
    csv_path,
    test_csv,
    world_size,
    class_weights,
    lr,
    dropout,
    result_queue,
):
    config = build_config(
        args,
        rank,
        world_size,
        class_weights,
        csv_path,
        test_csv,
        lr,
        dropout,
    )
    trainer = DDPTrainer(config)
    results = trainer.train_cross_validation()
    trainer.cleanup()
    if rank == 0 and result_queue is not None:
        scores = [float(fold["pathology"]) for fold in results]
        result_queue.put(scores)


def run_full_cv(args, csv_path, df_trainval, lr, dropout, test_csv=None):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        world_size = 1

    tasks = ["pathology"]
    class_weights = get_class_weight_tensors(df_trainval, tasks)

    result_queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        main_worker_cv,
        nprocs=world_size,
        args=(
            args,
            csv_path,
            test_csv,
            world_size,
            class_weights,
            lr,
            dropout,
            result_queue,
        ),
    )

    return result_queue.get()


def main():
    args = parse_args()
    log_dir = os.path.join(args.output_dir, "search_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Create patient-grouped splits
    if args.test_csv:
        folds, test_df = create_splits(
            train_val_csv_path=args.csv,
            test_csv_path=args.test_csv,
            n_splits=args.n_splits,
            seed=args.seed,
        )
    else:
        folds, test_df = create_splits(
            csv_path=args.csv,
            n_splits=args.n_splits,
            holdout_frac=args.holdout_frac,
            seed=args.seed,
        )

    train_df_first, val_df_first = folds[0]
    df_trainval = pd.concat([df for pair in folds for df in pair]).reset_index(
        drop=True
    )

    def sample_params():
        lr = 10 ** np.random.uniform(
            np.log10(args.lr_min),
            np.log10(args.lr_max),
        )
        dropout = np.random.uniform(args.dropout_min, args.dropout_max)
        return lr, dropout

    best_score = -1.0
    best_params = None

    for trial in range(args.num_trials):
        if args.num_trials > 1:
            lr, dropout = sample_params()
        else:
            lr, dropout = args.lr, args.dropout
        print(
            f"\n===== Trial {trial + 1}/{args.num_trials}: "
            f"lr={lr:.2e}, dropout={dropout:.2f} ====="
        )
        score = run_single_fold(
            args,
            train_df_first,
            val_df_first,
            test_df,
            df_trainval,
            lr,
            dropout,
        )
        print(f"Trial {trial + 1} fold-1 accuracy: {score:.4f}")
        writer.add_scalar("Trial/Accuracy", score, trial)
        writer.add_scalar("Trial/LR", lr, trial)
        writer.add_scalar("Trial/Dropout", dropout, trial)
        if score > best_score:
            best_score = score
            best_params = {"lr": lr, "dropout": dropout}

    print(f"\nBest params: {best_params} -- fold-1 accuracy {best_score:.4f}")
    writer.add_text("BestParams", str(best_params))

    fold_scores = run_full_cv(
        args,
        args.csv,
        df_trainval,
        best_params["lr"],
        best_params["dropout"],
        test_csv=args.test_csv,
    )
    for idx, fs in enumerate(fold_scores):
        writer.add_scalar("Final/Fold_Accuracy", fs, idx)
    final_score = float(np.mean(fold_scores))
    print(f"\nCross-validation accuracy with best params: {final_score:.4f}")
    writer.add_scalar("Final/CV_Accuracy", final_score)
    writer.add_hparams(
        {"lr": best_params["lr"], "dropout": best_params["dropout"]},
        {"cv_accuracy": final_score},
    )
    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
