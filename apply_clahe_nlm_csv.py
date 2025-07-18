import argparse
import os
from typing import Optional

import cv2
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm


def apply_filters_cv(
    path: str,
    clahe: cv2.CLAHE,
    h: float,
    gpu_id: Optional[int] = None,
) -> Optional[cv2.Mat]:
    """Apply CLAHE and Fast Non-local Means using OpenCV.

    Parameters
    ----------
    path: str
        Path to the input image.
    clahe: cv2.CLAHE
        Preconstructed CLAHE object.
    h: float
        Filter strength parameter. Should be scaled for 8-bit images.
    gpu_id: Optional[int]
        GPU index to run CUDA denoising on. If ``None`` or CUDA is unavailable,
        CPU denoising is used instead.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = clahe.apply(img)
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        gmat = cv2.cuda_GpuMat()
        gmat.upload(img)
        out = cv2.cuda.fastNlMeansDenoising(gmat, None, h)
        img = out.download()
    else:
        img = cv2.fastNlMeansDenoising(img, None, h)
    return img


def _process_dataframe(
    df: pd.DataFrame,
    base_dir: str,
    output_dir: str,
    clahe: cv2.CLAHE,
    h: float,
    gpu_id: Optional[int] = None,
) -> None:
    for _, row in tqdm(df.iterrows(), total=len(df), position=gpu_id or 0):
        inp_path = row["filepath"]
        result = apply_filters_cv(inp_path, clahe, h, gpu_id)
        if result is None:
            continue
        rel_path = os.path.relpath(inp_path, base_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, result)


def process_csv(
    csv_path: str, output_dir: str, clip_limit: float, h: float
) -> None:
    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns:
        raise ValueError("CSV must contain a 'filepath' column")

    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.commonpath(df["filepath"].tolist())

    clahe = cv2.createCLAHE(clipLimit=clip_limit)

    _process_dataframe(df, base_dir, output_dir, clahe, h * 255.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply CLAHE and NLM to images from a CSV while preserving the "
            "directory structure"
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with a 'filepath' column",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--clip-limit", type=float, default=0.2, help="CLAHE clip limit"
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.15,
        help="NLM filter strength",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs in parallel",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.multi_gpu and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()

        def _worker(rank: int, args):
            df = pd.read_csv(args.csv)
            base_dir = os.path.commonpath(df["filepath"].tolist())
            clahe = cv2.createCLAHE(clipLimit=args.clip_limit)
            df = df.iloc[rank::world_size].reset_index(drop=True)
            _process_dataframe(
                df,
                base_dir,
                args.output_dir,
                clahe,
                args.h * 255.0,
                gpu_id=rank,
            )

        mp.spawn(_worker, args=(args,), nprocs=world_size)
    else:
        process_csv(args.csv, args.output_dir, args.clip_limit, args.h)


if __name__ == "__main__":
    main()
