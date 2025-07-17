import argparse
import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage import exposure, restoration
from tqdm import tqdm


def apply_filters(
    img: Image.Image, clip_limit: float = 0.2, h: float = 0.15
) -> Image.Image:
    """Apply CLAHE and Non-local Means filtering to a PIL Image."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    arr = arr.astype(np.float32) / 255.0
    arr = exposure.equalize_adapthist(arr, clip_limit=clip_limit)
    arr = restoration.denoise_nl_means(arr, h=h, fast_mode=True)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def process_csv(csv_path: str, output_dir: str, clip_limit: float, h: float):
    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns:
        raise ValueError("CSV must contain a 'filepath' column")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        inp_path = row["filepath"]
        img = Image.open(inp_path)
        filtered = apply_filters(img, clip_limit=clip_limit, h=h)
        out_path = os.path.join(output_dir, os.path.basename(inp_path))
        filtered.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply CLAHE and NLM to images from a CSV"
    )
    parser.add_argument("--csv", required=True, help="CSV with a 'filepath' column")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save processed images"
    )
    parser.add_argument(
        "--clip-limit", type=float, default=0.2, help="CLAHE clip limit"
    )
    parser.add_argument("--h", type=float, default=0.15, help="NLM filter strength")
    return parser.parse_args()


def main():
    args = parse_args()
    process_csv(args.csv, args.output_dir, args.clip_limit, args.h)


if __name__ == "__main__":
    main()
