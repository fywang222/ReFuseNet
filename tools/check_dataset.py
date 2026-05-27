from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.common import build_dataloader, load_config
from utils.logger import setup_logger
from utils.visualization import save_segmentation_visualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("check_dataset")
    dataset, loader = build_dataloader(cfg, args.split, shuffle=False)
    class_names = dataset.dataset.class_names if hasattr(dataset, "dataset") else dataset.class_names

    logger.info("dataset size: %d", len(dataset))
    out_dir = Path(cfg["experiment"]["output_dir"]) / "dataset_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = Counter()
    for idx, batch in enumerate(loader):
        images = batch["image"]
        masks = batch["mask"]
        logger.info("sample[%d] image shape: %s | mask shape: %s", idx, tuple(images[0].shape), tuple(masks[0].shape))
        uniques = torch.unique(masks[0]).cpu().tolist()
        logger.info("sample[%d] mask unique values: %s", idx, uniques)
        valid = masks[0] != 255
        counts = torch.bincount(masks[0][valid].flatten(), minlength=len(class_names)).cpu().tolist()
        for class_idx, count in enumerate(counts):
            hist[class_idx] += int(count)
        save_segmentation_visualization(
            image=images[0],
            gt_mask=masks[0],
            pred_mask=masks[0],
            out_dir=out_dir,
            name=f"{args.split}_{idx}",
        )
        if idx + 1 >= args.num_samples:
            break

    logger.info("class histogram:")
    for class_idx, name in enumerate(class_names):
        logger.info("%s: %d", name, hist[class_idx])


if __name__ == "__main__":
    main()

