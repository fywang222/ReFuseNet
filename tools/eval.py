from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models import build_model
from tools.common import build_dataloader, build_metric, evaluate_model, format_metrics, load_config
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("eval")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset, loader = build_dataloader(cfg, args.split, shuffle=False)
    metric = build_metric(cfg, dataset)
    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, optimizer=None, strict=False, match_shape=False)

    save_dir = None
    if args.save_pred:
        save_dir = Path(cfg["experiment"]["output_dir"]) / "predictions" / args.split

    metrics = evaluate_model(model, loader, metric, device, save_dir=save_dir)
    class_names = dataset.dataset.class_names if hasattr(dataset, "dataset") else dataset.class_names
    logger.info(format_metrics(metrics, class_names))


if __name__ == "__main__":
    main()

