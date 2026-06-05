from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from models import build_model
from tools.common import build_dataloader, get_color_map, load_config
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.visualization import save_segmentation_visualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("visualize_predictions")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset, loader = build_dataloader(cfg, args.split, shuffle=False)
    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, optimizer=None, strict=False, match_shape=False)
    model.eval()

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["experiment"]["output_dir"]) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        logits = outputs["logits"]
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).cpu()
        for i, name in enumerate(batch["name"]):
            save_segmentation_visualization(
                image=images[i].cpu(),
                gt_mask=masks[i].cpu(),
                pred_mask=preds[i],
                out_dir=out_dir,
                name=name,
                color_map=get_color_map(cfg),
                overlay=True,
            )
            count += 1
            if count >= args.num_samples:
                logger.info("saved %d visualizations to %s", count, out_dir)
                return
    logger.info("saved %d visualizations to %s", count, out_dir)


if __name__ == "__main__":
    main()
