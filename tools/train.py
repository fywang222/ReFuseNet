from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from losses import build_loss
from models import build_model
from tools.common import build_dataloader, build_metric, evaluate_model, format_metrics, load_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import setup_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(exp_cfg["name"], out_dir)

    set_seed(exp_cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset, train_loader = build_dataloader(cfg, "train")
    val_dataset, val_loader = build_dataloader(cfg, "val", shuffle=False)
    class_names = train_dataset.dataset.class_names if hasattr(train_dataset, "dataset") else train_dataset.class_names

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda")

    start_epoch = 0
    best_miou = -1.0

    if args.pretrained:
        load_checkpoint(args.pretrained, model, optimizer=None, strict=False, match_shape=True)
        logger.info("Loaded pretrained weights from %s", args.pretrained)

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, strict=False, match_shape=False)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("metrics", {}).get("miou", best_miou))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs["logits"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        metric = build_metric(cfg, train_dataset)
        val_metrics = evaluate_model(model, val_loader, metric, device)
        logger.info(
            "epoch %d/%d | train_loss=%.4f | %s",
            epoch + 1,
            epochs,
            train_loss,
            format_metrics(val_metrics, class_names),
        )

        last_path = out_dir / "last.pth"
        save_checkpoint(last_path, model, optimizer, epoch, val_metrics, extra={"config": cfg})
        if val_metrics["miou"] >= best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, val_metrics, extra={"config": cfg})

    logger.info("Training complete. Best mIoU=%.4f", best_miou)


if __name__ == "__main__":
    main()
