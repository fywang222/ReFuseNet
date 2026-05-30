from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--save-epochs", default=None, help="Comma-separated 1-based epochs to save, e.g. 10,20,40.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _parse_epoch_set(value):
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    text = str(value).strip()
    if not text:
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def _resolve_resume_checkpoint(resume, out_dir):
    if resume is None:
        return None
    path = Path(resume)
    if path.exists():
        return path

    key = str(resume).strip().lower()
    if key in {"last", "best"}:
        candidate = out_dir / f"{key}.pth"
        if candidate.exists():
            return candidate

    try:
        epoch = int(key)
    except ValueError:
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}") from None

    candidates = [
        out_dir / "checkpoints" / f"epoch_{epoch:04d}.pth",
        out_dir / "checkpoints" / f"epoch_{epoch}.pth",
        out_dir / f"epoch_{epoch:04d}.pth",
        out_dir / f"epoch_{epoch}.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Resume epoch {epoch} not found. Searched: {searched}")


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
    param_groups = model.get_param_groups(cfg) if hasattr(model, "get_param_groups") else model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg["train"].get("lr", cfg["train"].get("lr_decoder", 1.0e-4)),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda")

    start_epoch = 0
    best_miou = -1.0

    if args.pretrained:
        load_checkpoint(args.pretrained, model, optimizer=None, strict=False, match_shape=True)
        logger.info("Loaded pretrained weights from %s", args.pretrained)

    if args.resume:
        resume_path = _resolve_resume_checkpoint(args.resume, out_dir)
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            strict=False,
            match_shape=False,
            scaler=scaler,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        metrics_miou = float(checkpoint.get("metrics", {}).get("miou", best_miou))
        extra = checkpoint.get("extra", {}) if isinstance(checkpoint.get("extra", {}), dict) else {}
        saved_best_miou = float(extra.get("best_miou", metrics_miou))
        best_miou = max(metrics_miou, saved_best_miou)
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    epochs = int(args.epochs or cfg["train"]["epochs"])
    save_every = args.save_every
    if save_every is None:
        save_every = int(cfg["train"].get("save_every", 0) or 0)
    save_epochs = _parse_epoch_set(args.save_epochs if args.save_epochs is not None else cfg["train"].get("save_epochs"))
    epoch_ckpt_dir = out_dir / "checkpoints"
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                logits = outputs["logits"]
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        metric = build_metric(cfg, val_dataset)
        val_metrics = evaluate_model(model, val_loader, metric, device)
        logger.info(
            "epoch %d/%d | train_loss=%.4f | %s",
            epoch + 1,
            epochs,
            train_loss,
            format_metrics(val_metrics, class_names),
        )

        completed_epoch = epoch + 1
        is_best = val_metrics["miou"] >= best_miou
        if is_best:
            best_miou = val_metrics["miou"]

        extra = {"config": cfg, "best_miou": best_miou, "train_loss": train_loss}
        last_path = out_dir / "last.pth"
        save_checkpoint(last_path, model, optimizer, epoch, val_metrics, extra=extra, scaler=scaler)
        if is_best:
            save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, val_metrics, extra=extra, scaler=scaler)
        if (save_every > 0 and completed_epoch % save_every == 0) or completed_epoch in save_epochs:
            epoch_path = epoch_ckpt_dir / f"epoch_{completed_epoch:04d}.pth"
            save_checkpoint(epoch_path, model, optimizer, epoch, val_metrics, extra=extra, scaler=scaler)
            logger.info("saved checkpoint %s", epoch_path)

    logger.info("Training complete. Best mIoU=%.4f", best_miou)


if __name__ == "__main__":
    main()
