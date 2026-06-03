from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", choices=["on", "off"], default=None)
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
    if key in {"last"}:
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


def build_criterion(cfg):
    return nn.CrossEntropyLoss(ignore_index=cfg["dataset"].get("ignore_index", 255))

def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def _format_millions(value: int) -> str:
    return f"{value / 1e6:.2f}M"


def _model_name_from_cfg(cfg: dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "unknown"))
    setting = model_cfg.get("setting")
    if setting is not None:
        name = f"{name}-{setting}"
    return name


def _log_model_summary(logger, model, cfg):
    total, trainable, frozen = _count_parameters(model)
    logger.info(
        "model=%s | params total=%s | trainable=%s | frozen=%s",
        _model_name_from_cfg(cfg),
        _format_millions(total),
        _format_millions(trainable),
        _format_millions(frozen),
    )

def _boundary_targets(masks: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [B,H,W], got {tuple(masks.shape)}")
    valid = masks.ne(ignore_index)
    boundary = torch.zeros_like(valid)

    if masks.shape[-1] > 1:
        horizontal = valid[:, :, :-1] & valid[:, :, 1:] & (masks[:, :, :-1] != masks[:, :, 1:])
        boundary[:, :, :-1] |= horizontal
        boundary[:, :, 1:] |= horizontal
    if masks.shape[-2] > 1:
        vertical = valid[:, :-1, :] & valid[:, 1:, :] & (masks[:, :-1, :] != masks[:, 1:, :])
        boundary[:, :-1, :] |= vertical
        boundary[:, 1:, :] |= vertical

    return boundary.unsqueeze(1).float(), valid.unsqueeze(1).float()


def _compute_total_loss(cfg: dict[str, Any], outputs: dict[str, torch.Tensor], masks: torch.Tensor, criterion):
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    total = criterion(outputs["logits"], masks)
    parts: dict[str, torch.Tensor] = {"primary": total}

    if "coarse_logits" in outputs:
        aux_weight = float(cfg["train"].get("lambda_aux", 0.4))
        coarse_logits = outputs["coarse_logits"]
        if coarse_logits.shape[-2:] != masks.shape[-2:]:
            coarse_logits = F.interpolate(coarse_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        aux_loss = criterion(coarse_logits, masks)
        total = total + aux_weight * aux_loss
        parts["aux"] = aux_loss

    if "boundary_logits" in outputs:
        boundary_weight = float(cfg["train"].get("lambda_boundary", 1.0))
        boundary_logits = outputs["boundary_logits"]
        if boundary_logits.shape[-2:] != masks.shape[-2:]:
            boundary_logits = F.interpolate(boundary_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        boundary_target, valid_mask = _boundary_targets(masks, ignore_index)
        boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target, reduction="none")
        boundary_loss = (boundary_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        total = total + boundary_weight * boundary_loss
        parts["boundary"] = boundary_loss

    parts["total"] = total
    return total, parts


def _wandb_enabled(cfg: dict[str, Any], arg_value: str | None) -> bool:
    if arg_value is not None:
        return arg_value == "on"
    return bool(cfg.get("wandb", {}).get("enabled", True))


def _init_wandb(cfg: dict[str, Any], enabled: bool, out_dir: Path, logger):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled by default. Install wandb or run with --wandb off.") from exc

    wandb_cfg = cfg.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", os.environ.get("WANDB_PROJECT", "refusenet")),
        name=wandb_cfg.get("name", cfg["experiment"]["name"]),
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        config=cfg,
        dir=str(out_dir),
        mode=wandb_cfg.get("mode", "online"),
    )
    logger.info("wandb enabled: project=%s | run=%s", run.project, run.name)
    return wandb


def _log_wandb(
    wandb_module,
    step: int,
    payload: dict[str, Any],
):
    if wandb_module is None:
        return
    wandb_module.log(payload, step=step)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(exp_cfg["name"], out_dir)
    wandb_module = _init_wandb(cfg, _wandb_enabled(cfg, args.wandb), out_dir, logger)

    set_seed(exp_cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset, train_loader = build_dataloader(cfg, "train")
    val_dataset, val_loader = build_dataloader(cfg, "val", shuffle=False)
    class_names = train_dataset.dataset.class_names if hasattr(train_dataset, "dataset") else train_dataset.class_names

    model = build_model(cfg).to(device)
    _log_model_summary(logger, model, cfg)
    criterion = build_criterion(cfg).to(device)
    param_groups = model.get_param_groups(cfg) if hasattr(model, "get_param_groups") else model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg["train"].get("lr", cfg["train"].get("lr_decoder", 1.0e-4)),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda")

    start_epoch = 0
    last_metrics: dict[str, Any] = {}
    last_eval_epoch = None

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
        if isinstance(checkpoint.get("metrics"), dict):
            last_metrics = checkpoint["metrics"]
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

        extra = checkpoint.get("extra", {}) if isinstance(checkpoint.get("extra", {}), dict) else {}
        last_eval_epoch = extra.get("last_eval_epoch")

    epochs = int(args.epochs or cfg["train"]["epochs"])

    save_every = args.save_every
    if save_every is None:
        save_every = int(cfg["train"].get("save_every", 0) or 0)

    eval_every = args.eval_every
    if eval_every is None:
        eval_every = int(cfg["train"].get("eval_every", 1) or 1)

    log_every_steps = args.log_every_steps
    if log_every_steps is None:
        log_every_steps = int(cfg["train"].get("log_every_steps", 50) or 0)

    epoch_ckpt_dir = out_dir / "checkpoints"
    global_step = start_epoch * len(train_loader)

    logger.info(
        "schedule | epochs=%d | log_every_steps=%d | eval_every=%d | save_every=%d",
        epochs,
        log_every_steps,
        eval_every,
        save_every,
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        running_total = 0.0
        running_primary = 0.0
        running_aux = 0.0
        running_boundary = 0.0
        saw_aux = False
        saw_boundary = False

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                loss, parts = _compute_total_loss(cfg, outputs, masks, criterion)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_value = float(parts["total"].item())
            primary_value = float(parts["primary"].item())

            running_total += total_value
            running_primary += primary_value

            aux_value = None
            boundary_value = None

            if "aux" in parts:
                aux_value = float(parts["aux"].item())
                running_aux += aux_value
                saw_aux = True

            if "boundary" in parts:
                boundary_value = float(parts["boundary"].item())
                running_boundary += boundary_value
                saw_boundary = True

            if log_every_steps > 0 and global_step % log_every_steps == 0:
                msg = (
                    f"epoch {epoch + 1}/{epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"global_step={global_step} "
                    f"loss={total_value:.4f} "
                    f"primary={primary_value:.4f}"
                )
                if aux_value is not None:
                    msg += f" aux={aux_value:.4f}"
                if boundary_value is not None:
                    msg += f" boundary={boundary_value:.4f}"
                logger.info(msg)

                wandb_payload = {
                    "train/step_loss": total_value,
                    "train/step_primary_loss": primary_value,
                    "epoch": epoch + 1,
                }
                if aux_value is not None:
                    wandb_payload["train/step_aux_loss"] = aux_value
                if boundary_value is not None:
                    wandb_payload["train/step_boundary_loss"] = boundary_value
                _log_wandb(wandb_module, global_step, wandb_payload)

        num_batches = max(len(train_loader), 1)
        train_loss = running_total / num_batches
        train_primary_loss = running_primary / num_batches
        train_aux_loss = running_aux / num_batches if saw_aux else None
        train_boundary_loss = running_boundary / num_batches if saw_boundary else None

        completed_epoch = epoch + 1
        do_eval = (eval_every > 0 and completed_epoch % eval_every == 0) or completed_epoch == epochs

        loss_msg = f"train_loss={train_loss:.4f} | train_primary={train_primary_loss:.4f}"
        if train_aux_loss is not None:
            loss_msg += f" | train_aux={train_aux_loss:.4f}"
        if train_boundary_loss is not None:
            loss_msg += f" | train_boundary={train_boundary_loss:.4f}"

        val_metrics = None
        if do_eval:
            metric = build_metric(cfg, val_dataset)
            val_metrics = evaluate_model(model, val_loader, metric, device, cfg=cfg)
            last_metrics = val_metrics
            last_eval_epoch = completed_epoch
            logger.info(
                "epoch %d/%d | %s | %s",
                completed_epoch,
                epochs,
                loss_msg,
                format_metrics(val_metrics, class_names),
            )
        else:
            logger.info(
                "epoch %d/%d | %s | eval skipped",
                completed_epoch,
                epochs,
                loss_msg,
            )

        epoch_payload = {
            "epoch": completed_epoch,
            "train/loss": train_loss,
            "train/loss_primary": train_primary_loss,
        }
        if train_aux_loss is not None:
            epoch_payload["train/loss_aux"] = train_aux_loss
        if train_boundary_loss is not None:
            epoch_payload["train/loss_boundary"] = train_boundary_loss
        if val_metrics is not None:
            epoch_payload["val/miou"] = val_metrics["miou"]
            epoch_payload["val/pixel_acc"] = val_metrics["pixel_acc"]
            epoch_payload["val/mean_acc"] = val_metrics["mean_acc"]
            if "rare_miou" in val_metrics:
                epoch_payload["val/rare_miou"] = val_metrics["rare_miou"]

        _log_wandb(wandb_module, global_step, epoch_payload)

        extra = {
            "config": cfg,
            "train_loss": train_loss,
            "global_step": global_step,
            "last_eval_epoch": last_eval_epoch,
        }

        checkpoint_metrics = last_metrics if last_metrics is not None else {}

        last_path = out_dir / "last.pth"
        save_checkpoint(last_path, model, optimizer, epoch, checkpoint_metrics, extra=extra, scaler=scaler)

        if save_every > 0 and completed_epoch % save_every == 0:
            epoch_path = epoch_ckpt_dir / f"epoch_{completed_epoch:04d}.pth"
            save_checkpoint(epoch_path, model, optimizer, epoch, checkpoint_metrics, extra=extra, scaler=scaler)
            logger.info("saved checkpoint %s", epoch_path)

    logger.info("Training complete. Last epoch=%d", epochs)
    if wandb_module is not None:
        wandb_module.finish()


if __name__ == "__main__":
    main()
