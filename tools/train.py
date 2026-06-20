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
    parser.add_argument("--s5-debug", action="store_true")
    return parser.parse_args()


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


def _optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    return (num_batches + grad_accum_steps - 1) // grad_accum_steps


def _set_initial_lrs(optimizer) -> None:
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])


def _lr_factor(train_cfg: dict[str, Any], optimizer_step: int, total_steps: int) -> float:
    scheduler_cfg = train_cfg.get("scheduler", {}) or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"none", "constant"}:
        return 1.0
    if scheduler_type != "poly":
        raise ValueError(f"Unsupported train.scheduler.type: {scheduler_type}")

    total_steps = max(int(total_steps), 1)
    warmup_iters = int(scheduler_cfg.get("warmup_iters", 0) or 0)
    warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 1.0e-6))
    power = float(scheduler_cfg.get("power", 1.0))
    min_factor = float(scheduler_cfg.get("min_factor", 0.0))

    if warmup_iters > 0 and optimizer_step < warmup_iters:
        alpha = optimizer_step / max(warmup_iters, 1)
        return warmup_ratio + alpha * (1.0 - warmup_ratio)

    progress = (optimizer_step - warmup_iters) / max(total_steps - warmup_iters, 1)
    factor = (1.0 - min(max(progress, 0.0), 1.0)) ** power
    return max(factor, min_factor)


def _apply_lr_schedule(optimizer, train_cfg: dict[str, Any], optimizer_step: int, total_steps: int) -> float:
    factor = _lr_factor(train_cfg, optimizer_step, total_steps)
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * factor
    return factor


def _format_group_lrs(optimizer) -> str:
    return ", ".join(
        f"{group.get('name', f'group{index}')}={group['lr']:.2e}"
        for index, group in enumerate(optimizer.param_groups)
    )


def _base_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _dataset_sample_preview(dataset, limit: int = 3) -> list[tuple[str, str]]:
    base_dataset = _base_dataset(dataset)
    samples = list(getattr(base_dataset, "samples", []))
    if hasattr(dataset, "indices"):
        samples = [samples[index] for index in list(dataset.indices)[:limit]]
    else:
        samples = samples[:limit]
    return [(str(image_path), str(mask_path)) for image_path, mask_path in samples]


def _log_dataset_summary(logger, split: str, dataset) -> None:
    logger.info("dataset %s | samples=%d", split, len(dataset))
    base_dataset = _base_dataset(dataset)
    if hasattr(base_dataset, "label_format"):
        logger.info("dataset %s | label_format=%s", split, base_dataset.label_format)
    for index, (image_path, mask_path) in enumerate(_dataset_sample_preview(dataset), start=1):
        logger.info("dataset %s preview %d | image=%s | mask=%s", split, index, image_path, mask_path)


def _boundary_targets(masks: torch.Tensor, ignore_index: int, dilation: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
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

    boundary = boundary.unsqueeze(1).float()
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=dilation)

    return boundary, valid.unsqueeze(1).float()


def _compute_total_loss(cfg: dict[str, Any], outputs: dict[str, torch.Tensor], masks: torch.Tensor, criterion):
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    total = criterion(outputs["logits"], masks)
    parts: dict[str, torch.Tensor] = {"primary": total}

    if "coarse_logits" in outputs:
        aux_weight = float(cfg["train"].get("lambda_aux", 0.0))
        if aux_weight > 0.0:
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
        boundary_dilation = int(cfg["train"].get("boundary_dilation", 3))
        boundary_target, valid_mask = _boundary_targets(masks, ignore_index, dilation=boundary_dilation)
        positive = (boundary_target * valid_mask).sum()
        negative = ((1.0 - boundary_target) * valid_mask).sum()
        pos_weight = negative / positive.clamp_min(1.0)
        boundary_loss = F.binary_cross_entropy_with_logits(
            boundary_logits,
            boundary_target,
            pos_weight=pos_weight,
            reduction="none",
        )
        boundary_loss = (boundary_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        total = total + boundary_weight * boundary_loss
        parts["boundary"] = boundary_loss

    parts["total"] = total
    return total, parts


def _resize_logits_to_masks(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _compute_s5_debug_loss(cfg: dict[str, Any], outputs: dict[str, Any], masks: torch.Tensor, criterion):
    total = criterion(_resize_logits_to_masks(outputs["logits"], masks), masks)
    parts: dict[str, torch.Tensor] = {"primary": total}

    refine_weight = float(cfg["train"].get("lambda_refine_aux", 0.1))
    refined_logits = outputs.get("aux_logits", []) or []
    coarse_logits = outputs.get("coarse_logits")
    supervised_aux = ([coarse_logits] if coarse_logits is not None else []) + list(refined_logits[:-1])
    for index, aux in enumerate(supervised_aux, start=1):
        aux_loss = criterion(_resize_logits_to_masks(aux, masks), masks)
        total = total + refine_weight * aux_loss
        parts[f"refine_aux_{index}"] = aux_loss

    previous = coarse_logits
    for index, aux in enumerate(refined_logits, start=1):
        if previous is not None:
            parts[f"refine_delta_abs_{index}"] = (aux.detach() - previous.detach()).abs().mean()
        previous = aux

    coarse_weight = float(cfg["train"].get("lambda_coarse_aux", 0.0))
    if coarse_weight > 0.0 and coarse_logits is not None:
        coarse_loss = criterion(_resize_logits_to_masks(coarse_logits, masks), masks)
        total = total + coarse_weight * coarse_loss
        parts["coarse_aux"] = coarse_loss
    if coarse_logits is not None:
        parts["final_vs_coarse_delta_abs"] = (
            outputs["logits"].detach() - coarse_logits.detach()
        ).abs().mean()

    parts["total"] = total
    return total, parts


def _is_s5_debug_enabled(args, cfg: dict[str, Any]) -> bool:
    setting = str(cfg.get("model", {}).get("setting", "")).upper()
    return bool(args.s5_debug and setting == "S5")


def _module_param_count(module) -> tuple[int, int]:
    if module is None:
        return 0, 0
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def _log_s5_debug_param_summary(logger, model):
    decoder_total, decoder_trainable = _module_param_count(getattr(model, "decoder", None))
    refiner_total, refiner_trainable = _module_param_count(getattr(model, "refiner", None))
    classifier_total, classifier_trainable = _module_param_count(getattr(model, "classifier", None))
    logger.info("decoder params total=%s trainable=%s", _format_millions(decoder_total), _format_millions(decoder_trainable))
    logger.info(
        "refiner params total/trainable=%s/%s",
        _format_millions(refiner_total),
        _format_millions(refiner_trainable),
    )
    logger.info(
        "classifier params total=%s trainable=%s",
        _format_millions(classifier_total),
        _format_millions(classifier_trainable),
    )


def _refiner_grad_stats(model, grad_scale: float = 1.0) -> tuple[float, int]:
    refiner = getattr(model, "refiner", None)
    if refiner is None:
        return 0.0, 0
    sq_sum = 0.0
    tensors = 0
    grad_scale = max(float(grad_scale), 1.0)
    for param in refiner.parameters():
        if param.grad is None:
            continue
        tensors += 1
        grad = param.grad.detach().float() / grad_scale
        sq_sum += float(grad.pow(2).sum().item())
    return sq_sum ** 0.5, tensors


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


@torch.no_grad()
def _forward_s5_debug_outputs(model, images):
    outputs = model(images)
    if not isinstance(outputs, dict) or "logits" not in outputs:
        raise TypeError("Model forward must return a dict with a 'logits' tensor.")
    return outputs


@torch.no_grad()
def _sliding_window_s5_debug_outputs(model, images, crop_size=(1024, 1024), stride=(768, 768)):
    crop_h, crop_w = crop_size
    stride_h, stride_w = stride
    batch, _, height, width = images.shape
    count = images.new_zeros((batch, 1, height, width))
    sums: dict[str, Any] | None = None

    h_grids = max(height - crop_h + stride_h - 1, 0) // stride_h + 1
    w_grids = max(width - crop_w + stride_w - 1, 0) // stride_w + 1
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            bottom = min(h_idx * stride_h + crop_h, height)
            right = min(w_idx * stride_w + crop_w, width)
            top = max(bottom - crop_h, 0)
            left = max(right - crop_w, 0)
            crop = images[:, :, top:bottom, left:right]

            crop_outputs = _forward_s5_debug_outputs(model, crop)
            crop_logits = crop_outputs["logits"]
            crop_coarse = crop_outputs["coarse_logits"]
            crop_aux = list(crop_outputs.get("aux_logits", []))
            region_size = (bottom - top, right - left)
            if crop_logits.shape[-2:] != region_size:
                crop_logits = F.interpolate(crop_logits, size=region_size, mode="bilinear", align_corners=False)
            if crop_coarse.shape[-2:] != region_size:
                crop_coarse = F.interpolate(crop_coarse, size=region_size, mode="bilinear", align_corners=False)
            crop_aux = [
                F.interpolate(aux, size=region_size, mode="bilinear", align_corners=False)
                if aux.shape[-2:] != region_size
                else aux
                for aux in crop_aux
            ]

            if sums is None:
                num_classes = crop_logits.shape[1]
                sums = {
                    "logits": images.new_zeros((batch, num_classes, height, width)),
                    "coarse_logits": images.new_zeros((batch, num_classes, height, width)),
                    "aux_logits": [
                        images.new_zeros((batch, num_classes, height, width))
                        for _ in crop_aux
                    ],
                }

            pad = (left, width - right, top, height - bottom)
            sums["logits"] += F.pad(crop_logits, pad)
            sums["coarse_logits"] += F.pad(crop_coarse, pad)
            for index, aux in enumerate(crop_aux):
                sums["aux_logits"][index] += F.pad(aux, pad)
            count[:, :, top:bottom, left:right] += 1

    if sums is None:
        raise RuntimeError("No sliding-window crops were evaluated.")
    if (count == 0).any():
        raise RuntimeError("S5 debug sliding-window inference produced uncovered pixels.")
    sums["logits"] = sums["logits"] / count.clamp_min(1)
    sums["coarse_logits"] = sums["coarse_logits"] / count.clamp_min(1)
    sums["aux_logits"] = [aux / count.clamp_min(1) for aux in sums["aux_logits"]]
    return sums


@torch.no_grad()
def _evaluate_s5_debug(model, loader, metric_builder, device, cfg):
    model.eval()
    eval_cfg = cfg.get("eval", {})
    inference = eval_cfg.get("inference", "sliding")
    crop_size = tuple(eval_cfg.get("sliding_crop_size", (1024, 1024)))
    stride = tuple(eval_cfg.get("sliding_stride", (768, 768)))

    final_metric = metric_builder()
    coarse_metric = metric_builder()
    aux_metrics = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if inference == "sliding":
            outputs = _sliding_window_s5_debug_outputs(model, images, crop_size=crop_size, stride=stride)
        elif inference == "whole":
            outputs = _forward_s5_debug_outputs(model, images)
        else:
            raise ValueError(f"Unsupported eval.inference: {inference}")

        final_metric.update(_resize_logits_to_masks(outputs["logits"], masks), masks)
        coarse_metric.update(_resize_logits_to_masks(outputs["coarse_logits"], masks), masks)
        aux_logits = outputs.get("aux_logits", []) or []
        if aux_metrics is None:
            aux_metrics = [metric_builder() for _ in aux_logits]
        for metric, aux in zip(aux_metrics, aux_logits):
            metric.update(_resize_logits_to_masks(aux, masks), masks)

    final = final_metric.compute()
    coarse = coarse_metric.compute()
    aux_results = [metric.compute() for metric in (aux_metrics or [])]
    debug_metrics = {
        "coarse_mIoU": coarse["miou"],
        "final_mIoU": final["miou"],
    }
    for index, result in enumerate(aux_results, start=1):
        debug_metrics[f"aux_{index}_mIoU"] = result["miou"]
    return final, debug_metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    s5_debug = _is_s5_debug_enabled(args, cfg)
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
    _log_dataset_summary(logger, "train", train_dataset)
    _log_dataset_summary(logger, "val", val_dataset)

    model = build_model(cfg).to(device)
    _log_model_summary(logger, model, cfg)
    if s5_debug:
        _log_s5_debug_param_summary(logger, model)
    criterion = build_criterion(cfg).to(device)
    param_groups = model.get_param_groups(cfg) if hasattr(model, "get_param_groups") else model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg["train"].get("lr", cfg["train"].get("lr_decoder", 1.0e-4)),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    _set_initial_lrs(optimizer)
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
        resumed_optimizer_step = extra.get("optimizer_step")

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
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1) or 1)
    if grad_accum_steps < 1:
        raise ValueError(f"train.grad_accum_steps must be >= 1, got {grad_accum_steps}")

    epoch_ckpt_dir = out_dir / "checkpoints"
    global_step = start_epoch * len(train_loader)
    steps_per_epoch = _optimizer_steps_per_epoch(len(train_loader), grad_accum_steps)
    total_optimizer_steps = max(epochs * steps_per_epoch, 1)
    optimizer_step = start_epoch * steps_per_epoch
    if "resumed_optimizer_step" in locals() and resumed_optimizer_step is not None:
        optimizer_step = int(resumed_optimizer_step)
    lr_factor = _apply_lr_schedule(optimizer, cfg["train"], optimizer_step, total_optimizer_steps)

    logger.info(
        "schedule | epochs=%d | log_every_steps=%d | eval_every=%d | save_every=%d | "
        "grad_accum_steps=%d | optimizer_steps=%d | lr_factor=%.6f | lrs=%s",
        epochs,
        log_every_steps,
        eval_every,
        save_every,
        grad_accum_steps,
        total_optimizer_steps,
        lr_factor,
        _format_group_lrs(optimizer),
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_total = 0.0
        running_primary = 0.0
        running_aux = 0.0
        running_boundary = 0.0
        saw_aux = False
        saw_boundary = False
        remainder_steps = len(train_loader) % grad_accum_steps

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            accum_denom = (
                remainder_steps
                if remainder_steps and step > len(train_loader) - remainder_steps
                else grad_accum_steps
            )

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                if s5_debug:
                    loss, parts = _compute_s5_debug_loss(cfg, outputs, masks, criterion)
                else:
                    loss, parts = _compute_total_loss(cfg, outputs, masks, criterion)

            scaler.scale(loss / accum_denom).backward()
            refiner_grad_norm = None
            refiner_grad_tensors = None
            if s5_debug:
                grad_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                refiner_grad_norm, refiner_grad_tensors = _refiner_grad_stats(model, grad_scale=grad_scale)
            should_step = step % grad_accum_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer_step += 1
                lr_factor = _apply_lr_schedule(optimizer, cfg["train"], optimizer_step, total_optimizer_steps)
                optimizer.zero_grad(set_to_none=True)

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
                    f"optimizer_step={optimizer_step} "
                    f"lr_factor={lr_factor:.6f} "
                    f"loss={total_value:.4f} "
                    f"primary={primary_value:.4f}"
                )
                if aux_value is not None:
                    msg += f" aux={aux_value:.4f}"
                if boundary_value is not None:
                    msg += f" boundary={boundary_value:.4f}"
                if s5_debug:
                    for index in range(1, 32):
                        key = f"refine_aux_{index}"
                        delta_key = f"refine_delta_abs_{index}"
                        if key in parts:
                            msg += f" loss_refine_aux_{index}={float(parts[key].item()):.4f}"
                        if delta_key in parts:
                            msg += f" refine_delta_abs_{index}={float(parts[delta_key].item()):.6f}"
                        if key not in parts and delta_key not in parts:
                            break
                    if "final_vs_coarse_delta_abs" in parts:
                        msg += f" final_vs_coarse_delta_abs={float(parts['final_vs_coarse_delta_abs'].item()):.6f}"
                    msg += f" refiner_grad_norm={refiner_grad_norm or 0.0:.6f}"
                    msg += f" refiner_grad_tensors={refiner_grad_tensors or 0}"
                logger.info(msg)

                wandb_payload = {
                    "train/step_loss": total_value,
                    "train/step_primary_loss": primary_value,
                    "train/optimizer_step": optimizer_step,
                    "train/lr_factor": lr_factor,
                    "epoch": epoch + 1,
                }
                for group in optimizer.param_groups:
                    if "name" in group:
                        wandb_payload[f"train/lr/{group['name']}"] = group["lr"]
                if aux_value is not None:
                    wandb_payload["train/step_aux_loss"] = aux_value
                if boundary_value is not None:
                    wandb_payload["train/step_boundary_loss"] = boundary_value
                if s5_debug:
                    for index in range(1, 32):
                        key = f"refine_aux_{index}"
                        delta_key = f"refine_delta_abs_{index}"
                        if key in parts:
                            wandb_payload[f"train/loss_refine_aux_{index}"] = float(parts[key].item())
                        if delta_key in parts:
                            wandb_payload[f"train/refine_delta_abs_{index}"] = float(parts[delta_key].item())
                        if key not in parts and delta_key not in parts:
                            break
                    if "final_vs_coarse_delta_abs" in parts:
                        wandb_payload["train/final_vs_coarse_delta_abs"] = float(parts["final_vs_coarse_delta_abs"].item())
                    wandb_payload["train/refiner_grad_norm"] = refiner_grad_norm or 0.0
                    wandb_payload["train/refiner_grad_tensors"] = refiner_grad_tensors or 0
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
        debug_metrics = None
        if do_eval:
            if s5_debug:
                metric_builder = lambda: build_metric(cfg, val_dataset)
                val_metrics, debug_metrics = _evaluate_s5_debug(model, val_loader, metric_builder, device, cfg)
                logger.info(
                    "S5 debug eval | coarse_mIoU=%.4f | %s | final_mIoU=%.4f",
                    debug_metrics["coarse_mIoU"],
                    " | ".join(
                        f"aux_{index}_mIoU={debug_metrics[f'aux_{index}_mIoU']:.4f}"
                        for index in range(1, len([key for key in debug_metrics if key.startswith("aux_")]) + 1)
                    ),
                    debug_metrics["final_mIoU"],
                )
                prev_miou = debug_metrics["coarse_mIoU"]
                diff_parts = [f"final - coarse={debug_metrics['final_mIoU'] - debug_metrics['coarse_mIoU']:.4f}"]
                aux_count = len([key for key in debug_metrics if key.startswith("aux_")])
                for index in range(1, aux_count + 1):
                    aux_miou = debug_metrics[f"aux_{index}_mIoU"]
                    if index == 1:
                        diff_parts.append(f"aux_1 - coarse={aux_miou - debug_metrics['coarse_mIoU']:.4f}")
                    else:
                        diff_parts.append(f"aux_{index} - aux_{index - 1}={aux_miou - prev_miou:.4f}")
                    prev_miou = aux_miou
                logger.info("S5 debug eval deltas | %s", " | ".join(diff_parts))
            else:
                metric = build_metric(cfg, val_dataset)
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    metric,
                    device,
                    cfg=cfg,
                )
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
            if "valid_pixels" in val_metrics:
                epoch_payload["val/valid_pixels"] = val_metrics["valid_pixels"]
            if "correct_pixels" in val_metrics:
                epoch_payload["val/correct_pixels"] = val_metrics["correct_pixels"]
            if "rare_miou" in val_metrics:
                epoch_payload["val/rare_miou"] = val_metrics["rare_miou"]
        if debug_metrics is not None:
            for key, value in debug_metrics.items():
                epoch_payload[f"val/{key}"] = value

        _log_wandb(wandb_module, global_step, epoch_payload)

        extra = {
            "config": cfg,
            "train_loss": train_loss,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
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
