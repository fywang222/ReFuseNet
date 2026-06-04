from __future__ import annotations

import os
from pathlib import Path
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from datasets import build_dataset, get_class_names
from datasets.color_maps import CAMVID_COLOR_MAP, CITYSCAPES_COLOR_MAP
from utils.metrics import SegMetric
from utils.visualization import save_segmentation_visualization


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return _expand_env_vars(yaml.safe_load(f))


def _expand_env_vars(value):
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def collate_segmentation_batch(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    names = [item["name"] for item in batch]
    orig_sizes = [tuple(item["orig_size"]) for item in batch]
    return {"image": images, "mask": masks, "name": names, "orig_size": orig_sizes}


def build_dataloader(cfg, split, shuffle=None):
    dataset = build_dataset(cfg, split=split)
    train = split == "train"
    if shuffle is None:
        shuffle = train
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"] if train else cfg["eval"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_segmentation_batch,
    )
    return dataset, loader


def build_metric(cfg, dataset):
    dataset_cfg = cfg["dataset"]
    class_names = get_class_names(dataset)
    rare_class_names = None
    if dataset_cfg["name"].lower() == "camvid":
        rare_class_names = ["Pole", "SignSymbol", "Pedestrian", "Bicyclist"]
    return SegMetric(
        num_classes=dataset_cfg["num_classes"],
        ignore_index=dataset_cfg.get("ignore_index", 255),
        class_names=class_names,
        rare_class_names=rare_class_names,
    )


def get_color_map(cfg):
    name = cfg["dataset"]["name"].lower()
    if name == "camvid":
        return CAMVID_COLOR_MAP
    if name == "cityscapes":
        return CITYSCAPES_COLOR_MAP
    return None


def format_metrics(metrics: Dict, class_names: Optional[List[str]] = None):
    parts = [
        f"mIoU={metrics['miou']:.4f}",
        f"pixel_acc={metrics['pixel_acc']:.4f}",
        f"mean_acc={metrics['mean_acc']:.4f}",
    ]
    if "valid_pixels" in metrics and "correct_pixels" in metrics:
        parts.append(f"valid_pixels={metrics['valid_pixels']}")
        parts.append(f"correct_pixels={metrics['correct_pixels']}")
    if "rare_miou" in metrics:
        parts.append(f"rare_mIoU={metrics['rare_miou']:.4f}")
    if class_names and "ious" in metrics:
        per_class = ", ".join(
            f"{name}:{iou:.3f}" for name, iou in zip(class_names, metrics["ious"])
        )
        parts.append(per_class)
    if class_names and "gt_pixels" in metrics and "pred_pixels" in metrics:
        total_gt = max(int(sum(metrics["gt_pixels"])), 1)
        total_pred = max(int(sum(metrics["pred_pixels"])), 1)

        def _top_pixels(key, total):
            ranked = sorted(enumerate(metrics[key]), key=lambda item: item[1], reverse=True)[:5]
            return ", ".join(
                f"{class_names[index]}:{count / total:.3f}"
                for index, count in ranked
            )

        parts.append(f"gt_top={_top_pixels('gt_pixels', total_gt)}")
        parts.append(f"pred_top={_top_pixels('pred_pixels', total_pred)}")
    return " | ".join(parts)


@torch.no_grad()
def _forward_logits(model, images):
    outputs = model(images)
    if not isinstance(outputs, dict) or "logits" not in outputs:
        raise TypeError("Model forward must return a dict with a 'logits' tensor.")
    return outputs["logits"]


@torch.no_grad()
def sliding_window_logits(model, images, crop_size=(1024, 1024), stride=(768, 768)):
    crop_h, crop_w = crop_size
    stride_h, stride_w = stride
    batch, _, height, width = images.shape

    num_classes = None
    logits_sum = None
    count = images.new_zeros((batch, 1, height, width))

    h_starts = list(range(0, max(height - crop_h, 0) + 1, stride_h))
    w_starts = list(range(0, max(width - crop_w, 0) + 1, stride_w))

    if not h_starts or h_starts[-1] != max(height - crop_h, 0):
        h_starts.append(max(height - crop_h, 0))
    if not w_starts or w_starts[-1] != max(width - crop_w, 0):
        w_starts.append(max(width - crop_w, 0))

    for top in h_starts:
        for left in w_starts:
            bottom = min(top + crop_h, height)
            right = min(left + crop_w, width)

            crop = images[:, :, top:bottom, left:right]
            pad_h = crop_h - crop.shape[-2]
            pad_w = crop_w - crop.shape[-1]

            if pad_h > 0 or pad_w > 0:
                crop = F.pad(crop, (0, pad_w, 0, pad_h))

            crop_logits = _forward_logits(model, crop)
            crop_logits = crop_logits[:, :, : bottom - top, : right - left]

            if num_classes is None:
                num_classes = crop_logits.shape[1]
                logits_sum = images.new_zeros((batch, num_classes, height, width))

            logits_sum[:, :, top:bottom, left:right] += crop_logits
            count[:, :, top:bottom, left:right] += 1

    return logits_sum / count.clamp_min(1)


@torch.no_grad()
def _log_overlap_debug(model, images, masks, metric, crop_size, stride, logger):
    whole_logits = _forward_logits(model, images)
    sliding_logits = sliding_window_logits(model, images, crop_size=crop_size, stride=stride)
    whole_logits = F.interpolate(whole_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    sliding_logits = F.interpolate(sliding_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

    whole_metric = SegMetric(
        num_classes=metric.num_classes,
        ignore_index=metric.ignore_index,
        class_names=metric.class_names,
        rare_class_names=metric.rare_class_names,
    )
    sliding_metric = SegMetric(
        num_classes=metric.num_classes,
        ignore_index=metric.ignore_index,
        class_names=metric.class_names,
        rare_class_names=metric.rare_class_names,
    )
    whole_metric.update(whole_logits, masks)
    sliding_metric.update(sliding_logits, masks)
    whole = whole_metric.compute()
    sliding = sliding_metric.compute()

    whole_pred = whole_logits.argmax(dim=1)
    sliding_pred = sliding_logits.argmax(dim=1)
    valid = masks.ne(metric.ignore_index)
    pred_diff = whole_pred.ne(sliding_pred) & valid
    pred_diff_ratio = float(pred_diff.sum().item() / valid.sum().clamp_min(1).item())
    max_logit_abs_diff = float((whole_logits - sliding_logits).abs().max().item())
    mean_logit_abs_diff = float((whole_logits - sliding_logits).abs().mean().item())

    logger.info(
        "eval overlap debug | whole_mIoU=%.4f | whole_pixel_acc=%.4f | "
        "sliding_mIoU=%.4f | sliding_pixel_acc=%.4f | pred_diff_ratio=%.6f | "
        "max_logit_abs_diff=%.6f | mean_logit_abs_diff=%.6f",
        whole["miou"],
        whole["pixel_acc"],
        sliding["miou"],
        sliding["pixel_acc"],
        pred_diff_ratio,
        max_logit_abs_diff,
        mean_logit_abs_diff,
    )


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    metric,
    device,
    cfg=None,
    save_dir=None,
    color_map=None,
    overlap_debug=False,
    logger=None,
):
    model.eval()
    metric.reset()
    save_dir = Path(save_dir) if save_dir is not None else None
    logger = logger or logging.getLogger(__name__)
    eval_cfg = (cfg or {}).get("eval", {})
    dataset_name = (cfg or {}).get("dataset", {}).get("name", "").lower()
    inference = eval_cfg.get("inference", "sliding" if dataset_name == "cityscapes" else "whole")
    crop_size = tuple(eval_cfg.get("sliding_crop_size", (1024, 1024)))
    stride = tuple(eval_cfg.get("sliding_stride", (768, 768)))
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if overlap_debug and batch_index == 0 and inference == "sliding":
            _log_overlap_debug(model, images, masks, metric, crop_size, stride, logger)
        if inference == "sliding":
            logits = sliding_window_logits(model, images, crop_size=crop_size, stride=stride)
        elif inference == "whole":
            logits = _forward_logits(model, images)
        else:
            raise ValueError(f"Unsupported eval.inference: {inference}")
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        metric.update(logits, masks)

        if save_dir is not None:
            preds = logits.argmax(dim=1).detach().cpu()
            for i, name in enumerate(batch["name"]):
                save_segmentation_visualization(
                    image=images[i].detach().cpu(),
                    gt_mask=masks[i].detach().cpu(),
                    pred_mask=preds[i],
                    out_dir=save_dir,
                    name=name,
                    color_map=color_map,
                    overlay=True,
                )
    return metric.compute()
