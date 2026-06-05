from __future__ import annotations

import os
from pathlib import Path
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


def _format_top_pixels(metrics: Dict, class_names: List[str], key: str, total: int):
    ranked = sorted(enumerate(metrics[key]), key=lambda item: item[1], reverse=True)[:5]
    return ", ".join(f"{class_names[index]}:{count / total:.3f}" for index, count in ranked)


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
        parts.append(f"gt_top={_format_top_pixels(metrics, class_names, 'gt_pixels', total_gt)}")
        parts.append(f"pred_top={_format_top_pixels(metrics, class_names, 'pred_pixels', total_pred)}")
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

    h_grids = max(height - crop_h + stride_h - 1, 0) // stride_h + 1
    w_grids = max(width - crop_w + stride_w - 1, 0) // stride_w + 1
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            bottom = min(h_idx * stride_h + crop_h, height)
            right = min(w_idx * stride_w + crop_w, width)
            top = max(bottom - crop_h, 0)
            left = max(right - crop_w, 0)

            crop = images[:, :, top:bottom, left:right]
            crop_logits = _forward_logits(model, crop)
            region_size = (bottom - top, right - left)
            if crop_logits.shape[-2:] != region_size:
                crop_logits = F.interpolate(crop_logits, size=region_size, mode="bilinear", align_corners=False)

            if num_classes is None:
                num_classes = crop_logits.shape[1]
                logits_sum = images.new_zeros((batch, num_classes, height, width))

            logits_sum += F.pad(
                crop_logits,
                (left, width - right, top, height - bottom),
            )
            count[:, :, top:bottom, left:right] += 1

    if (count == 0).any():
        raise RuntimeError("Sliding-window inference produced uncovered pixels.")
    return logits_sum / count.clamp_min(1)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    metric,
    device,
    cfg=None,
    save_dir=None,
    color_map=None,
):
    model.eval()
    metric.reset()
    save_dir = Path(save_dir) if save_dir is not None else None
    eval_cfg = (cfg or {}).get("eval", {})
    inference = eval_cfg.get("inference", "sliding")
    crop_size = tuple(eval_cfg.get("sliding_crop_size", (1024, 1024)))
    stride = tuple(eval_cfg.get("sliding_stride", (768, 768)))
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
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
