from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from datasets import build_dataset, get_class_names
from datasets.color_maps import CAMVID_COLOR_MAP, CITYSCAPES_COLOR_MAP
from losses import build_loss
from models import build_model
from utils.metrics import SegMetric
from utils.visualization import save_segmentation_visualization


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    if "rare_miou" in metrics:
        parts.append(f"rare_mIoU={metrics['rare_miou']:.4f}")
    if class_names and "ious" in metrics:
        per_class = ", ".join(
            f"{name}:{iou:.3f}" for name, iou in zip(class_names, metrics["ious"])
        )
        parts.append(per_class)
    return " | ".join(parts)


@torch.no_grad()
def evaluate_model(model, loader, metric, device, save_dir=None, color_map=None):
    model.eval()
    metric.reset()
    save_dir = Path(save_dir) if save_dir is not None else None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        if not isinstance(outputs, dict) or "logits" not in outputs:
            raise TypeError("Model forward must return a dict with a 'logits' tensor.")
        logits = outputs["logits"]
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
