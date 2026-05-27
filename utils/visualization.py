from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from datasets.color_maps import CAMVID_COLOR_MAP, CITYSCAPES_COLOR_MAP, id_mask_to_color


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_color_map(num_classes):
    if num_classes == 11:
        return CAMVID_COLOR_MAP
    if num_classes == 19:
        return CITYSCAPES_COLOR_MAP
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)


def colorize_mask(mask, color_map=None):
    if color_map is None:
        color_map = build_color_map(int(mask.max()) + 1 if np.asarray(mask).size else 1)
    return id_mask_to_color(mask, color_map)


def tensor_to_uint8_image(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if torch.is_tensor(image):
        image = image.detach().cpu().float()
        if image.ndim == 3:
            image = image.permute(1, 2, 0).numpy()
        mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        image = (image * std + mean).clip(0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
        return image
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)
    return image


def _error_map(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != 255
    mismatch = valid & (gt != pred)
    error[mismatch] = np.array([255, 0, 0], dtype=np.uint8)
    error[valid & (gt == pred)] = np.array([0, 0, 0], dtype=np.uint8)
    error[~valid] = np.array([80, 80, 80], dtype=np.uint8)
    return error


def save_segmentation_visualization(
    image,
    gt_mask,
    pred_mask,
    out_dir,
    name,
    color_map=None,
    overlay=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_uint8 = tensor_to_uint8_image(image)
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.detach().cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()

    gt_color = colorize_mask(gt_mask, color_map)
    pred_color = colorize_mask(pred_mask, color_map)
    error = _error_map(gt_mask, pred_mask)

    Image.fromarray(image_uint8).save(out_dir / f"{name}_image.png")
    Image.fromarray(gt_color).save(out_dir / f"{name}_gt.png")
    Image.fromarray(pred_color).save(out_dir / f"{name}_pred.png")
    Image.fromarray(error).save(out_dir / f"{name}_error.png")

    if overlay:
        overlay_img = (0.65 * image_uint8.astype(np.float32) + 0.35 * pred_color.astype(np.float32)).clip(0, 255)
        Image.fromarray(overlay_img.astype(np.uint8)).save(out_dir / f"{name}_overlay.png")

