from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


def _to_tuple(size):
    if isinstance(size, int):
        return (size, size)
    return tuple(size)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Resize:
    def __init__(self, size):
        self.size = _to_tuple(size)

    def __call__(self, sample):
        image = sample["image"].resize(self.size[::-1], Image.BILINEAR)
        mask = sample["mask"].resize(self.size[::-1], Image.NEAREST)
        sample["image"] = image
        sample["mask"] = mask
        return sample


class RandomResize:
    def __init__(self, scale, ratio_range=(0.5, 2.0), keep_ratio=True):
        self.scale = _to_tuple(scale)
        self.ratio_range = tuple(ratio_range)
        self.keep_ratio = keep_ratio

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        base_w, base_h = self.scale
        ratio = random.uniform(*self.ratio_range)
        if self.keep_ratio:
            target_size = (max(1, int(base_w * ratio)), max(1, int(base_h * ratio)))
        else:
            target_size = (
                max(1, int(base_w * random.uniform(*self.ratio_range))),
                max(1, int(base_h * random.uniform(*self.ratio_range))),
            )
        sample["image"] = image.resize(target_size, Image.BILINEAR)
        sample["mask"] = mask.resize(target_size, Image.NEAREST)
        return sample


class RandomCrop:
    def __init__(self, size):
        self.size = _to_tuple(size)

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        th, tw = self.size
        w, h = image.size
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=255)
            w, h = image.size
        if w == tw and h == th:
            sample["image"] = image
            sample["mask"] = mask
            return sample
        left = random.randint(0, w - tw)
        top = random.randint(0, h - th)
        sample["image"] = image.crop((left, top, left + tw, top + th))
        sample["mask"] = mask.crop((left, top, left + tw, top + th))
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image"] = sample["image"].transpose(Image.FLIP_LEFT_RIGHT)
            sample["mask"] = sample["mask"].transpose(Image.FLIP_LEFT_RIGHT)
        return sample


class PhotoMetricDistortion:
    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_range = tuple(contrast_range)
        self.saturation_range = tuple(saturation_range)
        self.hue_delta = hue_delta

    def __call__(self, sample):
        image = sample["image"]
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            image = ImageEnhance.Brightness(image).enhance(1.0 + delta / 255.0)
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*self.contrast_range))
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(*self.saturation_range))
        if random.random() < 0.5:
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            hue_delta = int(random.uniform(-self.hue_delta, self.hue_delta))
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_delta) % 180).astype(np.uint8)
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        sample["image"] = image
        return sample


class ToTensor:
    def __call__(self, sample):
        image = np.array(sample["image"], dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask = np.array(sample["mask"], dtype=np.int64)
        sample["image"] = image
        sample["mask"] = torch.from_numpy(mask).long()
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, sample):
        image = sample["image"]
        if not torch.is_tensor(image):
            raise TypeError("Normalize expects image tensor. Place ToTensor before Normalize.")
        sample["image"] = (image - self.mean) / self.std
        return sample


def build_segmentation_transforms(split, crop_size, mean, std, dataset_name=None, random_resize=None):
    split = split.lower()
    dataset_name = (dataset_name or "").lower()
    transforms = []
    if dataset_name == "cityscapes" and split == "train":
        resize_cfg = random_resize or {}
        transforms.extend(
            [
                RandomResize(
                    scale=resize_cfg.get("scale", (2048, 1024)),
                    ratio_range=resize_cfg.get("ratio_range", (0.5, 2.0)),
                    keep_ratio=resize_cfg.get("keep_ratio", True),
                ),
                RandomCrop(crop_size),
                RandomHorizontalFlip(p=0.5),
                PhotoMetricDistortion(),
            ]
        )
    elif dataset_name == "camvid":
        transforms.append(Resize(crop_size))
        if split == "train":
            transforms.append(RandomHorizontalFlip(p=0.5))
    elif split == "train":
        transforms.extend(
            [
                RandomCrop(crop_size),
                RandomHorizontalFlip(p=0.5),
            ]
        )
    transforms.extend([ToTensor(), Normalize(mean, std)])
    return Compose(transforms)
