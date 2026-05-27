from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps


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


def build_segmentation_transforms(split, crop_size, mean, std):
    split = split.lower()
    transforms = []
    if split == "train":
        transforms.extend(
            [
                RandomHorizontalFlip(p=0.5),
                RandomCrop(crop_size),
            ]
        )
    else:
        transforms.append(Resize(crop_size))
    transforms.extend([ToTensor(), Normalize(mean, std)])
    return Compose(transforms)

