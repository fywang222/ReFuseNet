from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


_CITYSCAPES_LUT = np.full(256, 255, dtype=np.uint8)
for label_id, train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    _CITYSCAPES_LUT[label_id] = train_id


def cityscapes_label_ids_to_train_ids(label_ids):
    label_ids = np.asarray(label_ids, dtype=np.int16)
    train_ids = np.full(label_ids.shape, 255, dtype=np.uint8)
    valid = (label_ids >= 0) & (label_ids < len(_CITYSCAPES_LUT))
    train_ids[valid] = _CITYSCAPES_LUT[label_ids[valid].astype(np.int64)]
    return train_ids


def cityscapes_ids_to_rgb(mask_ids):
    from .color_maps import CITYSCAPES_COLOR_MAP, id_mask_to_color

    return id_mask_to_color(mask_ids, CITYSCAPES_COLOR_MAP)


class CityscapesDataset(Dataset):
    class_names = CITYSCAPES_CLASSES

    def __init__(self, root, split="train", transform=None, list_file=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.list_file = list_file
        self.samples = self._build_index()

    def _resolve_from_list(self, line):
        parts = line.split()
        image_path = Path(parts[0])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if len(parts) >= 2:
            mask_path = Path(parts[1])
            if not mask_path.is_absolute():
                mask_path = self.root / mask_path
            return image_path, mask_path
        city = image_path.parent.name
        base = image_path.name.replace("_leftImg8bit.png", "")
        mask_path = self.root / "gtFine" / self.split / city / f"{base}_gtFine_labelIds.png"
        return image_path, mask_path if mask_path.exists() else None

    def _build_index(self):
        if self.list_file:
            list_path = Path(self.list_file)
            if not list_path.is_absolute():
                list_path = self.root / list_path
            if not list_path.exists():
                raise FileNotFoundError(f"Cityscapes list file not found: {list_path}")
            samples = []
            for line in list_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                samples.append(self._resolve_from_list(line))
            if not samples:
                raise FileNotFoundError(f"No Cityscapes samples found in list file: {list_path}")
            return samples

        img_root = self.root / "leftImg8bit" / self.split
        if not img_root.exists():
            img_root = self.root / self.split / "leftImg8bit"
        if not img_root.exists():
            img_root = self.root / "leftImg8bit"
        if not img_root.exists():
            raise FileNotFoundError(f"Could not find Cityscapes images under {self.root}")

        samples = []
        for image_path in sorted(img_root.rglob("*_leftImg8bit.png")):
            city = image_path.parent.name
            base = image_path.name.replace("_leftImg8bit.png", "")
            mask_path = self.root / "gtFine" / self.split / city / f"{base}_gtFine_labelIds.png"
            if not mask_path.exists():
                mask_path = None
            samples.append((image_path, mask_path))
        if not samples:
            raise FileNotFoundError(f"No Cityscapes samples found under {img_root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size[::-1]
        if mask_path is None:
            mask = Image.fromarray(np.full(orig_size, 255, dtype=np.uint8), mode="L")
        else:
            label_ids = np.array(Image.open(mask_path))
            mask_ids = cityscapes_label_ids_to_train_ids(label_ids)
            mask = Image.fromarray(mask_ids.astype(np.uint8), mode="L")

        sample = {
            "image": image,
            "mask": mask,
            "name": image_path.stem.replace("_leftImg8bit", ""),
            "orig_size": orig_size,
        }
        if self.transform is not None:
            sample = self.transform(sample)
            sample["name"] = image_path.stem.replace("_leftImg8bit", "")
            sample["orig_size"] = orig_size
        else:
            image_arr = np.array(image, dtype=np.float32) / 255.0
            sample["image"] = torch.from_numpy(image_arr).permute(2, 0, 1).contiguous()
            sample["mask"] = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
        return sample
