from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .color_maps import CAMVID_COLOR_MAP


CAMVID_CLASSES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]


_CAMVID_COLOR_TO_ID = {tuple(color.tolist()): idx for idx, color in enumerate(CAMVID_COLOR_MAP)}


def camvid_rgb_mask_to_ids(mask_rgb):
    if mask_rgb.ndim == 2:
        mask = mask_rgb.astype(np.int64)
        mask[(mask < 0) | (mask >= len(CAMVID_CLASSES))] = 255
        return mask.astype(np.uint8)

    h, w, _ = mask_rgb.shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for color, idx in _CAMVID_COLOR_TO_ID.items():
        matches = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        mask[matches] = idx
    return mask


def camvid_ids_to_rgb(mask_ids):
    from .color_maps import id_mask_to_color

    return id_mask_to_color(mask_ids, CAMVID_COLOR_MAP)


def _candidate_split_files(root: Path, split: str, list_file=None):
    if list_file:
        list_path = Path(list_file)
        if not list_path.is_absolute():
            list_path = root / list_path
        return [list_path]
    names = [f"{split}.txt", f"{split}_list.txt", f"{split}_split.txt"]
    dirs = ["", "splits", "Splits", "data_splits", "lists", "Lists"]
    candidates = []
    for directory in dirs:
        for name in names:
            candidates.append(root / directory / name)
    return candidates


def _resolve_camvid_pair(root: Path, token: str, split: str | None = None):
    token_path = Path(token)
    if token_path.exists():
        image_path = token_path
    else:
        possible = [
            root / token,
            root / "images" / token,
            root / "Images" / token,
            root / "train" / token,
            root / "val" / token,
            root / "test" / token,
        ]
        image_path = next((p for p in possible if p.exists()), None)
        if image_path is None:
            stem = token_path.stem if token_path.suffix else token
            for folder in [root, root / "images", root / "Images", root / "train", root / "val", root / "test"]:
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = folder / f"{stem}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                if image_path is not None:
                    break
    if image_path is None:
        raise FileNotFoundError(f"Could not resolve CamVid image path for token: {token}")

    stem = image_path.stem
    label_names = [
        f"{stem}_L.png",
        f"{stem}_l.png",
        f"{stem}_label.png",
        f"{stem}_labels.png",
        f"{stem}.png",
    ]
    label_candidates = [
        image_path.with_name(name)
        for name in label_names
        if image_path.with_name(name) != image_path
    ]
    label_dirs = [root, root / "labels", root / "Labels", root / "LabeledApproved_full", root / "labels_color"]
    if split is not None:
        label_dirs.extend([root / split / "labels", root / split / "Labels"])
    if image_path.parent.name.lower() in {"images", "image"}:
        label_dirs.extend([image_path.parent.parent / "labels", image_path.parent.parent / "Labels"])
    for folder in label_dirs:
        label_candidates.extend([folder / name for name in label_names])
    label_path = next((p for p in label_candidates if p.exists()), None)
    if label_path is None:
        raise FileNotFoundError(f"Could not resolve CamVid label path for token: {token}")
    return image_path, label_path


class CamVidDataset(Dataset):
    class_names = CAMVID_CLASSES

    def __init__(self, root, split="train", transform=None, list_file=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.list_file = list_file
        self.samples = self._build_index()

    def _build_index(self):
        split_file = next((p for p in _candidate_split_files(self.root, self.split, self.list_file) if p.exists()), None)
        samples = []
        if split_file is not None:
            for line in split_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    image_path = Path(parts[0])
                    label_path = Path(parts[1])
                    if not image_path.is_absolute():
                        image_path = self.root / image_path
                    if not label_path.is_absolute():
                        label_path = self.root / label_path
                    if not image_path.exists():
                        raise FileNotFoundError(f"CamVid image not found: {image_path}")
                    if not label_path.exists():
                        raise FileNotFoundError(f"CamVid label not found: {label_path}")
                    samples.append((image_path, label_path))
                else:
                    image_path, label_path = _resolve_camvid_pair(self.root, parts[0], split=self.split)
                    samples.append((image_path, label_path))
            if samples:
                return samples

        folders = []
        split_image_dir = self.root / self.split / "images"
        split_image_dir_alt = self.root / self.split / "Images"
        if split_image_dir.exists() or split_image_dir_alt.exists():
            folders.extend([split_image_dir, split_image_dir_alt])
        else:
            folders.extend([self.root / "images", self.root / "Images", self.root / self.split])
        for folder in folders:
            if folder.exists():
                for image_path in sorted(folder.rglob("*")):
                    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    label_path = _resolve_camvid_pair(self.root, str(image_path), split=self.split)[1]
                    samples.append((image_path, label_path))
        if not samples:
            for image_path in sorted(self.root.rglob("*")):
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if "_L" in image_path.stem.lower() or "label" in image_path.stem.lower():
                    continue
                label_path = _resolve_camvid_pair(self.root, str(image_path), split=self.split)[1]
                samples.append((image_path, label_path))
        if not samples:
            raise FileNotFoundError(f"No CamVid samples found under {self.root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        if mask_path is None or not mask_path.exists():
            raise FileNotFoundError(f"CamVid label not found for image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size[::-1]
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)
        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 4:
            mask_arr = mask_arr[..., :3]
        if mask_arr.ndim == 3:
            mask_arr = camvid_rgb_mask_to_ids(mask_arr)
        else:
            mask_arr = camvid_rgb_mask_to_ids(mask_arr)
        mask = Image.fromarray(mask_arr.astype(np.uint8), mode="L")

        sample = {
            "image": image,
            "mask": mask,
            "name": image_path.stem,
            "orig_size": orig_size,
        }
        if self.transform is not None:
            sample = self.transform(sample)
            sample["name"] = image_path.stem
            sample["orig_size"] = orig_size
        else:
            image_arr = np.array(image, dtype=np.float32) / 255.0
            sample["image"] = torch.from_numpy(image_arr).permute(2, 0, 1).contiguous()
            sample["mask"] = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
        return sample
