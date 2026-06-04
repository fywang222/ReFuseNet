from .camvid import CAMVID_CLASSES, CamVidDataset
from .cityscapes import CITYSCAPES_CLASSES, CityscapesDataset
from .transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    RandomResize,
    PhotoMetricDistortion,
    ToTensor,
    build_segmentation_transforms,
)


def build_dataset(cfg, split="train"):
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"].lower()
    crop_size = tuple(cfg.get("train", {}).get("crop_size", dataset_cfg.get("crop_size", (512, 512))))
    mean = tuple(dataset_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(dataset_cfg.get("std", (0.229, 0.224, 0.225)))

    transform = build_segmentation_transforms(
        split=split,
        crop_size=crop_size,
        mean=mean,
        std=std,
        dataset_name=name,
        random_resize=dataset_cfg.get("random_resize"),
    )

    if name == "camvid":
        dataset = CamVidDataset(
            root=dataset_cfg["root"],
            split=dataset_cfg.get(f"{split}_split", split),
            list_file=dataset_cfg.get(f"{split}_list"),
            transform=transform,
        )
    elif name == "cityscapes":
        dataset = CityscapesDataset(
            root=dataset_cfg["root"],
            split=dataset_cfg.get(f"{split}_split", split),
            list_file=dataset_cfg.get(f"{split}_list"),
            label_format=dataset_cfg.get("label_format", "auto"),
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_cfg['name']}")

    max_samples = dataset_cfg.get(f"{split}_num_samples")
    if max_samples is None and split == "train":
        max_samples = dataset_cfg.get("overfit_num_samples")
    if max_samples:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(min(len(dataset), int(max_samples)))))

    return dataset


def get_class_names(dataset):
    if hasattr(dataset, "class_names"):
        return list(dataset.class_names)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "class_names"):
        return list(dataset.dataset.class_names)
    return None
