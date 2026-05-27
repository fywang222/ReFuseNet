from __future__ import annotations

import numpy as np


CAMVID_COLOR_MAP = np.array(
    [
        [128, 128, 128],
        [128, 0, 0],
        [192, 192, 128],
        [128, 64, 128],
        [60, 40, 222],
        [128, 128, 0],
        [192, 128, 128],
        [64, 64, 128],
        [64, 0, 128],
        [64, 64, 0],
        [0, 128, 192],
    ],
    dtype=np.uint8,
)


CITYSCAPES_COLOR_MAP = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def id_mask_to_color(mask, color_map):
    if hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(color_map))
    color[valid] = color_map[mask[valid].astype(np.int64)]
    return color

