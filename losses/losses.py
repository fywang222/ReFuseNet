from __future__ import annotations

import torch
import torch.nn as nn


def build_loss(cfg):
    dataset_cfg = cfg["dataset"]
    ignore_index = dataset_cfg.get("ignore_index", 255)
    weights = dataset_cfg.get("class_weights")
    if weights is not None:
        weight = torch.tensor(weights, dtype=torch.float32)
    else:
        weight = None
    return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
