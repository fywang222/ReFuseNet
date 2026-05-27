from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50


class FCNResNet50Segmentation(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            self.model = fcn_resnet50(weights=None, weights_backbone=weights_backbone, aux_loss=False)
        except Exception as exc:
            if pretrained:
                warnings.warn(f"Falling back to random init for FCN-ResNet50: {exc}")
            self.model = fcn_resnet50(weights=None, weights_backbone=None, aux_loss=False)

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        logits = self.model(x)["out"]
        return {"logits": logits, "aux": {}}

