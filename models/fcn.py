from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50


class FCNResNet50Segmentation(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if not pretrained:
            raise ValueError("FCN-ResNet50 must use pretrained ResNet50 backbone weights.")
        self.num_classes = num_classes
        self.model = fcn_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            aux_loss=False,
        )

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        logits = self.model(x)["out"]
        return {"logits": logits, "aux": {}}
