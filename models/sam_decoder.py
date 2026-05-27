from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_encoder import SAMViTLEncoder


class SimpleSegDecoder(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256, num_classes=11):
        super().__init__()
        groups = 32 if hidden_channels % 32 == 0 else 16
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, features, size):
        logits = self.block(features)
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits


class SAMViTSegmentationBaseline(nn.Module):
    def __init__(self, num_classes, checkpoint_path, freeze_encoder=True):
        super().__init__()
        self.encoder = SAMViTLEncoder(checkpoint_path=checkpoint_path, freeze=freeze_encoder)
        self.decoder = SimpleSegDecoder(in_channels=256, hidden_channels=256, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features, size=size)
        return {"logits": logits, "aux": {}, "features": features}
