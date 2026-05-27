from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMViTLEncoder(nn.Module):
    def __init__(self, checkpoint_path: str | None, freeze=True, image_size=1024):
        super().__init__()
        if not checkpoint_path:
            raise FileNotFoundError(
                "SAM ViT-L baseline requires a checkpoint path. "
                "Provide model.checkpoint in the config or --pretrained for a matching checkpoint."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

        try:
            from segment_anything import sam_model_registry
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "segment-anything is required for sam_vitl_decoder. Install the package before using this model."
            ) from exc

        sam = sam_model_registry["vit_l"](checkpoint=str(checkpoint_path))
        self.image_encoder = sam.image_encoder
        self.image_size = int(image_size)
        self.register_buffer("_sam_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_sam_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        features = self.image_encoder(x)
        return features

