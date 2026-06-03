from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerSegmentation(nn.Module):
    def __init__(self, num_classes, pretrained=True, checkpoint="nvidia/mit-b5", variant="b5"):
        super().__init__()
        if not pretrained:
            raise ValueError(f"SegFormer-{variant.upper()} must use pretrained checkpoint weights.")
        self.num_classes = num_classes
        self.variant = variant
        id2label = {i: str(i) for i in range(num_classes)}
        label2id = {str(i): i for i in range(num_classes)}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        size = x.shape[-2:]
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return {"logits": logits, "aux": {}}


class SegFormerB5Segmentation(SegFormerSegmentation):
    def __init__(self, num_classes, pretrained=True, checkpoint="nvidia/mit-b5"):
        super().__init__(num_classes=num_classes, pretrained=pretrained, checkpoint=checkpoint, variant="b5")
