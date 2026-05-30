from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def _segformer_config(variant, num_classes, id2label, label2id):
    if variant == "b1":
        depths = [2, 2, 2, 2]
    elif variant == "b5":
        depths = [3, 6, 40, 3]
    else:
        raise ValueError(f"Unsupported SegFormer variant: {variant}")
    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=depths,
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=256,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        mlp_ratios=[4, 4, 4, 4],
        num_attention_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )


class SegFormerSegmentation(nn.Module):
    def __init__(self, num_classes, pretrained=True, checkpoint="nvidia/mit-b5", variant="b5"):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        id2label = {i: str(i) for i in range(num_classes)}
        label2id = {str(i): i for i in range(num_classes)}
        if pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    checkpoint,
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                )
            except Exception as exc:
                warnings.warn(f"Falling back to random init for SegFormer-{variant.upper()}: {exc}")
                try:
                    config = SegformerConfig.from_pretrained(checkpoint)
                    config.num_labels = num_classes
                    config.id2label = id2label
                    config.label2id = label2id
                except Exception:
                    config = _segformer_config(variant, num_classes, id2label, label2id)
                self.model = SegformerForSemanticSegmentation(config)
        else:
            config = _segformer_config(variant, num_classes, id2label, label2id)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        size = x.shape[-2:]
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return {"logits": logits, "aux": {}}


class SegFormerB1Segmentation(SegFormerSegmentation):
    def __init__(self, num_classes, pretrained=True, checkpoint="nvidia/mit-b1"):
        super().__init__(num_classes=num_classes, pretrained=pretrained, checkpoint=checkpoint, variant="b1")


class SegFormerB5Segmentation(SegFormerSegmentation):
    def __init__(self, num_classes, pretrained=True, checkpoint="nvidia/mit-b5"):
        super().__init__(num_classes=num_classes, pretrained=pretrained, checkpoint=checkpoint, variant="b5")
