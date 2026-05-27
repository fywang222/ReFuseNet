from __future__ import annotations

from .fcn import FCNResNet50Segmentation
from .sam_decoder import SAMViTSegmentationBaseline
from .segformer import SegFormerB1Segmentation


def build_model(cfg):
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()
    num_classes = int(model_cfg["num_classes"])
    pretrained = bool(model_cfg.get("pretrained", True))

    if name == "fcn_resnet50":
        return FCNResNet50Segmentation(num_classes=num_classes, pretrained=pretrained)
    if name == "segformer_b1":
        checkpoint = model_cfg.get("checkpoint", "nvidia/mit-b1")
        return SegFormerB1Segmentation(num_classes=num_classes, pretrained=pretrained, checkpoint=checkpoint)
    if name == "sam_vitl_decoder":
        checkpoint = model_cfg.get("checkpoint")
        freeze_encoder = bool(model_cfg.get("freeze_encoder", True))
        return SAMViTSegmentationBaseline(
            num_classes=num_classes,
            checkpoint_path=checkpoint,
            freeze_encoder=freeze_encoder,
        )
    raise ValueError(f"Unknown model name: {name}")

