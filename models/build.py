from __future__ import annotations

from .fcn import FCNResNet50Segmentation
from .refusenet import RefuseNet
from .sam_decoder import SAMViTSegmentationBaseline
from .segformer import SegFormerB1Segmentation, SegFormerB5Segmentation


def build_model(cfg):
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()
    num_classes = int(
        model_cfg.get(
            "num_classes",
            model_cfg.get("decoder", {}).get("num_classes", cfg.get("dataset", {}).get("num_classes")),
        )
    )
    pretrained = bool(model_cfg.get("pretrained", True))

    if name == "fcn_resnet50":
        return FCNResNet50Segmentation(num_classes=num_classes, pretrained=pretrained)
    if name == "segformer_b1":
        checkpoint = model_cfg.get("checkpoint", "nvidia/mit-b1")
        return SegFormerB1Segmentation(num_classes=num_classes, pretrained=pretrained, checkpoint=checkpoint)
    if name == "segformer_b5":
        checkpoint = model_cfg.get("checkpoint", "nvidia/mit-b5")
        return SegFormerB5Segmentation(num_classes=num_classes, pretrained=pretrained, checkpoint=checkpoint)
    if name == "sam_vitl_decoder":
        checkpoint = model_cfg.get("checkpoint")
        freeze_encoder = bool(model_cfg.get("freeze_encoder", True))
        return SAMViTSegmentationBaseline(
            num_classes=num_classes,
            checkpoint_path=checkpoint,
            freeze_encoder=freeze_encoder,
        )
    if name == "refusenet":
        model_cfg = dict(model_cfg)
        if "debug" not in model_cfg and "debug" in cfg:
            model_cfg["debug"] = cfg["debug"]
        decoder_cfg = dict(model_cfg.get("decoder", {}))
        decoder_cfg.setdefault("num_classes", num_classes)
        model_cfg["decoder"] = decoder_cfg
        return RefuseNet(model_cfg=model_cfg, train_cfg=cfg.get("train", {}))
    raise ValueError(f"Unknown model name: {name}")
