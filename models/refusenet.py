from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlainUpsampleDecoder(nn.Module):
    """S0/S1 decoder: one final SAM image feature, no pyramid or refinement."""

    def __init__(self, in_channels: int, decoder_dim: int, num_classes: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, decoder_dim, kernel_size=1)
        self.block1 = ConvGNAct(decoder_dim, decoder_dim)
        self.block2 = ConvGNAct(decoder_dim, decoder_dim)
        self.block3 = ConvGNAct(decoder_dim, decoder_dim)
        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(
        self,
        feature: torch.Tensor,
        output_size: tuple[int, int],
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self.project(feature)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        decoder_features = self.block3(x)
        logits = self.head(decoder_features)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        out = {"logits": logits}
        if return_features:
            out["decoder_features"] = decoder_features
        return out


class MultiScaleFusionDecoder(nn.Module):
    """S2/S3 decoder: project same-style feature lists and fuse them at one scale."""

    def __init__(
        self,
        in_channels: list[int],
        decoder_dim: int,
        num_classes: int,
        fuse_type: str = "sum",
    ):
        super().__init__()
        if fuse_type not in {"sum", "concat"}:
            raise ValueError(f"Unsupported fuse_type: {fuse_type}")
        self.fuse_type = fuse_type
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_dim, kernel_size=1) for channels in in_channels]
        )
        fused_channels = decoder_dim if fuse_type == "sum" else decoder_dim * len(in_channels)
        self.fuse = nn.Sequential(
            ConvGNAct(fused_channels, decoder_dim),
            ConvGNAct(decoder_dim, decoder_dim),
        )
        self.refine1 = ConvGNAct(decoder_dim, decoder_dim)
        self.refine2 = ConvGNAct(decoder_dim, decoder_dim)
        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")
        fusion_size = max((feature.shape[-2:] for feature in features), key=lambda size: size[0] * size[1])
        projected = []
        for feature, projection in zip(features, self.projections):
            x = projection(feature)
            if x.shape[-2:] != fusion_size:
                x = F.interpolate(x, size=fusion_size, mode="bilinear", align_corners=False)
            projected.append(x)
        if self.fuse_type == "sum":
            x = torch.stack(projected, dim=0).sum(dim=0)
        else:
            x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        decoder_features = self.refine2(x)
        logits = self.head(decoder_features)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        out = {"logits": logits}
        if return_features:
            out["decoder_features"] = decoder_features
        return out


class DPTResidualConvUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DPTFusionBlock(nn.Module):
    def __init__(self, channels: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        self.residual = DPTResidualConvUnit(channels) if has_residual else None
        self.output = DPTResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.has_residual and residual is not None:
            x = x + self.residual(residual)
        x = self.output(x)
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DualDPTBoundaryDecoder(nn.Module):
    """
    S6 decoder: DA3-style DualDPT with a PIDNet-style one-channel boundary head.

    The semantic and boundary branches have independent top-down DPT fusion blocks.
    Boundary supervision is intentionally left to the trainer/loss code.
    """

    def __init__(
        self,
        in_channels: list[int],
        decoder_dim: int,
        num_classes: int,
        out_channels: list[int] | tuple[int, int, int, int] = (96, 192, 384, 768),
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"DualDPTBoundaryDecoder expects 4 input features, got {len(in_channels)}")
        if len(out_channels) != 4:
            raise ValueError(f"decoder.dualdpt_out_channels must have 4 values, got {len(out_channels)}")

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(in_channels, out_channels)]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.adapters = nn.ModuleList(
            [nn.Conv2d(out_ch, decoder_dim, kernel_size=3, padding=1, bias=False) for out_ch in out_channels]
        )

        self.sem_refine4 = DPTFusionBlock(decoder_dim, has_residual=False)
        self.sem_refine3 = DPTFusionBlock(decoder_dim)
        self.sem_refine2 = DPTFusionBlock(decoder_dim)
        self.sem_refine1 = DPTFusionBlock(decoder_dim)
        self.sem_neck = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(decoder_dim // 2), decoder_dim // 2),
            nn.GELU(),
        )
        self.sem_head = nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=1)

        self.boundary_refine4 = DPTFusionBlock(decoder_dim, has_residual=False)
        self.boundary_refine3 = DPTFusionBlock(decoder_dim)
        self.boundary_refine2 = DPTFusionBlock(decoder_dim)
        self.boundary_refine1 = DPTFusionBlock(decoder_dim)
        self.boundary_neck = nn.Sequential(
            ConvGNAct(decoder_dim, decoder_dim // 2),
            ConvGNAct(decoder_dim // 2, decoder_dim // 2),
        )
        self.boundary_head = nn.Conv2d(decoder_dim // 2, 1, kernel_size=1)

    def _project_and_resize(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        resized = []
        for feature, project, resize in zip(features, self.projects, self.resize_layers):
            resized.append(resize(project(feature)))
        return [adapter(feature) for adapter, feature in zip(self.adapters, resized)]

    def _fuse_semantic(self, features: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = features
        x = self.sem_refine4(l4, size=l3.shape[-2:])
        x = self.sem_refine3(x, l3, size=l2.shape[-2:])
        x = self.sem_refine2(x, l2, size=l1.shape[-2:])
        return self.sem_refine1(x, l1)

    def _fuse_boundary(self, features: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = features
        x = self.boundary_refine4(l4, size=l3.shape[-2:])
        x = self.boundary_refine3(x, l3, size=l2.shape[-2:])
        x = self.boundary_refine2(x, l2, size=l1.shape[-2:])
        return self.boundary_refine1(x, l1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"DualDPTBoundaryDecoder expects 4 features, got {len(features)}")

        features = self._project_and_resize(features)
        semantic_features = self.sem_neck(self._fuse_semantic(features))
        boundary_features = self.boundary_neck(self._fuse_boundary(features))

        logits = self.sem_head(semantic_features)
        boundary_logits = self.boundary_head(boundary_features)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(boundary_logits, size=output_size, mode="bilinear", align_corners=False)

        out = {"logits": logits, "boundary_logits": boundary_logits}
        if return_features:
            out["decoder_features"] = semantic_features
            out["boundary_features"] = boundary_features
        return out


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        update, reset = self.gates(combined).chunk(2, dim=1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate


class GRURefiner(nn.Module):
    """S4 refinement: RAFT-like iterative hidden-state updates on coarse logits."""

    def __init__(self, num_classes: int, context_channels: int, hidden_channels: int, iters: int = 3):
        super().__init__()
        self.iters = int(iters)
        self.logits_to_hidden = nn.Conv2d(num_classes, hidden_channels, kernel_size=3, padding=1)
        self.context_proj = nn.Conv2d(context_channels, hidden_channels, kernel_size=1)
        self.cell = ConvGRUCell(hidden_channels, hidden_channels)
        self.delta_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, coarse_logits: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-2:] != coarse_logits.shape[-2:]:
            context = F.interpolate(context, size=coarse_logits.shape[-2:], mode="bilinear", align_corners=False)
        h = self.logits_to_hidden(coarse_logits)
        x = self.context_proj(context)
        logits = coarse_logits
        for _ in range(self.iters):
            h = self.cell(x, h)
            logits = logits + self.delta_head(h)
        return logits


class RefuseNet(nn.Module):
    """
    Single configurable ReFuseNet/RefuseNet ablation class.

    S0: frozen SAM image encoder + final feature + plain decoder.
    S1: low-LR SAM fine-tune + final feature + plain decoder.
    S2: low-LR SAM fine-tune + pseudo 4-scale features from the final feature.
    S3: low-LR SAM fine-tune + true intermediate SAM ViT block features.
    S4: S3 plus iterative GRU refinement and optional coarse-logit auxiliary loss.
    S6: DA3-style DualDPT decoder plus a PIDNet-style boundary head.
    """

    PRESETS: dict[str, dict[str, Any]] = {
        "S0": {
            "sam": {"train_mode": "frozen"},
            "decoder": {"feature_mode": "final", "fusion_mode": "single"},
            "refine": {"enabled": False},
        },
        "S1": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "final", "fusion_mode": "single"},
            "refine": {"enabled": False},
        },
        "S2": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "pseudo_pyramid", "fusion_mode": "multiscale"},
            "refine": {"enabled": False},
        },
        "S3": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "multi_level", "fusion_mode": "multiscale"},
            "refine": {"enabled": False},
        },
        "S4": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "multi_level", "fusion_mode": "multiscale"},
            "refine": {"enabled": True, "type": "gru"},
        },
        "S6": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "dualdpt", "fusion_mode": "dualdpt"},
            "refine": {"enabled": False},
            "boundary": {"enabled": True},
        },
    }

    DEFAULTS: dict[str, Any] = {
        "setting": "S0",
        "debug": False,
        "sam": {
            "model_type": "vit_b",
            "checkpoint": None,
            "train_mode": "frozen",
            "image_size": 1024,
            "intermediate_blocks": [3, 6, 9, 12],
        },
        "decoder": {
            "num_classes": None,
            "dim": 128,
            "feature_mode": "final",
            "fusion_mode": "single",
            "pseudo_scales": [4, 8, 16, 32],
            "fuse_type": "sum",
            "dualdpt_out_channels": [96, 192, 384, 768],
        },
        "refine": {
            "enabled": False,
            "type": "gru",
            "iters": 3,
        },
        "boundary": {
            "enabled": False,
        },
    }

    def __init__(self, model_cfg: dict[str, Any], train_cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = self.resolve_config(model_cfg)
        self.train_cfg = train_cfg or {}
        self.debug = bool(self.cfg.get("debug", False))
        self.sam_cfg = self.cfg["sam"]
        self.decoder_cfg = self.cfg["decoder"]
        self.refine_cfg = self.cfg["refine"]
        self.boundary_cfg = self.cfg["boundary"]
        self.num_classes = int(self.decoder_cfg["num_classes"])
        self.train_mode = self.sam_cfg["train_mode"]
        self.feature_mode = self.decoder_cfg["feature_mode"]
        self.fusion_mode = self.decoder_cfg["fusion_mode"]

        self.image_encoder = self._build_sam_image_encoder()
        self._set_sam_trainability()

        final_channels = self._infer_final_channels()
        token_channels = self._infer_token_channels()
        decoder_dim = int(self.decoder_cfg["dim"])
        if self.feature_mode == "final":
            self.decoder = PlainUpsampleDecoder(final_channels, decoder_dim, self.num_classes)
        elif self.feature_mode == "pseudo_pyramid":
            in_channels = [final_channels for _ in self.decoder_cfg["pseudo_scales"]]
            self.decoder = MultiScaleFusionDecoder(
                in_channels=in_channels,
                decoder_dim=decoder_dim,
                num_classes=self.num_classes,
                fuse_type=self.decoder_cfg.get("fuse_type", "sum"),
            )
        elif self.feature_mode == "multi_level":
            in_channels = [token_channels for _ in self.sam_cfg["intermediate_blocks"]]
            self.decoder = MultiScaleFusionDecoder(
                in_channels=in_channels,
                decoder_dim=decoder_dim,
                num_classes=self.num_classes,
                fuse_type=self.decoder_cfg.get("fuse_type", "sum"),
            )
        elif self.feature_mode == "dualdpt":
            in_channels = [token_channels for _ in self.sam_cfg["intermediate_blocks"]]
            self.decoder = DualDPTBoundaryDecoder(
                in_channels=in_channels,
                decoder_dim=decoder_dim,
                num_classes=self.num_classes,
                out_channels=self.decoder_cfg.get("dualdpt_out_channels", [96, 192, 384, 768]),
            )
        else:
            raise ValueError(f"Unsupported decoder.feature_mode: {self.feature_mode}")

        self.refiner = None
        if bool(self.refine_cfg.get("enabled", False)):
            if self.refine_cfg.get("type", "gru") != "gru":
                raise ValueError(f"Unsupported refine.type: {self.refine_cfg.get('type')}")
            self.refiner = GRURefiner(
                num_classes=self.num_classes,
                context_channels=decoder_dim,
                hidden_channels=decoder_dim,
                iters=int(self.refine_cfg.get("iters", 3)),
            )

    @classmethod
    def resolve_config(cls, model_cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = deepcopy(cls.DEFAULTS)
        setting = str(model_cfg.get("setting", cfg["setting"])).upper()
        if setting not in cls.PRESETS:
            raise ValueError(f"Unknown ReFuseNet setting: {setting}")
        explicit_decoder = model_cfg.get("decoder", {})
        explicit_fusion = isinstance(explicit_decoder, dict) and "fusion_mode" in explicit_decoder
        _deep_update(cfg, deepcopy(cls.PRESETS[setting]))
        _deep_update(cfg, deepcopy(model_cfg))
        cfg["setting"] = setting

        if "checkpoint" in model_cfg:
            cfg["sam"]["checkpoint"] = model_cfg["checkpoint"]
        if "freeze_encoder" in model_cfg:
            cfg["sam"]["train_mode"] = "frozen" if bool(model_cfg["freeze_encoder"]) else "low_lr_ft"
        if "num_classes" in model_cfg:
            cfg["decoder"]["num_classes"] = model_cfg["num_classes"]
        if cfg["decoder"]["num_classes"] is None:
            raise ValueError("ReFuseNet requires model.decoder.num_classes or model.num_classes.")

        cfg["sam"]["train_mode"] = str(cfg["sam"]["train_mode"]).lower()
        cfg["decoder"]["feature_mode"] = str(cfg["decoder"]["feature_mode"]).lower()
        cfg["decoder"]["fusion_mode"] = str(cfg["decoder"]["fusion_mode"]).lower()
        if cfg["sam"]["train_mode"] not in {"frozen", "low_lr_ft"}:
            raise ValueError("sam.train_mode must be 'frozen' or 'low_lr_ft'.")
        if cfg["decoder"]["feature_mode"] == "final":
            expected_fusion = "single"
        elif cfg["decoder"]["feature_mode"] == "dualdpt":
            expected_fusion = "dualdpt"
        else:
            expected_fusion = "multiscale"
        if not explicit_fusion:
            cfg["decoder"]["fusion_mode"] = expected_fusion
        if cfg["decoder"]["fusion_mode"] != expected_fusion:
            raise ValueError(
                f"decoder.fusion_mode={cfg['decoder']['fusion_mode']} is incompatible with "
                f"feature_mode={cfg['decoder']['feature_mode']}; expected {expected_fusion}."
            )
        cfg["sam"]["intermediate_blocks"] = [int(block) for block in cfg["sam"]["intermediate_blocks"]]
        cfg["decoder"]["pseudo_scales"] = [int(scale) for scale in cfg["decoder"]["pseudo_scales"]]
        cfg["decoder"]["dualdpt_out_channels"] = [int(ch) for ch in cfg["decoder"]["dualdpt_out_channels"]]
        if cfg["decoder"]["feature_mode"] == "dualdpt":
            cfg["boundary"]["enabled"] = True
            if len(cfg["sam"]["intermediate_blocks"]) != 4:
                raise ValueError("S6/DualDPT requires exactly four sam.intermediate_blocks.")
        return cfg

    def _build_sam_image_encoder(self) -> nn.Module:
        checkpoint = self.sam_cfg.get("checkpoint")
        if not checkpoint:
            raise FileNotFoundError("ReFuseNet requires model.sam.checkpoint or model.checkpoint.")
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        try:
            from segment_anything import sam_model_registry
        except Exception as exc:  # pragma: no cover
            raise ImportError("segment-anything is required for ReFuseNet.") from exc

        model_type = str(self.sam_cfg.get("model_type", "vit_b")).lower()
        if model_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM model_type '{model_type}'. Available: {sorted(sam_model_registry)}")
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        return sam.image_encoder

    def _set_sam_trainability(self) -> None:
        requires_grad = self.train_mode == "low_lr_ft"
        for param in self.image_encoder.parameters():
            param.requires_grad = requires_grad
        if not requires_grad:
            self.image_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_mode == "frozen":
            self.image_encoder.eval()
        return self

    def _infer_final_channels(self) -> int:
        for module in reversed(list(self.image_encoder.neck.modules())):
            if isinstance(module, nn.Conv2d):
                return int(module.out_channels)
        return 256

    def _infer_token_channels(self) -> int:
        patch_embed = getattr(self.image_encoder, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        if isinstance(proj, nn.Conv2d):
            return int(proj.out_channels)
        pos_embed = getattr(self.image_encoder, "pos_embed", None)
        if pos_embed is not None:
            return int(pos_embed.shape[-1])
        return self._infer_final_channels()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        image_size = int(self.sam_cfg.get("image_size", 1024))
        return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)

    def _run_sam_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.feature_mode in {"multi_level", "dualdpt"}:
            return self._run_sam_encoder_with_intermediates(x)
        return self.image_encoder(x), []

    def _run_sam_encoder_with_intermediates(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        blocks = set(self.sam_cfg["intermediate_blocks"])
        max_block = len(self.image_encoder.blocks)
        if any(block < 1 or block > max_block for block in blocks):
            raise ValueError(f"sam.intermediate_blocks must be in [1, {max_block}], got {sorted(blocks)}")

        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        intermediates: list[torch.Tensor] = []
        for index, block in enumerate(self.image_encoder.blocks, start=1):
            x = block(x)
            if index in blocks:
                intermediates.append(self._tokens_to_spatial(x))
        final_feature = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        return final_feature, intermediates

    @staticmethod
    def _tokens_to_spatial(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4D SAM block feature, got shape {tuple(feature.shape)}")
        channels_last = feature.shape[-1] >= feature.shape[1]
        if channels_last:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def _make_pseudo_pyramid(
        self,
        final_feature: torch.Tensor,
        output_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        features = []
        height, width = output_size
        for scale in self.decoder_cfg["pseudo_scales"]:
            size = (max(1, (height + scale - 1) // scale), max(1, (width + scale - 1) // scale))
            features.append(F.interpolate(final_feature, size=size, mode="bilinear", align_corners=False))
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | None | dict[str, Any]]:
        output_size = images.shape[-2:]
        sam_input = self._preprocess(images)
        with torch.set_grad_enabled(self.train_mode != "frozen" and torch.is_grad_enabled()):
            final_feature, intermediate_features = self._run_sam_encoder(sam_input)

        if self.feature_mode == "final":
            decoder_out = self.decoder(final_feature, output_size, return_features=self.refiner is not None or self.debug)
            decoder_inputs: torch.Tensor | list[torch.Tensor] = final_feature
        elif self.feature_mode == "pseudo_pyramid":
            decoder_inputs = self._make_pseudo_pyramid(final_feature, output_size)
            decoder_out = self.decoder(decoder_inputs, output_size, return_features=self.refiner is not None or self.debug)
        elif self.feature_mode == "multi_level":
            decoder_inputs = intermediate_features
            decoder_out = self.decoder(decoder_inputs, output_size, return_features=self.refiner is not None or self.debug)
        else:
            decoder_inputs = intermediate_features
            decoder_out = self.decoder(decoder_inputs, output_size, return_features=self.debug)

        coarse_logits = decoder_out["logits"]
        logits = coarse_logits
        returned_coarse = None
        if self.refiner is not None:
            logits = self.refiner(coarse_logits, decoder_out["decoder_features"])
            returned_coarse = coarse_logits

        out: dict[str, torch.Tensor | None | dict[str, Any]] = {
            "logits": logits,
            "coarse_logits": returned_coarse,
            "boundary_logits": decoder_out.get("boundary_logits"),
        }
        if self.debug:
            out["features"] = {
                "final": final_feature,
                "decoder_inputs": decoder_inputs,
                "decoder": decoder_out.get("decoder_features"),
                "boundary": decoder_out.get("boundary_features"),
            }
        return out

    def get_param_groups(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        train_cfg = cfg.get("train", {})
        lr_decoder = float(train_cfg.get("lr_decoder", train_cfg.get("lr", 1.0e-4)))
        lr_sam = float(train_cfg.get("lr_sam", lr_decoder))
        weight_decay = float(train_cfg.get("weight_decay", 0.0))

        groups: list[dict[str, Any]] = []
        sam_params = [param for param in self.image_encoder.parameters() if param.requires_grad]
        if sam_params:
            groups.append({"params": sam_params, "lr": lr_sam, "weight_decay": weight_decay, "name": "sam_image_encoder"})

        decoder_params = []
        for module in (self.decoder, self.refiner):
            if module is not None:
                decoder_params.extend(param for param in module.parameters() if param.requires_grad)
        groups.append({"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay, "name": "decoder"})
        return groups


ReFuseNet = RefuseNet
