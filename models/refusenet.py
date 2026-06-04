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


class LegacyPlainUpsampleDecoder(nn.Module):
    """S0 decoder: one final SAM image feature with the legacy private head."""

    def __init__(self, in_channels: int, decoder_dim: int, num_classes: int):
        super().__init__()
        self.out_channels = decoder_dim
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


class FinalFeatureSemanticDecoder(nn.Module):
    """S1 decoder: one final SAM image feature to a fixed semantic head grid."""

    def __init__(
        self,
        in_channels: int,
        decoder_dim: int,
        head_channels: int,
        head_resolution: tuple[int, int],
    ):
        super().__init__()
        self.out_channels = head_channels
        self.head_resolution = tuple(head_resolution)
        self.project = nn.Conv2d(in_channels, decoder_dim, kernel_size=1)
        self.block1 = ConvGNAct(decoder_dim, decoder_dim)
        self.block2 = ConvGNAct(decoder_dim, decoder_dim)
        self.neck = nn.Sequential(
            ConvGNAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, head_channels, kernel_size=1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        output_size: tuple[int, int],
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        del output_size, return_features
        x = self.project(feature)
        x = self.block1(x)
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, size=self.head_resolution, mode="bilinear", align_corners=False)
        return {"semantic_features": self.neck(x)}


class MultiScaleFusionDecoder(nn.Module):
    """S2/S3 decoder: project same-style feature lists and fuse them at one scale."""

    def __init__(
        self,
        in_channels: list[int],
        decoder_dim: int,
        head_channels: int,
        head_resolution: tuple[int, int],
        fuse_type: str = "sum",
    ):
        super().__init__()
        if fuse_type not in {"sum", "concat"}:
            raise ValueError(f"Unsupported fuse_type: {fuse_type}")
        self.out_channels = head_channels
        self.head_resolution = tuple(head_resolution)
        self.fuse_type = fuse_type
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_dim, kernel_size=1) for channels in in_channels]
        )
        fused_channels = decoder_dim if fuse_type == "sum" else decoder_dim * len(in_channels)
        self.fuse = nn.Sequential(
            ConvGNAct(fused_channels, decoder_dim),
            ConvGNAct(decoder_dim, decoder_dim),
        )
        self.neck = nn.Sequential(
            ConvGNAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, head_channels, kernel_size=1),
        )

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
        if x.shape[-2:] != self.head_resolution:
            x = F.interpolate(x, size=self.head_resolution, mode="bilinear", align_corners=False)
        return {"semantic_features": self.neck(x)}


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


class DPTReassembly(nn.Module):
    """Shared DA3/DPT-style projection, resize, and adapter stage for 4 SAM features."""

    def __init__(
        self,
        in_channels: list[int],
        decoder_dim: int,
        out_channels: list[int] | tuple[int, int, int, int],
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"DPTReassembly expects 4 input features, got {len(in_channels)}")
        if len(out_channels) != 4:
            raise ValueError(f"decoder.reassembly_channels must have 4 values, got {len(out_channels)}")

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

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"DPTReassembly expects 4 features, got {len(features)}")
        resized = []
        for feature, project, resize in zip(features, self.projects, self.resize_layers):
            resized.append(resize(project(feature)))
        return [adapter(feature) for adapter, feature in zip(self.adapters, resized)]


class DPTSemanticDecoder(nn.Module):
    """
    S4/S5 decoder: shared reassembly plus one semantic DPT top-down fusion branch.

    This intentionally has no boundary branch. Dual-branch boundary decoding is
    reserved for the S6 DualDPTBoundaryDecoder.
    """

    def __init__(
        self,
        in_channels: list[int],
        decoder_dim: int,
        head_channels: int,
        out_channels: list[int] | tuple[int, int, int, int] = (96, 192, 384, 768),
        dpt_head_scale: str = "p1",
    ):
        super().__init__()

        if dpt_head_scale != "p1":
            raise ValueError(f"Only decoder.dpt_head_scale='p1' is supported, got {dpt_head_scale}")
        self.out_channels = head_channels
        self.reassemble = DPTReassembly(in_channels, decoder_dim, out_channels)
        self.refine4 = DPTFusionBlock(decoder_dim, has_residual=False)
        self.refine3 = DPTFusionBlock(decoder_dim)
        self.refine2 = DPTFusionBlock(decoder_dim)
        self.refine1 = DPTFusionBlock(decoder_dim)
        self.neck = nn.Sequential(
            nn.Conv2d(decoder_dim, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(head_channels), head_channels),
            nn.GELU(),
        )

    def _fuse_semantic(self, features: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = features
        x = self.refine4(l4, size=l3.shape[-2:])
        x = self.refine3(x, l3, size=l2.shape[-2:])
        x = self.refine2(x, l2, size=l1.shape[-2:])
        return self.refine1(x, l1, size=l1.shape[-2:])

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        del output_size, return_features
        features = self.reassemble(features)
        semantic_features = self.neck(self._fuse_semantic(features))
        return {"semantic_features": semantic_features}


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
        head_channels: int,
        out_channels: list[int] | tuple[int, int, int, int] = (96, 192, 384, 768),
        dpt_head_scale: str = "p1",
    ):
        super().__init__()

        if dpt_head_scale != "p1":
            raise ValueError(f"Only decoder.dpt_head_scale='p1' is supported, got {dpt_head_scale}")
        self.out_channels = head_channels
        self.reassemble = DPTReassembly(in_channels, decoder_dim, out_channels)
        self.sem_refine4 = DPTFusionBlock(decoder_dim, has_residual=False)
        self.sem_refine3 = DPTFusionBlock(decoder_dim)
        self.sem_refine2 = DPTFusionBlock(decoder_dim)
        self.sem_refine1 = DPTFusionBlock(decoder_dim)
        self.sem_neck = nn.Sequential(
            nn.Conv2d(decoder_dim, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(head_channels), head_channels),
            nn.GELU(),
        )

        self.boundary_refine4 = DPTFusionBlock(decoder_dim, has_residual=False)
        self.boundary_refine3 = DPTFusionBlock(decoder_dim)
        self.boundary_refine2 = DPTFusionBlock(decoder_dim)
        self.boundary_refine1 = DPTFusionBlock(decoder_dim)
        self.boundary_neck = nn.Sequential(
            ConvGNAct(decoder_dim, head_channels),
            ConvGNAct(head_channels, head_channels),
        )
        self.boundary_head = nn.Conv2d(head_channels, 1, kernel_size=1)

    def _fuse_semantic(self, features: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = features
        x = self.sem_refine4(l4, size=l3.shape[-2:])
        x = self.sem_refine3(x, l3, size=l2.shape[-2:])
        x = self.sem_refine2(x, l2, size=l1.shape[-2:])
        return self.sem_refine1(x, l1, size=l1.shape[-2:])

    def _fuse_boundary(self, features: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = features
        x = self.boundary_refine4(l4, size=l3.shape[-2:])
        x = self.boundary_refine3(x, l3, size=l2.shape[-2:])
        x = self.boundary_refine2(x, l2, size=l1.shape[-2:])
        return self.boundary_refine1(x, l1, size=l1.shape[-2:])

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_features
        features = self.reassemble(features)
        semantic_features = self.sem_neck(self._fuse_semantic(features))
        boundary_features = self.boundary_neck(self._fuse_boundary(features))

        boundary_logits = self.boundary_head(boundary_features)
        boundary_logits = F.interpolate(boundary_logits, size=output_size, mode="bilinear", align_corners=False)

        return {"semantic_features": semantic_features, "boundary_logits": boundary_logits}


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
    """S5 refinement: shared iterative hidden-state updates on head-resolution logits."""

    def __init__(
        self,
        num_classes: int,
        context_channels: int,
        hidden_channels: int = 64,
        iters: int = 3,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.iters = int(iters)
        self.delta_scale = float(delta_scale)
        self.logits_to_hidden = nn.Conv2d(num_classes, hidden_channels, kernel_size=3, padding=1)
        self.context_proj = nn.Conv2d(context_channels, hidden_channels, kernel_size=1)
        self.cell = ConvGRUCell(hidden_channels, hidden_channels)
        self.delta_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, coarse_logits: torch.Tensor, context: torch.Tensor) -> list[torch.Tensor]:
        if coarse_logits.shape[-2:] != context.shape[-2:]:
            raise ValueError(
                "GRURefiner expects head-resolution coarse logits and context with matching spatial size, "
                f"got {tuple(coarse_logits.shape[-2:])} and {tuple(context.shape[-2:])}."
            )
        h = self.logits_to_hidden(coarse_logits)
        x = self.context_proj(context)
        logits = coarse_logits
        refined_logits = []
        for _ in range(self.iters):
            h = self.cell(x, h)
            logits = logits + self.delta_scale * self.delta_head(h)
            refined_logits.append(logits)
        return refined_logits


class RefuseNet(nn.Module):
    """
    Single configurable ReFuseNet/RefuseNet ablation class.

    S0: frozen SAM image encoder + final feature + plain decoder.
    S1: low-LR SAM fine-tune + final feature + plain decoder.
    S2: low-LR SAM fine-tune + pseudo 4-scale features + naive multiscale decoder.
    S3: low-LR SAM fine-tune + true 4-level SAM features + naive multiscale decoder.
    S4: low-LR SAM fine-tune + true 4-level SAM features + DPT semantic decoder.
    S5: S4 plus iterative GRU refinement and coarse-logit auxiliary loss.
    S6: DA3-style DualDPT decoder plus a PIDNet-style boundary head.
    S7: S2 plus iterative GRU refinement.
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
            "decoder": {"feature_mode": "multi_level", "fusion_mode": "dpt"},
            "refine": {"enabled": False},
        },
        "S5": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "multi_level", "fusion_mode": "dpt"},
            "refine": {"enabled": True, "type": "gru"},
        },
        "S6": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "dualdpt", "fusion_mode": "dualdpt"},
            "refine": {"enabled": False},
            "boundary": {"enabled": True},
        },
        "S7": {
            "sam": {"train_mode": "low_lr_ft"},
            "decoder": {"feature_mode": "pseudo_pyramid", "fusion_mode": "multiscale"},
            "refine": {"enabled": True, "type": "gru"},
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
            "head_channels": 64,
            "head_resolution": [256, 256],
            "feature_mode": "final",
            "fusion_mode": "single",
            "pseudo_scales": [4, 8, 16, 32],
            "fuse_type": "sum",
            "reassembly_channels": [96, 192, 384, 768],
            "dpt_head_scale": "p1",
        },
        "refine": {
            "enabled": False,
            "type": "gru",
            "iters": 3,
            "hidden_dim": 64,
            "delta_scale": 1.0,
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
        self.setting = str(self.cfg["setting"]).upper()
        self.train_mode = self.sam_cfg["train_mode"]
        self.feature_mode = self.decoder_cfg["feature_mode"]
        self.fusion_mode = self.decoder_cfg["fusion_mode"]

        self.image_encoder = self._build_sam_image_encoder()
        self._set_sam_trainability()

        final_channels = self._infer_final_channels()
        token_channels = self._infer_token_channels()
        decoder_dim = int(self.decoder_cfg["dim"])
        self.head_channels = int(self.decoder_cfg["head_channels"])
        self.head_resolution = tuple(int(v) for v in self.decoder_cfg["head_resolution"])
        if len(self.head_resolution) != 2:
            raise ValueError(f"decoder.head_resolution must have two values, got {self.decoder_cfg['head_resolution']}")

        self.classifier = None if self.setting == "S0" else nn.Conv2d(self.head_channels, self.num_classes, kernel_size=1)

        if self.feature_mode == "final":
            if self.setting == "S0":
                self.decoder = LegacyPlainUpsampleDecoder(final_channels, decoder_dim, self.num_classes)
            else:
                self.decoder = FinalFeatureSemanticDecoder(
                    in_channels=final_channels,
                    decoder_dim=decoder_dim,
                    head_channels=self.head_channels,
                    head_resolution=self.head_resolution,
                )
        elif self.feature_mode == "pseudo_pyramid":
            in_channels = [final_channels for _ in self.decoder_cfg["pseudo_scales"]]
            self.decoder = MultiScaleFusionDecoder(
                in_channels=in_channels,
                decoder_dim=decoder_dim,
                head_channels=self.head_channels,
                head_resolution=self.head_resolution,
                fuse_type=self.decoder_cfg.get("fuse_type", "sum"),
            )
        elif self.feature_mode == "multi_level":
            in_channels = [token_channels for _ in self.sam_cfg["intermediate_blocks"]]
            if self.fusion_mode == "multiscale":
                self.decoder = MultiScaleFusionDecoder(
                    in_channels=in_channels,
                    decoder_dim=decoder_dim,
                    head_channels=self.head_channels,
                    head_resolution=self.head_resolution,
                    fuse_type=self.decoder_cfg.get("fuse_type", "sum"),
                )
            elif self.fusion_mode == "dpt":
                self.decoder = DPTSemanticDecoder(
                    in_channels=in_channels,
                    decoder_dim=decoder_dim,
                    head_channels=self.head_channels,
                    out_channels=self.decoder_cfg["reassembly_channels"],
                    dpt_head_scale=self.decoder_cfg.get("dpt_head_scale", "p1"),
                )
            else:
                raise ValueError(
                    "decoder.fusion_mode must be 'multiscale' or 'dpt' when feature_mode is 'multi_level'."
                )
        elif self.feature_mode == "dualdpt":
            in_channels = [token_channels for _ in self.sam_cfg["intermediate_blocks"]]
            self.decoder = DualDPTBoundaryDecoder(
                in_channels=in_channels,
                decoder_dim=decoder_dim,
                head_channels=self.head_channels,
                out_channels=self.decoder_cfg["reassembly_channels"],
                dpt_head_scale=self.decoder_cfg.get("dpt_head_scale", "p1"),
            )
        else:
            raise ValueError(f"Unsupported decoder.feature_mode: {self.feature_mode}")

        self.refiner = None
        if bool(self.refine_cfg.get("enabled", False)):
            if self.refine_cfg.get("type", "gru") != "gru":
                raise ValueError(f"Unsupported refine.type: {self.refine_cfg.get('type')}")
            self.refiner = GRURefiner(
                num_classes=self.num_classes,
                context_channels=self.head_channels,
                hidden_channels=int(self.refine_cfg.get("hidden_dim", 64)),
                iters=int(self.refine_cfg.get("iters", 3)),
                delta_scale=float(self.refine_cfg.get("delta_scale", 1.0)),
            )

    @classmethod
    def resolve_config(cls, model_cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = deepcopy(cls.DEFAULTS)
        setting = str(model_cfg.get("setting", cfg["setting"])).upper()
        if setting not in cls.PRESETS:
            raise ValueError(f"Unknown ReFuseNet setting: {setting}")
        explicit_decoder = model_cfg.get("decoder", {})
        explicit_feature = isinstance(explicit_decoder, dict) and "feature_mode" in explicit_decoder
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
        feature_mode = cfg["decoder"]["feature_mode"]
        fusion_mode = cfg["decoder"]["fusion_mode"]
        if feature_mode not in {"final", "pseudo_pyramid", "multi_level", "dualdpt"}:
            raise ValueError(f"Unsupported decoder.feature_mode: {feature_mode}")
        allowed_fusions = {
            "final": {"single"},
            "pseudo_pyramid": {"multiscale"},
            "multi_level": {"multiscale", "dpt"},
            "dualdpt": {"dualdpt"},
        }
        default_fusion = {
            "final": "single",
            "pseudo_pyramid": "multiscale",
            "multi_level": "multiscale",
            "dualdpt": "dualdpt",
        }[feature_mode]
        if explicit_feature and not explicit_fusion:
            cfg["decoder"]["fusion_mode"] = default_fusion
            fusion_mode = default_fusion
        if fusion_mode not in allowed_fusions[feature_mode]:
            raise ValueError(
                f"decoder.fusion_mode={cfg['decoder']['fusion_mode']} is incompatible with "
                f"feature_mode={cfg['decoder']['feature_mode']}; expected one of "
                f"{sorted(allowed_fusions[feature_mode])}."
            )
        cfg["sam"]["intermediate_blocks"] = [int(block) for block in cfg["sam"]["intermediate_blocks"]]
        cfg["decoder"]["pseudo_scales"] = [int(scale) for scale in cfg["decoder"]["pseudo_scales"]]
        cfg["decoder"]["reassembly_channels"] = [int(ch) for ch in cfg["decoder"]["reassembly_channels"]]
        cfg["decoder"]["head_channels"] = int(cfg["decoder"]["head_channels"])
        cfg["decoder"]["head_resolution"] = [int(v) for v in cfg["decoder"]["head_resolution"]]
        cfg["decoder"]["dpt_head_scale"] = str(cfg["decoder"].get("dpt_head_scale", "p1")).lower()
        cfg["refine"]["iters"] = int(cfg["refine"].get("iters", 3))
        cfg["refine"]["hidden_dim"] = int(cfg["refine"].get("hidden_dim", 64))
        cfg["refine"]["delta_scale"] = float(cfg["refine"].get("delta_scale", 1.0))
        if cfg["decoder"]["head_channels"] <= 0:
            raise ValueError("decoder.head_channels must be positive.")
        if len(cfg["decoder"]["head_resolution"]) != 2 or any(v <= 0 for v in cfg["decoder"]["head_resolution"]):
            raise ValueError("decoder.head_resolution must contain two positive integers.")
        if cfg["decoder"]["dpt_head_scale"] != "p1":
            raise ValueError("decoder.dpt_head_scale currently supports only 'p1'.")
        if cfg["refine"]["iters"] < 0:
            raise ValueError("refine.iters must be non-negative.")
        if cfg["refine"]["hidden_dim"] <= 0:
            raise ValueError("refine.hidden_dim must be positive.")
        if cfg["decoder"]["feature_mode"] in {"multi_level", "dualdpt"}:
            if len(cfg["sam"]["intermediate_blocks"]) != 4:
                raise ValueError(f"{cfg['decoder']['fusion_mode']}/DPT requires exactly four sam.intermediate_blocks.")
        if cfg["decoder"]["feature_mode"] == "dualdpt":
            cfg["boundary"]["enabled"] = True
        elif bool(cfg["boundary"].get("enabled", False)):
            raise ValueError("boundary.enabled is only supported when decoder.feature_mode is 'dualdpt'.")
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
        processed, _ = self._preprocess_with_meta(x)
        return processed

    def _preprocess_with_meta(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, tuple[int, int] | tuple[int, int, int, int]]]:
        image_size = int(self.sam_cfg.get("image_size", 1024))
        if image_size <= 0:
            raise ValueError(f"sam.image_size must be positive, got {image_size}")
        orig_size = (int(x.shape[-2]), int(x.shape[-1]))
        scale = image_size / max(orig_size)
        resized_size = (
            min(image_size, max(1, int(round(orig_size[0] * scale)))),
            min(image_size, max(1, int(round(orig_size[1] * scale)))),
        )
        resized = F.interpolate(x, size=resized_size, mode="bilinear", align_corners=False)
        pad_h = image_size - resized_size[0]
        pad_w = image_size - resized_size[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Internal SAM preprocessing error: resized size {resized_size} exceeds target {image_size}."
            )
        processed = F.pad(resized, (0, pad_w, 0, pad_h)) if pad_h or pad_w else resized
        meta = {
            "orig_size": orig_size,
            "resized_size": resized_size,
            "pad": (0, pad_w, 0, pad_h),
            "processed_size": (image_size, image_size),
        }
        return processed, meta

    @staticmethod
    def _restore_logits(logits: torch.Tensor, meta: dict[str, tuple[int, int] | tuple[int, int, int, int]]) -> torch.Tensor:
        resized_size = tuple(meta["resized_size"])  # type: ignore[arg-type]
        orig_size = tuple(meta["orig_size"])  # type: ignore[arg-type]
        logits = logits[:, :, : resized_size[0], : resized_size[1]]
        if logits.shape[-2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
        return logits

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
    ) -> list[torch.Tensor]:
        features = []
        height, width = self.head_resolution
        for scale in (1, 2, 4, 8):
            size = (max(1, height // scale), max(1, width // scale))
            features.append(F.interpolate(final_feature, size=size, mode="bilinear", align_corners=False))
        return features

    def _make_intermediate_pyramid(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        height, width = self.head_resolution
        pyramid = []
        for feature, scale in zip(features, (1, 2, 4, 8)):
            size = (max(1, height // scale), max(1, width // scale))
            pyramid.append(F.interpolate(feature, size=size, mode="bilinear", align_corners=False))
        return pyramid

    def _assert_head_features(self, semantic_features: torch.Tensor) -> None:
        expected = (self.head_channels, *self.head_resolution)
        actual = (semantic_features.shape[1], *semantic_features.shape[-2:])
        if actual != expected:
            raise RuntimeError(
                f"{self.setting} classifier input must be [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got [B,{actual[0]},{actual[1]},{actual[2]}]."
            )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | None | dict[str, Any]]:
        sam_input, preprocess_meta = self._preprocess_with_meta(images)
        output_size = tuple(preprocess_meta["processed_size"])  # type: ignore[arg-type]
        with torch.set_grad_enabled(self.train_mode != "frozen" and torch.is_grad_enabled()):
            final_feature, intermediate_features = self._run_sam_encoder(sam_input)

        if self.feature_mode == "final":
            decoder_out = self.decoder(final_feature, output_size, return_features=self.refiner is not None or self.debug)
            decoder_inputs: torch.Tensor | list[torch.Tensor] = final_feature
        elif self.feature_mode == "pseudo_pyramid":
            decoder_inputs = self._make_pseudo_pyramid(final_feature)
            decoder_out = self.decoder(decoder_inputs, output_size, return_features=self.refiner is not None or self.debug)
        elif self.feature_mode == "multi_level":
            decoder_inputs = (
                self._make_intermediate_pyramid(intermediate_features)
                if self.fusion_mode == "multiscale"
                else intermediate_features
            )
            decoder_out = self.decoder(decoder_inputs, output_size, return_features=self.refiner is not None or self.debug)
        else:
            decoder_inputs = intermediate_features
            decoder_out = self.decoder(decoder_inputs, output_size, return_features=self.debug)

        if self.setting == "S0":
            logits = self._restore_logits(decoder_out["logits"], preprocess_meta)
            out: dict[str, torch.Tensor | None | dict[str, Any]] = {"logits": logits}
            if self.debug:
                out["features"] = {
                    "final": final_feature,
                    "decoder_inputs": decoder_inputs,
                    "decoder": decoder_out.get("decoder_features"),
                }
            return out

        semantic_features = decoder_out["semantic_features"]
        self._assert_head_features(semantic_features)
        if self.classifier is None:
            raise RuntimeError(f"{self.setting} requires a shared classifier.")
        coarse_logits_head = self.classifier(semantic_features)

        returned_coarse = None
        aux_logits_head: list[torch.Tensor] = []
        aux_logits_full: list[torch.Tensor] = []
        logits_head = coarse_logits_head
        if self.refiner is not None:
            aux_logits_head = self.refiner(coarse_logits_head, semantic_features)
            if aux_logits_head:
                logits_head = aux_logits_head[-1]
            returned_coarse = self._restore_logits(
                F.interpolate(coarse_logits_head, size=output_size, mode="bilinear", align_corners=False),
                preprocess_meta,
            )
            aux_logits_full = [
                self._restore_logits(
                    F.interpolate(aux_head, size=output_size, mode="bilinear", align_corners=False),
                    preprocess_meta,
                )
                for aux_head in aux_logits_head
            ]

        logits = self._restore_logits(
            F.interpolate(logits_head, size=output_size, mode="bilinear", align_corners=False),
            preprocess_meta,
        )

        out: dict[str, torch.Tensor | None | dict[str, Any]] = {"logits": logits}
        if returned_coarse is not None:
            out["coarse_logits"] = returned_coarse
            out["aux_logits"] = aux_logits_full
            out["coarse_logits_head"] = coarse_logits_head
            out["aux_logits_head"] = aux_logits_head
        if "boundary_logits" in decoder_out:
            out["boundary_logits"] = self._restore_logits(decoder_out["boundary_logits"], preprocess_meta)
        if self.debug:
            out["features"] = {
                "final": final_feature,
                "decoder_inputs": decoder_inputs,
                "decoder": semantic_features,
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
        for module in (self.decoder, self.refiner, self.classifier):
            if module is not None:
                decoder_params.extend(param for param in module.parameters() if param.requires_grad)
        groups.append({"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay, "name": "decoder"})
        return groups


ReFuseNet = RefuseNet
