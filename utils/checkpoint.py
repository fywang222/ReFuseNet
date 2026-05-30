from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        stripped[key] = value
    return stripped


def save_checkpoint(path, model, optimizer, epoch, metrics, extra=None, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "epoch_completed": int(epoch) + 1,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer=None, strict=False, match_shape=True, verbose=False, scaler=None):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if isinstance(state_dict, dict):
        state_dict = _strip_module_prefix(state_dict)
    else:
        raise TypeError(f"Unsupported checkpoint format in {path}")

    model_state = _unwrap_model(model).state_dict()
    if match_shape:
        filtered = {}
        loaded, skipped = [], []
        for key, value in state_dict.items():
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
                filtered[key] = value
                loaded.append(key)
            else:
                skipped.append(key)
        _unwrap_model(model).load_state_dict(filtered, strict=False)
        print(f"[checkpoint] loaded tensors: {len(loaded)}")
        print(f"[checkpoint] skipped tensors: {len(skipped)}")
        if verbose and skipped:
            for key in skipped:
                print(f"[checkpoint] skipped: {key}")
    else:
        missing, unexpected = _unwrap_model(model).load_state_dict(state_dict, strict=strict)
        print(f"[checkpoint] loaded checkpoint from {path}")
        if verbose:
            if missing:
                print(f"[checkpoint] missing keys: {missing}")
            if unexpected:
                print(f"[checkpoint] unexpected keys: {unexpected}")

    if optimizer is not None and checkpoint.get("optimizer") is not None and not match_shape:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint.get("scaler") is not None and not match_shape:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint
