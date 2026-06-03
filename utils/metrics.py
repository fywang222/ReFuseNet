from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class SegMetric:
    def __init__(self, num_classes, ignore_index=255, class_names: Optional[List[str]] = None, rare_class_names=None):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = list(class_names) if class_names is not None else None
        self.rare_class_names = list(rare_class_names) if rare_class_names is not None else None
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @staticmethod
    def _to_pred(logits_or_pred):
        if torch.is_tensor(logits_or_pred):
            if logits_or_pred.ndim == 4:
                return logits_or_pred.argmax(dim=1)
            return logits_or_pred
        raise TypeError("SegMetric expects torch.Tensor inputs.")

    def update(self, logits_or_pred, target):
        pred = self._to_pred(logits_or_pred).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        if pred.ndim == 2:
            pred = pred[None, ...]
        if target.ndim == 2:
            target = target[None, ...]

        for p, t in zip(pred, target):
            valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
            p = p[valid]
            t = t[valid]
            if t.size == 0:
                continue
            inds = self.num_classes * t.astype(np.int64) + p.astype(np.int64)
            hist = np.bincount(inds, minlength=self.num_classes * self.num_classes)
            self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self):
        tp = np.diag(self.confusion).astype(np.float64)
        pos_gt = self.confusion.sum(axis=1).astype(np.float64)
        pos_pred = self.confusion.sum(axis=0).astype(np.float64)
        union = pos_gt + pos_pred - tp
        ious = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        acc = np.divide(tp, pos_gt, out=np.zeros_like(tp), where=pos_gt > 0)
        valid_iou = union > 0
        valid_acc = pos_gt > 0

        pixel_acc = float(tp.sum() / max(self.confusion.sum(), 1))
        mean_acc = float(acc[valid_acc].mean()) if valid_acc.any() else 0.0
        miou = float(ious[valid_iou].mean()) if valid_iou.any() else 0.0
        result = {
            "miou": miou,
            "ious": ious.tolist(),
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
        }

        if self.class_names is not None and self.rare_class_names:
            rare_ids = [self.class_names.index(name) for name in self.rare_class_names if name in self.class_names]
            if rare_ids:
                rare_valid = valid_iou[rare_ids]
                rare_iou = ious[rare_ids]
                result["rare_miou"] = float(rare_iou[rare_valid].mean()) if rare_valid.any() else 0.0
        return result
