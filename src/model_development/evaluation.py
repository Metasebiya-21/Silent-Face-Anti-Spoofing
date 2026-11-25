"""Score a trained MiniFASNet checkpoint on a labelled folder (validation or test split).

Folder layout, same as training: <data_dir>/{0,1,2}/*.png, already cropped to the model's patch type.
Label 1 is a real face; 0 and 2 are attack types.

Metrics (ISO/IEC 30107-3 names):
- APCER: share of attacks accepted as real     (security: lower is safer)
- BPCER: share of real faces rejected          (usability: lower is friendlier)
- ACER:  (APCER + BPCER) / 2, the single number used to pick a checkpoint
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from liveness_service.ml.inference.predictor import AntiSpoofModel

REAL_LABEL = 1


def metrics(y_true, y_pred, num_classes=3):
    """Accuracy, APCER, BPCER, ACER and the confusion matrix (rows: true label, columns: predicted)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    confusion = np.zeros((num_classes, num_classes), int)
    np.add.at(confusion, (y_true, y_pred), 1)
    real, accepted = y_true == REAL_LABEL, y_pred == REAL_LABEL
    apcer = float(accepted[~real].mean()) if (~real).any() else float("nan")
    bpcer = float((~accepted[real]).mean()) if real.any() else float("nan")
    return {"samples": int(len(y_true)), "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
            "apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2, "confusion_matrix": confusion.tolist()}


def predict_folder(model, data_dir):
    """(true labels, predicted labels) for every readable image under data_dir/<label>/."""
    y_true, y_pred = [], []
    for label_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir() and p.name.isdigit()):
        for path in sorted(label_dir.iterdir()):
            image = cv2.imread(str(path))
            if image is None:
                continue  # not an image (e.g. .DS_Store)
            if image.shape[:2] != (model.height, model.width):
                image = cv2.resize(image, (model.width, model.height))
            y_true.append(int(label_dir.name))
            y_pred.append(int(np.argmax(model.predict(image))))
    return y_true, y_pred


def evaluate(model_path, data_dir, device="cpu"):
    """Metrics of one checkpoint on one split."""
    model = AntiSpoofModel(model_path, torch.device(device))
    return {"checkpoint": Path(model_path).name, **metrics(*predict_folder(model, data_dir))}


def evaluate_checkpoints(snapshot_dir, data_dir, device="cpu"):
    """Metrics of every .pth in snapshot_dir, best (lowest ACER) first."""
    results = [evaluate(path, data_dir, device) for path in sorted(Path(snapshot_dir).glob("*.pth"))]
    return sorted(results, key=lambda r: (np.nan_to_num(r["acer"], nan=1.0), r["checkpoint"]))
